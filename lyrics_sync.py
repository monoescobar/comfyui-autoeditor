"""
Lyrics alignment engine using OpenAI Whisper.

Provides word-level timestamps by forced-aligning provided lyrics text
to audio using Whisper's word-level transcription, plus BPM detection
for animation speed control.
"""

import os
import re
import math
import difflib
import unicodedata
import torch
import numpy as np

# ── Model cache ──────────────────────────────────────────────────────────────
_whisper_model = None
_whisper_model_name = None


def unload_whisper():
    """Free Whisper model from VRAM."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        _whisper_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[LyricsSync] 🗑️ Whisper model unloaded")


# ── Main alignment function ─────────────────────────────────────────────────

def _audio_duration_seconds(audio_dict):
    """Return audio duration from a ComfyUI AUDIO dict without heavy conversion."""
    try:
        waveform = audio_dict["waveform"]
        sample_rate = float(audio_dict["sample_rate"])
        if sample_rate <= 0:
            return 0.0
        return float(waveform.shape[-1]) / sample_rate
    except Exception:
        return 0.0


def _normalize_word(text):
    """Normalize lyrics/transcript words for matching, including accents."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^\w']", "", text, flags=re.UNICODE)


_LYRIC_METADATA_RE = re.compile(
    r"^[\[<{]\s*(?:intro|verse|pre[- ]?chorus|chorus|post[- ]?chorus|hook|bridge|outro|"
    r"instrumental|music|break|interlude|drop|refrain|repeat|end|fade|solo|spoken|"
    r"whispered|female vocal|male vocal)(?:[\s\d:_-].*)?[\]}>]$",
    re.IGNORECASE,
)


def _is_lyrics_metadata_line(line):
    """Return True for song-structure labels that should never appear on screen."""
    return bool(_LYRIC_METADATA_RE.fullmatch(str(line).strip()))


def _lyrics_initial_prompt(lyrics_text, max_chars=1000):
    """Give Whisper a short lyric hint so song transcription follows the supplied words."""
    cleaned = " ".join(
        line.strip()
        for line in lyrics_text.splitlines()
        if line.strip() and not _is_lyrics_metadata_line(line)
    )
    return cleaned[:max_chars] if cleaned else None


def _transcribe_with_lyrics_hint(model, audio_np, lyrics_text):
    kwargs = {
        "word_timestamps": True,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }
    prompt = _lyrics_initial_prompt(lyrics_text)
    if prompt:
        kwargs["initial_prompt"] = prompt
    try:
        return model.transcribe(audio_np, **kwargs)
    except TypeError:
        kwargs.pop("initial_prompt", None)
        kwargs.pop("condition_on_previous_text", None)
        return model.transcribe(audio_np, **kwargs)


def _apply_timing_offset(aligned_lines, timing_offset_ms):
    """Shift all lyric timestamps by a manual offset and keep them non-negative."""
    offset_s = timing_offset_ms / 1000.0
    if abs(offset_s) <= 0.001:
        return
    for line in aligned_lines:
        line["line_start"] = max(0, line["line_start"] + offset_s)
        line["line_end"] = max(line["line_start"] + 0.01, line["line_end"] + offset_s)
        for w in line["words"]:
            w["start"] = max(0, w["start"] + offset_s)
            w["end"] = max(w["start"] + 0.01, w["end"] + offset_s)
    print(f"[LyricsSync] Applied offset: {timing_offset_ms:+d}ms")


def fallback_align_lyrics(
    lyrics_text,
    duration,
    timing_offset_ms=0,
    active_start=None,
    active_end=None,
    fallback_reason="estimated",
):
    """
    Estimate clean lyric timings when Whisper is unavailable.

    This keeps the Lyrics Overlay useful even if the Whisper model is missing,
    corrupted, downloading, or blocked by network issues.
    """
    lines = _parse_lyrics(lyrics_text)
    if not lines:
        return []

    duration = float(duration or 0.0)
    duration = max(0.1, duration) if duration > 0 else max(0.1, len(lines) * 0.55)
    start_pad = min(0.45, duration * 0.06)
    end_pad = min(0.45, duration * 0.06)
    if active_start is None or active_end is None:
        usable_start = start_pad
        usable_end = max(usable_start + 0.1, duration - end_pad)
    else:
        usable_start = max(0.0, min(float(active_start), duration - 0.1))
        usable_end = max(usable_start + 0.1, min(float(active_end), duration))
        minimum_window = min(duration, max(0.8, len(lines) * 0.35))
        if usable_end - usable_start < minimum_window:
            usable_start = start_pad
            usable_end = max(usable_start + 0.1, duration - end_pad)
    usable_duration = max(0.1, usable_end - usable_start)
    line_gap = min(0.12, usable_duration * 0.015)
    gap_total = line_gap * max(0, len(lines) - 1)
    lyric_duration = max(0.1, usable_duration - gap_total)

    weights = [max(1, len(line["words"])) for line in lines]
    total_weight = max(1, sum(weights))
    cursor = usable_start
    aligned = []

    for line, weight in zip(lines, weights):
        line_duration = max(0.1, lyric_duration * weight / total_weight)
        line_start = cursor
        line_end = min(usable_end, cursor + line_duration)
        words = line["words"]
        word_duration = max(0.01, (line_end - line_start) / max(1, len(words)))
        line_words = []

        for idx, word in enumerate(words):
            word_start = line_start + idx * word_duration
            word_end = line_start + (idx + 1) * word_duration
            line_words.append({
                "word": word,
                "start": word_start,
                "end": max(word_start + 0.01, word_end),
                "interpolated": True,
                "estimated": True,
            })

        aligned.append({
            "text": line["text"],
            "line_start": line_words[0]["start"] if line_words else line_start,
            "line_end": line_words[-1]["end"] if line_words else line_end,
            "words": line_words,
            "fallback_timing": True,
            "timing_source": fallback_reason,
        })
        cursor = line_end + line_gap

    _apply_timing_offset(aligned, timing_offset_ms)
    _sanitize_aligned_timings(aligned, duration)
    print(f"[LyricsSync] Stable timing fallback ({fallback_reason}): {len(aligned)} lines over {usable_start:.1f}-{usable_end:.1f}s")
    return aligned


def _align_lyrics_whisper_legacy(audio_dict, lyrics_text, model_size="base", timing_offset_ms=0):
    """
    Align lyrics text to song audio using Whisper word-level timestamps.

    Args:
        audio_dict: ComfyUI AUDIO dict {waveform, sample_rate}
        lyrics_text: Full lyrics text with line breaks
        model_size: Whisper model size (tiny/base/small/medium)
        timing_offset_ms: Manual offset in milliseconds

    Returns:
        List of line dicts: [{
            "text": "full line text",
            "line_start": float,
            "line_end": float,
            "words": [{"word": str, "start": float, "end": float}, ...]
        }, ...]
    """
    global _whisper_model, _whisper_model_name

    # ── Prepare audio for Whisper (mono float32 @ 16kHz) ─────────────
    audio_np = _prepare_audio(audio_dict)
    duration = len(audio_np) / 16000.0
    print(f"[LyricsSync] Audio duration: {duration:.1f}s")

    # ── Load Whisper model ───────────────────────────────────────────
    import whisper
    if _whisper_model is None or _whisper_model_name != model_size:
        print(f"[LyricsSync] Loading Whisper '{model_size}' model...")
        _whisper_model = whisper.load_model(model_size)
        _whisper_model_name = model_size
        print(f"[LyricsSync] ✅ Whisper '{model_size}' loaded")

    # ── Transcribe with word timestamps ──────────────────────────────
    print(f"[LyricsSync] 🎤 Transcribing audio...")
    result = _transcribe_with_lyrics_hint(_whisper_model, audio_np, lyrics_text)

    # Extract word timestamps
    whisper_words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word_text = w.get("word", "").strip()
            if word_text:
                whisper_words.append({
                    "word": word_text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

    print(f"[LyricsSync] Whisper detected {len(whisper_words)} words")

    # ── Parse provided lyrics ────────────────────────────────────────
    lines = _parse_lyrics(lyrics_text)
    total_lyrics_words = sum(len(ln["words"]) for ln in lines)
    print(f"[LyricsSync] Provided lyrics: {len(lines)} lines, {total_lyrics_words} words")

    # ── Align lyrics to Whisper output ───────────────────────────────
    aligned_lines = _align_to_whisper(lines, whisper_words, duration)

    # ── Apply timing offset ──────────────────────────────────────────
    offset_s = timing_offset_ms / 1000.0
    if abs(offset_s) > 0.001:
        for line in aligned_lines:
            line["line_start"] = max(0, line["line_start"] + offset_s)
            line["line_end"] = max(0, line["line_end"] + offset_s)
            for w in line["words"]:
                w["start"] = max(0, w["start"] + offset_s)
                w["end"] = max(0, w["end"] + offset_s)
        print(f"[LyricsSync] Applied offset: {timing_offset_ms:+d}ms")

    # ── Summary ──────────────────────────────────────────────────────
    matched = sum(1 for ln in aligned_lines for w in ln["words"] if not w.get("interpolated"))
    print(f"[LyricsSync] ✅ Aligned {matched}/{total_lyrics_words} words directly, "
          f"rest interpolated")

    return aligned_lines


# ── Audio preparation ────────────────────────────────────────────────────────

def _prepare_audio(audio_dict):
    """Convert ComfyUI AUDIO dict to mono float32 numpy at 16kHz."""
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # Resample to 16kHz
    if sample_rate != 16000:
        n_in = waveform.shape[0]
        n_out = int(n_in * 16000 / sample_rate)
        indices = torch.linspace(0, n_in - 1, n_out).long()
        waveform = waveform[indices]
        print(f"[LyricsSync] Resampled {sample_rate}Hz → 16000Hz")

    return waveform.cpu().numpy().astype(np.float32)


# ── Lyrics parsing ───────────────────────────────────────────────────────────

def _parse_lyrics(lyrics_text):
    """Parse lyrics text into structured lines with words."""
    lines = []
    for raw_line in lyrics_text.strip().split("\n"):
        stripped = raw_line.strip()
        if not stripped or _is_lyrics_metadata_line(stripped):
            continue
        # Extract words (keep original case for display, lowercase for matching)
        display_words = stripped.split()
        match_words = [_normalize_word(w) for w in display_words]
        # Filter out empty matches but keep display words aligned
        filtered = [(d, m) for d, m in zip(display_words, match_words) if m]
        if filtered:
            lines.append({
                "text": stripped,
                "words": [d for d, m in filtered],
                "match_words": [m for d, m in filtered],
            })
    return lines


# ── Sequence alignment ──────────────────────────────────────────────────────

def _align_to_whisper(lines, whisper_words, audio_duration):
    """Align parsed lyrics lines to Whisper word timestamps."""
    # Flatten lyrics words for global alignment
    flat_lyrics = []
    for line_idx, line in enumerate(lines):
        for word_idx, match_word in enumerate(line["match_words"]):
            flat_lyrics.append({
                "word": match_word,
                "display": line["words"][word_idx],
                "line_idx": line_idx,
                "word_idx": word_idx,
            })

    # Flatten Whisper words for matching
    whisper_normalized = [_normalize_word(w["word"]) for w in whisper_words]

    lyrics_normalized = [w["word"] for w in flat_lyrics]

    # ── Pass 1: Exact matching via SequenceMatcher ───────────────────
    matcher = difflib.SequenceMatcher(
        None, lyrics_normalized, whisper_normalized, autojunk=False
    )

    timestamps = [None] * len(flat_lyrics)
    whisper_used = [False] * len(whisper_words)

    for block in matcher.get_matching_blocks():
        li_start, wi_start, size = block
        for i in range(size):
            li = li_start + i
            wi = wi_start + i
            if li < len(timestamps) and wi < len(whisper_words):
                timestamps[li] = {
                    "start": whisper_words[wi]["start"],
                    "end": whisper_words[wi]["end"],
                    "interpolated": False,
                    "whisper_idx": wi,
                }
                whisper_used[wi] = True

    exact_matches = sum(1 for t in timestamps if t is not None)

    # ── Pass 2: Fuzzy matching for unmatched words ───────────────────
    # For each unmatched lyrics word, find the best fuzzy match among
    # nearby unmatched Whisper words (within a time-proportional window)
    if len(flat_lyrics) > 0 and len(whisper_words) > 0:
        ratio = len(whisper_words) / len(flat_lyrics)  # Approx position ratio

        for li in range(len(flat_lyrics)):
            if timestamps[li] is not None:
                continue

            lw = lyrics_normalized[li]
            # Estimate where this word should be in the Whisper sequence
            expected_wi = int(li * ratio)
            # Search window: ±20% of total whisper words, at least ±5
            window = max(4, min(12, int(len(whisper_words) * 0.04)))

            best_score = 0.75 if len(lw) > 4 else 0.88
            best_wi = None

            for wi in range(max(0, expected_wi - window),
                            min(len(whisper_words), expected_wi + window)):
                if whisper_used[wi]:
                    continue
                ww = whisper_normalized[wi]
                # Quick length check before expensive ratio calculation
                if abs(len(lw) - len(ww)) > max(len(lw), len(ww)) * 0.5:
                    continue
                score = difflib.SequenceMatcher(None, lw, ww, autojunk=False).ratio()
                if score > best_score:
                    best_score = score
                    best_wi = wi

            if best_wi is not None:
                prev_wi = -1
                for prev_li in range(li - 1, -1, -1):
                    prev_ts = timestamps[prev_li]
                    if prev_ts is not None and "whisper_idx" in prev_ts:
                        prev_wi = prev_ts["whisper_idx"]
                        break
                next_wi = len(whisper_words)
                for next_li in range(li + 1, len(timestamps)):
                    next_ts = timestamps[next_li]
                    if next_ts is not None and "whisper_idx" in next_ts:
                        next_wi = next_ts["whisper_idx"]
                        break
                if best_wi <= prev_wi or best_wi >= next_wi:
                    continue
                timestamps[li] = {
                    "start": whisper_words[best_wi]["start"],
                    "end": whisper_words[best_wi]["end"],
                    "interpolated": False,
                    "whisper_idx": best_wi,
                }
                whisper_used[best_wi] = True

    _drop_non_monotonic_matches(timestamps)

    matched_after_guard = sum(1 for t in timestamps if t is not None)
    exact_matches = min(exact_matches, matched_after_guard)
    fuzzy_matches = max(0, matched_after_guard - exact_matches)
    total = len(flat_lyrics)
    print(f"[LyricsSync] Match stats: {exact_matches} exact + {fuzzy_matches} fuzzy "
          f"= {exact_matches + fuzzy_matches}/{total} "
          f"({(exact_matches + fuzzy_matches) / max(total, 1) * 100:.0f}%)")

    # ── Interpolate remaining unmatched words ────────────────────────
    _interpolate_timestamps(timestamps, audio_duration)

    # ── Build aligned lines ──────────────────────────────────────────
    result = []
    for line_idx, line in enumerate(lines):
        line_words = []
        for word_idx in range(len(line["words"])):
            flat_idx = sum(len(lines[i]["words"]) for i in range(line_idx)) + word_idx
            ts = timestamps[flat_idx] if flat_idx < len(timestamps) else None
            if ts is None:
                ts = {"start": 0.0, "end": 0.1, "interpolated": True}
            line_words.append({
                "word": line["words"][word_idx],
                "start": ts["start"],
                "end": ts["end"],
                "interpolated": ts.get("interpolated", False),
            })

        line_start = line_words[0]["start"] if line_words else 0.0
        line_end = line_words[-1]["end"] if line_words else 0.0

        result.append({
            "text": line["text"],
            "line_start": line_start,
            "line_end": line_end,
            "words": line_words,
        })

    return result


def _drop_non_monotonic_matches(timestamps):
    """Discard rare bad anchors that would make lyrics jump backward in time."""
    last_end = -1.0
    for i, ts in enumerate(timestamps):
        if ts is None:
            continue
        if ts["start"] < last_end - 0.05:
            timestamps[i] = None
        else:
            last_end = max(last_end, ts["end"])


def _interpolate_timestamps(timestamps, duration):
    """Fill in missing timestamps by interpolating between known ones."""
    n = len(timestamps)
    if n == 0:
        return

    # Find first and last known timestamps
    first_known = None
    last_known = None
    for i, ts in enumerate(timestamps):
        if ts is not None:
            if first_known is None:
                first_known = i
            last_known = i

    if first_known is None:
        # No matches at all — distribute evenly across duration
        word_dur = duration / max(n, 1)
        for i in range(n):
            timestamps[i] = {
                "start": i * word_dur,
                "end": (i + 1) * word_dur,
                "interpolated": True,
            }
        return

    # Fill before first known
    if first_known > 0:
        ref_start = timestamps[first_known]["start"]
        gap = ref_start / first_known if first_known > 0 else 0.5
        for i in range(first_known):
            timestamps[i] = {
                "start": max(0, ref_start - (first_known - i) * gap),
                "end": max(0, ref_start - (first_known - i - 1) * gap),
                "interpolated": True,
            }

    # Fill after last known
    if last_known < n - 1:
        ref_end = timestamps[last_known]["end"]
        remaining = n - 1 - last_known
        gap = min((duration - ref_end) / remaining, 1.0) if remaining > 0 else 0.5
        for i in range(last_known + 1, n):
            offset = i - last_known
            timestamps[i] = {
                "start": ref_end + (offset - 1) * gap,
                "end": ref_end + offset * gap,
                "interpolated": True,
            }

    # Fill gaps between known timestamps
    i = 0
    while i < n:
        if timestamps[i] is None:
            # Find the gap boundaries
            gap_start = i
            while i < n and timestamps[i] is None:
                i += 1
            gap_end = i  # First non-None after gap

            # Get boundary timestamps
            before_end = timestamps[gap_start - 1]["end"] if gap_start > 0 else 0.0
            after_start = timestamps[gap_end]["start"] if gap_end < n else duration

            gap_size = gap_end - gap_start
            step = (after_start - before_end) / (gap_size + 1)

            for j in range(gap_size):
                timestamps[gap_start + j] = {
                    "start": before_end + (j + 1) * step - step * 0.5,
                    "end": before_end + (j + 1) * step + step * 0.5,
                    "interpolated": True,
                }
        else:
            i += 1


# ── BPM detection ────────────────────────────────────────────────────────────

def _longest_interpolated_run(aligned_lines):
    longest = 0
    current = 0
    for line in aligned_lines:
        for word in line.get("words", []):
            if word.get("interpolated"):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
    return longest


def _alignment_reliability(aligned_lines, total_lyrics_words, transcript_word_count, audio_duration):
    if total_lyrics_words <= 0 or not aligned_lines:
        return False, "no lyric words"

    matched = sum(
        1 for line in aligned_lines for word in line.get("words", [])
        if not word.get("interpolated")
    )
    minimum_matches = min(total_lyrics_words, max(2, math.ceil(total_lyrics_words * 0.32)))
    if matched < minimum_matches:
        return False, f"only {matched}/{total_lyrics_words} reliable word anchors"

    transcript_ratio = transcript_word_count / max(total_lyrics_words, 1)
    if transcript_ratio < 0.25 or transcript_ratio > 4.0:
        return False, f"transcript/lyrics word ratio {transcript_ratio:.2f} is unreliable"

    longest_gap = _longest_interpolated_run(aligned_lines)
    if longest_gap > max(8, math.ceil(total_lyrics_words * 0.45)):
        return False, f"{longest_gap} consecutive words lacked anchors"

    last_start = -1.0
    repairable_timestamps = 0
    duration = max(0.1, float(audio_duration or 0.0))
    for line in aligned_lines:
        for word in line.get("words", []):
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                return False, "invalid word timestamp"
            if not math.isfinite(start) or not math.isfinite(end):
                return False, "non-finite word timestamp"
            if start < -0.25 or start > duration + 0.25 or end < -0.25 or end > duration + 0.50:
                return False, "out-of-range timestamps"
            if start < last_start - 0.15 or end < start - 0.15:
                return False, "severely non-monotonic timestamps"
            if start < last_start or end <= start or start < 0.0 or end > duration:
                repairable_timestamps += 1
            last_start = max(last_start, start)

    quality = f"{matched}/{total_lyrics_words} anchored words"
    if repairable_timestamps:
        quality += f"; normalized {repairable_timestamps} minor timestamp issues"
    return True, quality


def _sanitize_aligned_timings(aligned_lines, audio_duration):
    """Make every word and line strictly monotonic before binary-search rendering."""
    duration = max(0.1, float(audio_duration or 0.0))
    all_words = [word for line in aligned_lines for word in line.get("words", [])]
    if not all_words:
        return aligned_lines

    minimum_word_duration = min(0.01, duration / len(all_words))
    cursor = 0.0
    word_index = 0
    for line in aligned_lines:
        words = line.get("words", [])
        for word in words:
            remaining = len(all_words) - word_index
            latest_start = max(0.0, duration - minimum_word_duration * remaining)
            latest_end = duration - minimum_word_duration * (remaining - 1)
            start = max(cursor, min(latest_start, float(word.get("start", cursor))))
            end = max(
                start + minimum_word_duration,
                min(latest_end, float(word.get("end", start + minimum_word_duration))),
            )
            word["start"] = start
            word["end"] = end
            cursor = end
            word_index += 1
        if words:
            line["line_start"] = words[0]["start"]
            line["line_end"] = max(words[0]["start"] + minimum_word_duration, words[-1]["end"])
    return aligned_lines

def align_lyrics(audio_dict, lyrics_text, model_size="base", timing_offset_ms=0):
    """Align lyrics with Whisper, falling back to estimated timing when needed."""
    global _whisper_model, _whisper_model_name

    try:
        audio_np = _prepare_audio(audio_dict)
        duration = len(audio_np) / 16000.0
        print(f"[LyricsSync] Audio duration: {duration:.1f}s")

        import whisper
        if _whisper_model is None or _whisper_model_name != model_size:
            print(f"[LyricsSync] Loading Whisper '{model_size}' model...")
            _whisper_model = whisper.load_model(model_size)
            _whisper_model_name = model_size
            print(f"[LyricsSync] Whisper '{model_size}' loaded")

        print("[LyricsSync] Transcribing audio...")
        result = _transcribe_with_lyrics_hint(_whisper_model, audio_np, lyrics_text)

        whisper_words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_text = word_info.get("word", "").strip()
                if word_text:
                    whisper_words.append({
                        "word": word_text,
                        "start": float(word_info["start"]),
                        "end": float(word_info["end"]),
                    })
    except Exception as e:
        duration = _audio_duration_seconds(audio_dict)
        print(f"[LyricsSync] Whisper unavailable ({type(e).__name__}: {e})")
        return fallback_align_lyrics(lyrics_text, duration, timing_offset_ms)

    if not whisper_words:
        print("[LyricsSync] Whisper found no words; using estimated lyric timing")
        return fallback_align_lyrics(lyrics_text, duration, timing_offset_ms)

    print(f"[LyricsSync] Whisper detected {len(whisper_words)} words")
    lines = _parse_lyrics(lyrics_text)
    total_lyrics_words = sum(len(line["words"]) for line in lines)
    print(f"[LyricsSync] Provided lyrics: {len(lines)} lines, {total_lyrics_words} words")
    if not lines:
        return []

    aligned_lines = _align_to_whisper(lines, whisper_words, duration)
    reliable, quality_reason = _alignment_reliability(
        aligned_lines, total_lyrics_words, len(whisper_words), duration
    )
    if not reliable:
        vocal_start = max(0.0, float(whisper_words[0]["start"]) - 0.15)
        vocal_end = min(duration, float(whisper_words[-1]["end"]) + 0.20)
        print(f"[LyricsSync] Low-confidence alignment ({quality_reason}); using stable vocal-window timing")
        return fallback_align_lyrics(
            lyrics_text,
            duration,
            timing_offset_ms,
            active_start=vocal_start,
            active_end=vocal_end,
            fallback_reason=f"vocal-window fallback: {quality_reason}",
        )

    _apply_timing_offset(aligned_lines, timing_offset_ms)
    aligned_lines = _sanitize_aligned_timings(aligned_lines, duration)

    matched = sum(1 for line in aligned_lines for word in line["words"] if not word.get("interpolated"))
    print(f"[LyricsSync] Reliable alignment: {quality_reason}; {total_lyrics_words-matched} words interpolated")
    return aligned_lines


def _detect_bpm_torch(audio_dict):
    """Torch-only BPM fallback for environments where tensor->numpy is unavailable."""
    waveform = audio_dict["waveform"].detach().float()
    sample_rate = int(audio_dict["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.reshape(-1)

    if sample_rate <= 0 or waveform.numel() < sample_rate // 2:
        return 120.0

    hop = max(1, int(sample_rate * 0.01))
    win = max(hop, int(sample_rate * 0.02))
    if waveform.numel() <= win:
        return 120.0

    frames = waveform.unfold(0, win, hop)
    energy = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-10)
    if energy.numel() < 3:
        return 120.0

    onset = torch.clamp(energy[1:] - energy[:-1], min=0)
    max_onset = torch.max(onset)
    if max_onset > 0:
        onset = onset / max_onset

    fps = sample_rate / hop
    min_lag = max(1, int(fps * 60 / 200))
    max_lag = min(int(fps * 60 / 60), onset.numel() - 1)
    if max_lag <= min_lag:
        return 120.0

    best_corr = -1.0
    best_lag = max(min_lag, min(max_lag - 1, int(fps * 60 / 120)))
    for lag in range(min_lag, max_lag):
        n_overlap = onset.numel() - lag
        if n_overlap <= 0:
            continue
        corr = torch.sum(onset[:n_overlap] * onset[lag:lag + n_overlap]).item()
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    bpm = 60.0 * fps / max(1, best_lag)
    return max(60.0, min(200.0, round(bpm, 1)))


def detect_bpm(audio_dict):
    """
    Detect BPM from audio using onset autocorrelation.
    Returns estimated BPM (60-200 range).
    """
    waveform = audio_dict["waveform"]
    try:
        sample_rate = int(audio_dict["sample_rate"])
    except (TypeError, ValueError):
        return 120.0
    if sample_rate <= 0:
        return 120.0

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    try:
        audio = waveform.cpu().numpy().astype(np.float32)
    except Exception as e:
        bpm = _detect_bpm_torch(audio_dict)
        print(f"[LyricsSync] BPM numpy path unavailable ({type(e).__name__}: {e}); using torch fallback: {bpm}")
        return bpm

    # Energy envelope in 10ms windows
    hop = max(1, int(sample_rate * 0.01))
    win = max(hop, int(sample_rate * 0.02))
    if len(audio) <= win:
        return 120.0
    n_frames = max(1, (len(audio) - win) // hop)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + win]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

    # Onset strength (positive differences)
    if len(energy) < 3:
        return 120.0

    onset = np.diff(energy)
    onset = np.maximum(onset, 0)
    if onset.max() > 0:
        onset = onset / onset.max()

    # Autocorrelation for BPM (60-200 range)
    fps = sample_rate / hop
    min_lag = max(1, int(fps * 60 / 200))
    max_lag = min(int(fps * 60 / 60), len(onset) - 1)

    if max_lag <= min_lag:
        return 120.0

    best_corr = 0
    best_lag = int(fps * 60 / 120)

    for lag in range(min_lag, max_lag):
        n_overlap = len(onset) - lag
        if n_overlap <= 0:
            continue
        corr = np.sum(onset[:n_overlap] * onset[lag:lag + n_overlap])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    bpm = 60.0 * fps / best_lag
    bpm = max(60.0, min(200.0, round(bpm, 1)))

    print(f"[LyricsSync] 🥁 Detected BPM: {bpm}")
    return bpm
