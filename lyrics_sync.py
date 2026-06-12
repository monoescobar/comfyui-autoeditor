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

def align_lyrics(audio_dict, lyrics_text, model_size="base", timing_offset_ms=0):
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
    result = _whisper_model.transcribe(
        audio_np,
        word_timestamps=True,
        language="en",
    )

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
        if not stripped:
            continue
        # Extract words (keep original case for display, lowercase for matching)
        display_words = stripped.split()
        match_words = [re.sub(r"[^a-zA-Z']", "", w).lower() for w in display_words]
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
    whisper_normalized = [re.sub(r"[^a-zA-Z']", "", w["word"]).lower()
                          for w in whisper_words]

    lyrics_normalized = [w["word"] for w in flat_lyrics]

    # ── Pass 1: Exact matching via SequenceMatcher ───────────────────
    matcher = difflib.SequenceMatcher(None, lyrics_normalized, whisper_normalized)

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
            window = max(5, int(len(whisper_words) * 0.2))

            best_score = 0.55  # Minimum threshold for fuzzy match
            best_wi = None

            for wi in range(max(0, expected_wi - window),
                            min(len(whisper_words), expected_wi + window)):
                if whisper_used[wi]:
                    continue
                ww = whisper_normalized[wi]
                # Quick length check before expensive ratio calculation
                if abs(len(lw) - len(ww)) > max(len(lw), len(ww)) * 0.5:
                    continue
                score = difflib.SequenceMatcher(None, lw, ww).ratio()
                if score > best_score:
                    best_score = score
                    best_wi = wi

            if best_wi is not None:
                timestamps[li] = {
                    "start": whisper_words[best_wi]["start"],
                    "end": whisper_words[best_wi]["end"],
                    "interpolated": False,
                }
                whisper_used[best_wi] = True

    fuzzy_matches = sum(1 for t in timestamps if t is not None) - exact_matches
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

def detect_bpm(audio_dict):
    """
    Detect BPM from audio using onset autocorrelation.
    Returns estimated BPM (60-200 range).
    """
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    audio = waveform.cpu().numpy().astype(np.float32)

    # Energy envelope in 10ms windows
    hop = int(sample_rate * 0.01)
    win = int(sample_rate * 0.02)
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
