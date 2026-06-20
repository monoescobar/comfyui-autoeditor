"""
DJ_LyricsOverlay — Karaoke & Lyrics Overlay Node for ComfyUI.

Syncs song lyrics to audio using Whisper word-level timestamps,
then renders animated text overlays on video frames with 10
TikTok-optimized display styles. Supports LLM auto-styling.
"""

import torch
from collections.abc import Mapping

from .lyrics_sync import align_lyrics, detect_bpm, fallback_align_lyrics, unload_whisper
from .text_renderer import TextRenderer, DISPLAY_STYLES, LINE_MODES, POSITIONS
from .ollama_bridge import list_ollama_models


LYRICS_OVERLAY_NODE_VERSION = "v2026.06.20.3"
LYRICS_FONT_SCALE = 0.85


class DJ_LyricsOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        try:
            ollama_models = list_ollama_models()
        except Exception:
            ollama_models = []
        # Always include offline marker so saved workflows don't break
        if "(ollama offline)" not in ollama_models:
            ollama_models.append("(ollama offline)")

        return {
            "required": {
                "song_audio": ("AUDIO",),
                "lyrics_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Paste your song lyrics here.\nEach line = one subtitle line.\nBlank lines are ignored.",
                }),
                "video_frames": ("IMAGE",),
                "display_style": (DISPLAY_STYLES, {
                    "default": "subtitles",
                    "tooltip": "Text animation style. karaoke=word highlight, word_pop=viral TikTok, neon_flash=EDM/club",
                }),
                "whisper_model": (["tiny", "base", "small", "medium"], {
                    "default": "base",
                    "tooltip": "Whisper model for lyrics alignment. tiny=fast, base=balanced, medium=accurate",
                }),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO",),
                "fps_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1,
                    "tooltip": "Override FPS (0=auto from video_info, or 25fps default). Set this if lyrics are out of sync.",
                }),
                "ai_stylist": (["OFF", "ON"], {
                    "default": "OFF",
                    "tooltip": "When ON, LLM auto-picks style/colors/font based on lyrics content. Manual settings ignored.",
                }),
                "llm_model": (ollama_models, {"default": ollama_models[0]}),
                "timing_offset_ms": ("INT", {
                    "default": 0, "min": -2000, "max": 2000, "step": 10,
                    "tooltip": "Manual timing offset in ms. Positive=lyrics appear later, Negative=earlier.",
                }),
                "font_size": ("INT", {
                    "default": 42, "min": 16, "max": 120, "step": 2,
                    "tooltip": "Text font size in pixels",
                }),
                "font_family": (["arial", "impact", "roboto", "montserrat", "bebas_neue",
                                  "comic_sans", "times", "courier"], {
                    "default": "arial",
                }),
                "text_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Text color as hex (e.g. #FFFFFF for white)",
                }),
                "highlight_color": ("STRING", {
                    "default": "#FFD700",
                    "tooltip": "Active/highlighted word color as hex (e.g. #FFD700 for gold)",
                }),
                "text_position": (POSITIONS, {
                    "default": "lower_third",
                    "tooltip": "Vertical position of text on the video",
                }),
                "text_alignment": (["left", "center", "right"], {
                    "default": "center",
                }),
                "outline_thickness": ("INT", {
                    "default": 3, "min": 0, "max": 10, "step": 1,
                }),
                "outline_color": ("STRING", {
                    "default": "#000000",
                }),
                "background_style": (["none", "solid_bar", "gradient_bar", "rounded_box",
                                       "blur_box", "shadow_only"], {
                    "default": "none",
                }),
                "background_opacity": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "text_shadow": (["enable", "disable"], {"default": "enable"}),
                "line_display": (LINE_MODES, {
                    "default": "single_line",
                    "tooltip": "How many lyrics lines to show at once",
                }),
            }
        }

    CATEGORY = "audio/video processing"
    FUNCTION = "overlay_lyrics"
    RETURN_NAMES = ("images_output", "audio_output", "video_info_output", "sync_report")
    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_VIDEOINFO", "STRING")

    @staticmethod
    def _safe_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _resolve_fps(cls, video_info, n_frames, fps_override=0.0):
        override = cls._safe_float(fps_override, 0.0) or 0.0
        if override > 0:
            return override, f"manual override ({override:.1f})"

        info = video_info if isinstance(video_info, Mapping) else {}
        fps = cls._safe_float(info.get("loaded_fps"), 25.0) or 25.0
        duration = cls._safe_float(info.get("loaded_duration"))
        if duration is None:
            duration = cls._safe_float(info.get("source_duration"))

        if duration and duration > 0 and n_frames > 0:
            inferred_fps = n_frames / duration
            mismatch = abs(fps - inferred_fps) / max(inferred_fps, 1.0)
            if 1.0 <= inferred_fps <= 120.0 and (fps <= 0 or fps > 60.0 or mismatch > 0.20):
                return inferred_fps, f"duration-derived video_info ({inferred_fps:.1f})"

        if fps <= 0 or fps > 120.0:
            return 25.0, "safe default 25fps (invalid video_info fps)"
        if isinstance(video_info, Mapping):
            return fps, f"video_info ({fps:.1f})"
        return 25.0, "default 25fps (connect video_info for accuracy)"

    def overlay_lyrics(self, song_audio, lyrics_text, video_frames,
                       display_style, whisper_model, **kwargs):

        ai_stylist = kwargs.get("ai_stylist", "OFF") == "ON"
        llm_model = kwargs.get("llm_model", "")
        timing_offset = kwargs.get("timing_offset_ms", 0)
        video_info = kwargs.get("video_info", None)
        fps_override = self._safe_float(kwargs.get("fps_override", 0.0), 0.0) or 0.0

        # ── Determine FPS (critical for sync!) ───────────────────────
        n_frames = video_frames.shape[0]

        # Calculate audio duration for diagnostics
        try:
            sr = song_audio["sample_rate"]
            wf = song_audio["waveform"]
            audio_duration = wf.shape[-1] / sr
        except Exception:
            audio_duration = 0.0

        if fps_override and fps_override > 0:
            fps = fps_override
            fps_source = f"manual override ({fps:.1f})"
        elif isinstance(video_info, Mapping):
            fps = self._safe_float(video_info.get("loaded_fps", 25), 25.0) or 25.0
            if isinstance(fps, str):
                fps = float(fps)
            fps_source = f"video_info ({fps:.1f})"
        else:
            # Default to 25fps — DO NOT estimate from audio!
            # Audio and video are often different lengths.
            fps = 25.0
            fps_source = "default 25fps (connect video_info for accuracy)"

        fps, fps_source = self._resolve_fps(video_info, n_frames, fps_override)
        video_duration = n_frames / fps

        # Build a video_info dict for pass-through if none provided
        if video_info is None:
            video_info = {"loaded_fps": fps, "loaded_frame_count": n_frames}

        print(f"\n{'='*60}")
        print(f"[LyricsOverlay] DJ Lyrics Overlay {LYRICS_OVERLAY_NODE_VERSION}")
        print(f"[LyricsOverlay] Video: {n_frames} frames @ {fps:.1f}fps = {video_duration:.1f}s")
        print(f"[LyricsOverlay] Audio: {audio_duration:.1f}s")
        print(f"[LyricsOverlay] FPS source: {fps_source}")
        if abs(video_duration - audio_duration) > 1.0:
            print(f"[LyricsOverlay] WARNING: Video ({video_duration:.1f}s) and audio ({audio_duration:.1f}s) "
                  f"are different lengths! Lyrics follow audio timestamps on video frames.")
        print(f"[LyricsOverlay] Style: {display_style}")
        print(f"{'='*60}")

        if not lyrics_text.strip():
            print("[LyricsOverlay] No lyrics provided - passing through unchanged")
            return (video_frames, song_audio, video_info, "No lyrics provided.")

        # ── Step 1: Align lyrics to audio ────────────────────────────
        print(f"\n[LyricsOverlay] Step 1/3: Aligning lyrics with Whisper '{whisper_model}'...")
        try:
            aligned = align_lyrics(song_audio, lyrics_text, whisper_model, timing_offset)
        except Exception as e:
            print(f"[LyricsOverlay] Alignment engine failed ({type(e).__name__}: {e})")
            timing_duration = audio_duration if audio_duration > 0 else video_duration
            aligned = fallback_align_lyrics(lyrics_text, timing_duration, timing_offset)
        finally:
            unload_whisper()

        try:
            bpm = detect_bpm(song_audio)
        except Exception as e:
            print(f"[LyricsOverlay] BPM detection failed ({type(e).__name__}: {e}); using 120")
            bpm = 120.0

        if not aligned:
            print("[LyricsOverlay] ❌ Alignment failed — passing through unchanged")
            return (video_frames, song_audio, video_info, "Alignment failed.")

        timing_estimated = any(
            line.get("fallback_timing")
            or any(word.get("estimated") for word in line.get("words", []))
            for line in aligned
        )

        # ── Step 2: LLM auto-styling (if enabled) ───────────────────
        config = self._build_config(kwargs, display_style)

        if ai_stylist and llm_model and "(ollama" not in llm_model:
            print(f"\n[LyricsOverlay] Step 2/3: AI Stylist via '{llm_model}'...")
            try:
                from .ollama_bridge import ask_ollama_lyrics_style
                llm_config = ask_ollama_lyrics_style(llm_model, lyrics_text, bpm)
                if llm_config:
                    # Keep the AI stylist deterministic and limited to safe visual choices.
                    for key in ("display_style", "font_family", "font_size"):
                        if key in llm_config:
                            config[key] = llm_config[key]
                    print(f"[LyricsOverlay] AI picked style: {config.get('display_style', display_style)}")
            except Exception as e:
                print(f"[LyricsOverlay] ⚠️ AI Stylist failed ({e}), using manual settings")
        else:
            print(f"\n[LyricsOverlay] Step 2/3: Using manual style settings")

        # ── Step 3: Render text on frames ────────────────────────────
        if timing_estimated and config.get("display_style") not in {"subtitles", "fade_flow"}:
            print("[LyricsOverlay] Sync safety: using subtitles because word timing confidence is low")
            config["display_style"] = "subtitles"
        config = self._normalize_config(config)

        print(f"\n[LyricsOverlay] Step 3/3: Rendering '{config['display_style']}' on {n_frames} frames...")
        renderer = TextRenderer(config)

        result = torch.empty_like(video_frames)
        for i in range(n_frames):
            timestamp = i / fps
            rendered = renderer.render_frame(video_frames[i], timestamp, aligned, bpm)
            result[i].copy_(rendered)
            if (i + 1) % 100 == 0 or i == n_frames - 1:
                print(f"[LyricsOverlay]   Frame {i+1}/{n_frames} ({(i+1)/n_frames*100:.0f}%)")

        # ── Build report ─────────────────────────────────────────────
        report = self._build_report(aligned, bpm, config, n_frames, fps)

        print(f"\n[LyricsOverlay] Complete!")
        print(f"{'='*60}\n")

        return (result, song_audio, video_info, report)

    def _build_config(self, kwargs, display_style):
        return {
            "display_style": display_style,
            "font_size": kwargs.get("font_size", 42),
            "font_family": kwargs.get("font_family", "arial"),
            "text_color": kwargs.get("text_color", "#FFFFFF"),
            "highlight_color": kwargs.get("highlight_color", "#FFD700"),
            "text_position": kwargs.get("text_position", "lower_third"),
            "text_alignment": kwargs.get("text_alignment", "center"),
            "outline_thickness": kwargs.get("outline_thickness", 3),
            "outline_color": kwargs.get("outline_color", "#000000"),
            "background_style": kwargs.get("background_style", "none"),
            "background_opacity": kwargs.get("background_opacity", 0.6),
            "text_shadow": kwargs.get("text_shadow", "enable"),
            "line_display": kwargs.get("line_display", "single_line"),
        }

    def _normalize_config(self, config):
        config = dict(config)
        if config.get("display_style") not in DISPLAY_STYLES:
            config["display_style"] = "subtitles"
        if config.get("line_display") not in LINE_MODES:
            config["line_display"] = "single_line"
        if config.get("text_position") not in POSITIONS or config.get("text_position") == "bottom":
            config["text_position"] = "lower_third"
        if config.get("text_alignment") not in {"left", "center", "right"}:
            config["text_alignment"] = "center"
        try:
            raw_size = int(float(config.get("font_size", 42)))
        except (TypeError, ValueError):
            raw_size = 42
        config["font_size"] = max(16, min(120, int(round(raw_size * LYRICS_FONT_SCALE))))
        try:
            config["background_opacity"] = max(0.0, min(1.0, float(config.get("background_opacity", 0.6))))
        except (TypeError, ValueError):
            config["background_opacity"] = 0.6
        return config

    def _build_report(self, aligned, bpm, config, n_frames, fps):
        r = []
        r.append("═" * 50)
        r.append("🎤 LYRICS OVERLAY — SYNC REPORT")
        r.append("═" * 50)
        r.append("")
        r.append(f"📊 ALIGNMENT")
        r.append("─" * 40)
        r.append(f"  Lines: {len(aligned)}")
        total_words = sum(len(ln['words']) for ln in aligned)
        matched = sum(1 for ln in aligned for w in ln['words'] if not w.get('interpolated'))
        r.append(f"  Words: {total_words} ({matched} matched, {total_words-matched} interpolated)")
        estimated = any(ln.get("fallback_timing") or any(w.get("estimated") for w in ln["words"]) for ln in aligned)
        if estimated:
            timing_source = next(
                (ln.get("timing_source") for ln in aligned if ln.get("timing_source")),
                "estimated fallback",
            )
        else:
            timing_source = "Whisper word timing"
        r.append(f"  Timing source: {timing_source}")
        r.append(f"  BPM: {bpm}")
        r.append("")
        r.append(f"🎨 STYLE")
        r.append("─" * 40)
        r.append(f"  Display: {config['display_style']}")
        r.append(f"  Font: {config['font_family']} @ {config['font_size']}px")
        r.append(f"  Colors: text={config['text_color']}, highlight={config['highlight_color']}")
        r.append(f"  Position: {config['text_position']}, align={config['text_alignment']}")
        r.append(f"  Background: {config['background_style']} ({config['background_opacity']:.0%})")
        r.append("")
        r.append(f"📐 OUTPUT")
        r.append("─" * 40)
        r.append(f"  Frames: {n_frames} @ {fps}fps ({n_frames/fps:.1f}s)")
        r.append("")
        r.append(f"📝 LYRICS TIMING")
        r.append("─" * 40)
        for i, line in enumerate(aligned[:20]):
            r.append(f"  [{line['line_start']:6.2f}s - {line['line_end']:6.2f}s] {line['text'][:50]}")
        if len(aligned) > 20:
            r.append(f"  ... and {len(aligned)-20} more lines")
        r.append("")
        r.append("═" * 50)
        return "\n".join(r)


NODE_CLASS_MAPPINGS = {
    "DJ_LyricsOverlay": DJ_LyricsOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_LyricsOverlay": f"Lyrics Overlay {LYRICS_OVERLAY_NODE_VERSION} 🎤",
}
