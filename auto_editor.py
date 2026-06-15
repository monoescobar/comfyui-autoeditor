"""
DJ_AutoEditor - Fragment & Shuffle Video Editing Engine for ComfyUI.

Professional edit engine that chops source videos into fragments,
shuffles them across the timeline, and applies per-chunk effects
(reverse, punch-in, speed ramps, hold frames, black breaths).

20 Moods + LLM Auto mode. Every frame is used.
"""

import torch
import math
import random
from collections.abc import Mapping

from .presets import MOODS
from .transitions import join_segment_sequence
from .color_grading import apply_color_grade, get_grade_names, apply_visual_effects
from .ollama_bridge import (
    ACE_KEY_SCALES,
    ACE_TIME_SIGNATURES,
    list_ollama_models,
    ask_ollama,
    ask_ollama_with_descriptions,
    frames_to_base64,
)
from .vision_analysis import analyze_videos, format_descriptions_for_llm, detect_distortions, remove_distorted_frames, get_vision_quality_names


AUTOEDITOR_NODE_VERSION = "v2026.06.15.1"
MAX_FRAME_BATCH_ELEMENTS = 12_000_000


class DJ_AutoEditor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        try:
            ollama_models = list_ollama_models()
        except Exception:
            ollama_models = ["(ollama offline)"]

        return {
            "required": {
                "llm_model": (ollama_models, {"default": ollama_models[0]}),
                "llm_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional creative direction. Example: premium skincare, elegant but energetic, make the product feel real and desirable.",
                }),
                "images1": ("IMAGE",),
                "video_info1": ("VHS_VIDEOINFO",),
                "images2": ("IMAGE",),
                "video_info2": ("VHS_VIDEOINFO",),
            },
            "optional": {
                "video_understanding": (["ON", "FAST", "OFF"], {
                    "default": "FAST",
                    "tooltip": "Controls Florence-2 video understanding. ON=best quality/slow, FAST=fewer keyframes, OFF=skip vision analysis.",
                }),
                "lyrics_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional lyrics for music direction. Auto Editor uses this with video mood and BPM to recommend ACE-Step keyscale, time signature, and tags.",
                }),
                # Source videos
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "images3": ("IMAGE",),
                "audio3": ("AUDIO",),
                "video_info3": ("VHS_VIDEOINFO",),
                "images4": ("IMAGE",),
                "audio4": ("AUDIO",),
                "video_info4": ("VHS_VIDEOINFO",),
                "images5": ("IMAGE",),
                "audio5": ("AUDIO",),
                "video_info5": ("VHS_VIDEOINFO",),
                "images6": ("IMAGE",),
                "audio6": ("AUDIO",),
                "video_info6": ("VHS_VIDEOINFO",),
            }
        }

    CATEGORY = "audio/video processing"
    FUNCTION = "auto_edit"
    RETURN_NAMES = (
        "images_output", "audio_output", "video_info_output", "edit_report",
        "vision_descriptions", "output_frame_count", "recommended_bpm",
        "recommended_keyscale", "recommended_timesignature", "recommended_music_tags",
    )
    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_VIDEOINFO", "STRING", "STRING", "INT", "INT", "STRING", "INT", "STRING")

    # ─── Mood descriptions for the edit report ────────────────────────────
    MOOD_DESCRIPTIONS = {
        "dreamy": (
            "DREAMY — Soft, floaty, ethereal.\n"
            "Long dissolves let each shot breathe. Slow motion stretches time "
            "so the viewer lingers on textures, light, and detail. "
            "Pastel color grading lifts the shadows and softens contrast.\n"
            "Best for: baby products, skincare, candles, cream, silk."
        ),
        "romantic": (
            "ROMANTIC — Tender, soft-focus, warm.\n"
            "Gentle dissolves and luma fades create intimacy between shots. "
            "Slow motion adds a sense of savoring each moment. "
            "Pastel tones with soft pink warmth.\n"
            "Best for: beauty, love, weddings, cosmetics, soft fashion."
        ),
        "calm": (
            "CALM — Peaceful, serene, natural.\n"
            "Gentle cross-dissolves at a natural pace. No speed manipulation — "
            "the footage plays exactly as shot, preserving organic movement. "
            "Clean bright grading keeps colors true and fresh.\n"
            "Best for: wellness, tea, yoga, nature, organic products."
        ),
        "nature": (
            "NATURE — Organic, earthy, flowing.\n"
            "Smooth dissolves that feel like one scene flowing into the next. "
            "Natural speed preserves the rhythm of the outdoors. "
            "Clean bright grading enhances greens and sky blues.\n"
            "Best for: outdoor gear, eco brands, gardening, natural cosmetics."
        ),
        "elegant": (
            "ELEGANT — Luxurious, refined, premium.\n"
            "Measured pacing with luma-based fades that transition through "
            "brightness. Slight slow motion adds weight and intentionality. "
            "Cinematic warm grading adds golden richness.\n"
            "Best for: jewelry, fashion, watches, perfume, high-end goods."
        ),
        "warm": (
            "WARM — Cozy, inviting, friendly.\n"
            "Comfortable mid-length cuts with soft dissolves and gentle wipes. "
            "Natural speed keeps everything feeling real and approachable. "
            "Vintage warm grading adds golden nostalgia.\n"
            "Best for: food, home goods, family products, bakeries."
        ),
        "minimal": (
            "MINIMAL — Clean, simple, no effects.\n"
            "Straight hard cuts with no transitions, no speed changes, "
            "and no color grading. Lets the footage speak for itself. "
            "Maximum clarity, zero distraction.\n"
            "Best for: modern tech, clean design, architecture, apps."
        ),
        "playful": (
            "PLAYFUL — Fun, bouncy, colorful.\n"
            "Varied cut lengths create a peppy, unpredictable rhythm. "
            "Mixed transitions (wipes, zooms, hard cuts) keep the eye surprised. "
            "Vivid pop grading cranks up saturation for max color.\n"
            "Best for: toys, games, kids products, casual wear, snacks."
        ),
        "retro": (
            "RETRO — Vintage, analog, nostalgic.\n"
            "Hard cuts with occasional black flashes mimic old film edits. "
            "Natural speed with no tricks — authentic and raw. "
            "Vintage warm grading gives a golden, aged feel.\n"
            "Best for: heritage brands, vinyl, classic style, throwbacks."
        ),
        "cinematic": (
            "CINEMATIC — Film-grade, dramatic, epic.\n"
            "Theatrical pacing with dark luma fades and black flash transitions. "
            "Slow motion adds gravitas. Moody dark grading crushes blacks, "
            "lifts fog into shadows, and adds a cool tint.\n"
            "Best for: luxury brands, trailers, architecture, premium campaigns."
        ),
        "bold": (
            "BOLD — Confident, striking, powerful.\n"
            "Deliberate pacing with flash accents and zoom punches at cut points. "
            "Speed ramps up toward the end, building momentum. "
            "High contrast grading sharpens every detail.\n"
            "Best for: tech, sneakers, cars, fragrances, launches."
        ),
        "luxury": (
            "LUXURY — Ultra-premium, slow, golden.\n"
            "Extra-long cuts with smooth luma fades let every angle be admired. "
            "Deep slow motion at 0.70x creates weight and exclusivity. "
            "Cinematic warm grading bathes everything in gold.\n"
            "Best for: haute couture, supercars, five-star, limited editions."
        ),
        "urban": (
            "URBAN — Gritty, fast, street-style.\n"
            "Quick cuts with whip pans and glitch effects. "
            "Fast playback compresses the energy. "
            "Teal-orange split toning adds a raw cinematic edge.\n"
            "Best for: streetwear, music, graffiti, nightlife, sneakers."
        ),
        "energetic": (
            "ENERGETIC — High-energy, dynamic, exciting.\n"
            "Fast interleaved cuts that accelerate through the edit. "
            "Whip pans, shakes, and flashes create constant motion. "
            "Teal-orange grading adds cinematic punch.\n"
            "Best for: sports, fitness, energy drinks, activewear."
        ),
        "intense": (
            "INTENSE — Aggressive, raw, maximal.\n"
            "Rapid-fire cuts with chaotic ordering for maximum visual assault. "
            "Glitch effects, RGB splits, and camera shakes between every cut. "
            "Everything runs fast — compressed and forceful.\n"
            "Best for: extreme sports, gaming, EDM, action, streetwear."
        ),
        "hypnotic": (
            "HYPNOTIC — Trance-like, rhythmic, mesmerizing.\n"
            "Repetitive rhythm with subtle variations creates a hypnotic pull. "
            "Reverse segments and hold frames add visual loops. "
            "Moody dark grading with heavy vignette focuses attention.\n"
            "Best for: perfume, tech, ASMR, ambient, meditation."
        ),
        "raw": (
            "RAW — Documentary, behind-the-scenes, authentic.\n"
            "Handheld-feel with shake cuts and hard cuts only. "
            "No color grading — footage stays honest and unpolished. "
            "Film grain and vignette add analog character.\n"
            "Best for: streetwear, behind-the-scenes, authentic brands."
        ),
        "neon": (
            "NEON — Cyberpunk, saturated, glitch-forward.\n"
            "Aggressive fragmentation with glitch and flash transitions. "
            "Vivid pop grading cranks saturation into neon territory. "
            "Chromatic aberration and rapid bursts create digital energy.\n"
            "Best for: gaming, nightlife, tech, EDM, cyberpunk."
        ),
        "editorial": (
            "EDITORIAL — Vogue, fashion magazine, deliberate.\n"
            "Sharp impact moments punctuate measured pacing. "
            "Hold frames create dramatic pauses. Hollywood grading "
            "with anamorphic streaks adds a magazine-cover sheen.\n"
            "Best for: high fashion, luxury campaigns, beauty, editorial."
        ),
        "chaos": (
            "CHAOS — Maximum fragmentation, strobe, experimental.\n"
            "Extreme rapid-fire cuts with reverse segments and punch-ins. "
            "Every effect at maximum — glitches, flashes, shakes. "
            "Best for: art campaigns, music videos, experimental brands."
        ),
        "llm_auto": (
            "LLM AUTO — AI-directed editing.\n"
            "An Ollama language model analyzed your prompt and chose "
            "every editing parameter to match your creative vision."
        ),
    }

    # ─── Audio helpers ────────────────────────────────────────────────────

    @staticmethod
    def _get_audio_data(audio_input):
        if audio_input is None:
            return None, None
        if isinstance(audio_input, Mapping):
            try:
                waveform = audio_input["waveform"].squeeze(0)
                return waveform, audio_input["sample_rate"]
            except (KeyError, Exception):
                return None, None
        if isinstance(audio_input, dict) and "waveform" in audio_input:
            return audio_input["waveform"].squeeze(0), audio_input["sample_rate"]
        return None, None

    @staticmethod
    def _sanitize_audio(audio, label=""):
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = torch.clamp(audio, -2.0, 2.0)
        max_amp = audio.abs().max()
        if max_amp > 0.95:
            audio = audio * (0.95 / max_amp)
        return torch.clamp(audio, -0.95, 0.95)

    # ─── Speed helpers ────────────────────────────────────────────────────

    @staticmethod
    def _frame_batch_size(frames, max_elements=MAX_FRAME_BATCH_ELEMENTS):
        """Return a safe frame batch size for full-resolution image operations."""
        if not torch.is_tensor(frames) or frames.ndim < 4 or frames.shape[0] <= 1:
            return max(1, int(frames.shape[0])) if hasattr(frames, "shape") else 1
        per_frame_elements = max(1, int(frames[0].numel()))
        return max(1, min(int(frames.shape[0]), int(max_elements // per_frame_elements)))

    @staticmethod
    def _get_speed_for_cut(cut_idx, total_cuts, speed_ramp, speed_factor):
        if speed_ramp == "none":
            return 1.0
        elif speed_ramp == "slow_motion":
            return speed_factor
        elif speed_ramp == "speed_up":
            return speed_factor
        elif speed_ramp == "fast_start":
            return speed_factor if cut_idx < total_cuts / 2 else 1.0
        elif speed_ramp == "speed_up_end":
            return speed_factor if cut_idx >= total_cuts / 2 else 1.0
        elif speed_ramp == "accelerate":
            t = cut_idx / max(total_cuts - 1, 1)
            return 1.0 + (speed_factor - 1.0) * t
        return 1.0

    @staticmethod
    def _speed_ramp_frames(frames, speed):
        if abs(speed - 1.0) < 0.01:
            return frames
        n_in = frames.shape[0]
        n_out = max(1, int(n_in / speed))
        indices = torch.linspace(0, n_in - 1, n_out, device=frames.device).long()
        return frames[indices]

    @staticmethod
    def _speed_ramp_audio(audio, speed):
        if audio is None or abs(speed - 1.0) < 0.01:
            return audio
        n_in = audio.shape[-1]
        n_out = max(1, int(n_in / speed))
        indices = torch.linspace(0, n_in - 1, n_out, device=audio.device).long()
        return audio[..., indices]

    @staticmethod
    def _match_frame_count(frames, target_frames):
        """Temporally resample frames to the exact requested count."""
        target_frames = max(1, int(target_frames))
        if frames.shape[0] == target_frames:
            return frames
        if frames.shape[0] <= 1:
            return frames[:1].expand(target_frames, -1, -1, -1).clone()
        positions = torch.linspace(
            0, frames.shape[0] - 1, target_frames,
            device=frames.device,
            dtype=torch.float32,
        )
        result = torch.empty(
            (target_frames, frames.shape[1], frames.shape[2], frames.shape[3]),
            device=frames.device,
            dtype=frames.dtype,
        )
        batch_size = DJ_AutoEditor._frame_batch_size(result)
        if batch_size < target_frames:
            print(
                f"[AutoEditor] Memory-safe duration resample: "
                f"{target_frames} frames in batches of {batch_size}"
            )
        for start in range(0, target_frames, batch_size):
            end = min(start + batch_size, target_frames)
            pos = positions[start:end]
            left = torch.floor(pos).long()
            right = torch.clamp(left + 1, max=frames.shape[0] - 1)
            blend = (pos - left.float()).view(-1, 1, 1, 1).to(frames.dtype)
            blended = frames[left] * (1.0 - blend) + frames[right] * blend
            result[start:end].copy_(blended)
        return result

    @staticmethod
    def _match_audio_samples(audio, target_samples):
        """Temporally resample audio to the exact requested sample count."""
        target_samples = max(1, int(target_samples))
        if audio.shape[-1] == target_samples:
            return audio
        if audio.shape[-1] <= 1:
            return audio[..., :1].expand(*audio.shape[:-1], target_samples).clone()
        import torch.nn.functional as F
        original_shape = audio.shape
        flattened = audio.reshape(-1, 1, original_shape[-1])
        resized = F.interpolate(
            flattened,
            size=target_samples,
            mode="linear",
            align_corners=False,
        )
        return resized.reshape(*original_shape[:-1], target_samples)

    # ─── Micro speed ramp (slow→fast within one clip) ─────────────────────

    @staticmethod
    def _micro_speed_ramp(frames):
        """Apply slow-motion start → snap to normal within a single clip."""
        n = frames.shape[0]
        if n < 6:
            return frames
        # First 60% plays at ~0.6x (stretched), last 40% at ~1.4x (compressed)
        split = int(n * 0.6)
        slow_part = frames[:split]
        fast_part = frames[split:]
        # Stretch slow part
        n_slow_out = int(split * 1.5)
        slow_indices = torch.linspace(0, split - 1, n_slow_out, device=frames.device).long()
        slow_stretched = slow_part[slow_indices]
        # Compress fast part to maintain total ≈ original
        n_fast_out = max(1, n - n_slow_out)
        fast_indices = torch.linspace(0, fast_part.shape[0] - 1, n_fast_out, device=frames.device).long()
        fast_compressed = fast_part[fast_indices]
        return torch.cat([slow_stretched, fast_compressed], dim=0)

    # ─── Scale punch-in ──────────────────────────────────────────────────

    @staticmethod
    def _apply_punch_in(frames, scale=None):
        """Random subtle zoom crop on a chunk — simulates camera movement."""
        import torch.nn.functional as F
        if scale is None:
            scale = 1.0 + random.uniform(0.02, 0.08)
        n, h, w, c = frames.shape
        crop_h = max(1, int(h / scale))
        crop_w = max(1, int(w / scale))
        # Random offset within the crop area
        max_y = h - crop_h
        max_x = w - crop_w
        y0 = random.randint(0, max(max_y, 0))
        x0 = random.randint(0, max(max_x, 0))
        result = torch.empty_like(frames)
        batch_size = DJ_AutoEditor._frame_batch_size(frames)
        if batch_size < n:
            print(
                f"[AutoEditor] Memory-safe punch-in: "
                f"{n} frames in batches of {batch_size}"
            )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cropped = frames[start:end, y0:y0 + crop_h, x0:x0 + crop_w, :]
            cropped_p = cropped.permute(0, 3, 1, 2)
            resized = F.interpolate(
                cropped_p, size=(h, w), mode='bilinear', align_corners=False
            )
            result[start:end].copy_(resized.permute(0, 2, 3, 1))
        return result

    # ─── Hold frame (freeze last frame) ──────────────────────────────────

    @staticmethod
    def _apply_hold_frame(frames, n_hold=None):
        """Freeze the last frame for n_hold extra frames."""
        if n_hold is None:
            n_hold = random.randint(3, 5)
        last = frames[-1:].expand(n_hold, -1, -1, -1)
        return torch.cat([frames, last], dim=0)

    # ─── Resolution matching ──────────────────────────────────────────────

    @staticmethod
    def _match_resolution(images, target_h, target_w):
        if images.shape[1] == target_h and images.shape[2] == target_w:
            return images
        import comfy.utils
        n, _, _, c = images.shape
        result = torch.empty(
            (n, target_h, target_w, c),
            device=images.device,
            dtype=images.dtype,
        )
        batch_size = DJ_AutoEditor._frame_batch_size(result)
        if batch_size < n:
            print(
                f"[AutoEditor] Memory-safe resolution match: "
                f"{n} frames in batches of {batch_size}"
            )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s = images[start:end].permute(0, 3, 1, 2)
            s = comfy.utils.common_upscale(
                s, target_w, target_h, "bilinear", "center"
            )
            result[start:end].copy_(s.permute(0, 2, 3, 1))
        return result

    @staticmethod
    def _premium_memory_budget(fps, height, width, total_source_frames):
        """Return output limits for premium mode without crushing normal ads."""
        fps = max(1, float(fps))
        total_source_frames = max(1, int(total_source_frames))
        bytes_per_frame = max(1, height * width * 3 * 4)
        memory_frame_cap = max(96, int(3_500_000_000 // bytes_per_frame))

        standard_tiktok_frames = int(fps * 30)
        min_sales_frames = min(total_source_frames, int(max(fps * 12, total_source_frames * 0.75)))

        if total_source_frames <= standard_tiktok_frames:
            max_frames = total_source_frames
            guard_mode = "full_source_for_standard_tiktok_ad"
        else:
            preferred_cap = min(total_source_frames, int(fps * 45))
            if memory_frame_cap < min_sales_frames:
                max_frames = min_sales_frames
                guard_mode = "sales_duration_floor_over_memory_cap"
            else:
                max_frames = max(min_sales_frames, min(memory_frame_cap, preferred_cap))
                guard_mode = "memory_safe_long_source"

        return {
            "quality_mode": "premium",
            "max_output_frames": max_frames,
            "min_output_frames": min_sales_frames,
            "min_output_seconds": min_sales_frames / fps,
            "max_output_seconds": max_frames / fps,
            "source_frames": total_source_frames,
            "source_seconds": total_source_frames / fps,
            "memory_frame_cap": memory_frame_cap,
            "guard_mode": guard_mode,
            "bytes_per_frame": bytes_per_frame,
        }

    @staticmethod
    def _safe_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reliable_fps(video_info, frames, default=25.0):
        """Use duration-derived FPS when node metadata is obviously inconsistent."""
        info = video_info if isinstance(video_info, Mapping) else {}
        fps = DJ_AutoEditor._safe_float(info.get("loaded_fps"), default)
        if fps is None:
            fps = default

        frame_count = DJ_AutoEditor._safe_float(
            info.get("loaded_frame_count"),
            float(frames.shape[0]) if hasattr(frames, "shape") else None,
        )
        if not frame_count or frame_count <= 0:
            frame_count = float(frames.shape[0]) if hasattr(frames, "shape") else 0.0

        duration = DJ_AutoEditor._safe_float(info.get("loaded_duration"))
        if duration is None:
            duration = DJ_AutoEditor._safe_float(info.get("source_duration"))

        if duration and duration > 0 and frame_count > 0:
            inferred_fps = frame_count / duration
            fps_mismatch = abs(fps - inferred_fps) / max(inferred_fps, 1.0)
            if 1.0 <= inferred_fps <= 120.0 and (fps <= 0 or fps > 60.0 or fps_mismatch > 0.20):
                note = (
                    f"FPS safety: using duration-derived {inferred_fps:.2f}fps "
                    f"instead of metadata {fps:.2f}fps"
                )
                return inferred_fps, note

        if fps <= 0 or fps > 120.0:
            return default, f"FPS safety: using default {default:.2f}fps instead of metadata {fps:.2f}fps"
        return fps, ""

    @staticmethod
    def _score_source_videos(all_images, vision_descriptions=None):
        """Score each source for sharpness, exposure, stability, and product cues."""
        product_words = (
            "product", "bottle", "box", "package", "packaging", "label",
            "logo", "close-up", "closeup", "hand", "face", "model", "wearing",
            "spray", "cream", "device", "phone", "shoe", "bag", "watch",
        )
        scores = {}
        details = {}

        for idx, frames in all_images.items():
            n = frames.shape[0]
            if n <= 0:
                scores[idx] = 0.0
                details[idx] = "empty source"
                continue

            sample_n = min(12, n)
            sample_idx = torch.linspace(0, n - 1, sample_n).long()
            sample = frames[sample_idx].float()

            luminance = (
                sample[..., 0:1] * 0.299
                + sample[..., 1:2] * 0.587
                + sample[..., 2:3] * 0.114
            )
            brightness = luminance.mean().item()
            exposure_score = max(0.0, 1.0 - abs(brightness - 0.55) / 0.55)
            contrast_score = min(1.0, luminance.std().item() * 3.5)

            if sample.shape[1] > 1 and sample.shape[2] > 1:
                edge_x = (sample[:, :, 1:, :] - sample[:, :, :-1, :]).abs().mean().item()
                edge_y = (sample[:, 1:, :, :] - sample[:, :-1, :, :]).abs().mean().item()
                sharpness_score = min(1.0, (edge_x + edge_y) * 10.0)
            else:
                sharpness_score = 0.5

            if sample_n > 1:
                motion = (sample[1:] - sample[:-1]).abs().mean().item()
                stability_score = max(0.0, min(1.0, 1.0 - motion * 3.0))
            else:
                stability_score = 0.7

            caption_bonus = 0.0
            if vision_descriptions and idx in vision_descriptions:
                text = " ".join(c for _, c in vision_descriptions[idx]).lower()
                cue_hits = sum(1 for word in product_words if word in text)
                caption_bonus = min(0.25, cue_hits * 0.035)

            score = (
                exposure_score * 0.25
                + contrast_score * 0.20
                + sharpness_score * 0.30
                + stability_score * 0.25
                + caption_bonus
            )
            scores[idx] = max(0.0, min(1.0, score))
            details[idx] = (
                f"score={scores[idx]:.2f}, exposure={exposure_score:.2f}, "
                f"detail={sharpness_score:.2f}, stability={stability_score:.2f}"
            )

        return scores, details

    @staticmethod
    def _estimate_motion_energy(all_images):
        """Estimate visual motion energy across all source videos."""
        energies = []
        for frames in all_images.values():
            n = frames.shape[0]
            if n < 2:
                continue
            sample_n = min(24, n)
            sample_idx = torch.linspace(0, n - 1, sample_n).long()
            sample = frames[sample_idx].float()
            motion = (sample[1:] - sample[:-1]).abs().mean().item()
            energies.append(motion)
        if not energies:
            return 0.35
        return max(0.0, min(1.0, sum(energies) / len(energies) * 4.0))

    @staticmethod
    def _recommend_bpm(config, all_images, source_scores):
        """Recommend a song BPM from mood, motion, and footage confidence."""
        mood = str(config.get("selected_mood", "cinematic")).lower()
        mood_base = {
            "calm": 86,
            "dreamy": 92,
            "romantic": 94,
            "nature": 96,
            "elegant": 100,
            "warm": 104,
            "minimal": 100,
            "luxury": 104,
            "cinematic": 112,
            "editorial": 116,
            "retro": 116,
            "playful": 124,
            "bold": 126,
            "urban": 128,
            "energetic": 138,
            "neon": 140,
            "intense": 146,
            "chaos": 150,
            "raw": 118,
            "hypnotic": 108,
        }
        bpm = mood_base.get(mood, 118)
        motion_energy = DJ_AutoEditor._estimate_motion_energy(all_images)
        average_quality = (
            sum(source_scores.values()) / max(1, len(source_scores))
            if source_scores else 0.55
        )

        bpm += int(round((motion_energy - 0.35) * 32))
        if config.get("burst_frequency", 0) > 0.12:
            bpm += 6
        if config.get("shuffle_intensity", 0) > 0.6:
            bpm += 4
        if average_quality < 0.45:
            bpm -= 6

        bpm = int(max(82, min(152, bpm)))
        return int(round(bpm / 2) * 2)

    @staticmethod
    def _normalize_keyscale(value, default):
        if isinstance(value, str):
            clean = value.replace("NI ", "").strip()
            clean = " ".join(clean.split())
            for keyscale in ACE_KEY_SCALES:
                if clean.lower() == keyscale.lower():
                    return keyscale
        return default

    @staticmethod
    def _normalize_timesignature(value, default):
        if isinstance(value, str):
            value = value.strip()
            if "/" in value:
                value = value.split("/", 1)[0]
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        return value if value in ACE_TIME_SIGNATURES else default

    @staticmethod
    def _fallback_music_direction(config, bpm, lyrics_text):
        """Choose ACE-Step music metadata from mood, BPM, and lyric intent."""
        mood = str(config.get("selected_mood", "cinematic")).lower()
        lyrics_lower = (lyrics_text or "").lower()

        mood_keyscale = {
            "dreamy": "Db major",
            "romantic": "Bb major",
            "calm": "C major",
            "nature": "G major",
            "elegant": "F minor",
            "warm": "A major",
            "minimal": "C major",
            "playful": "D major",
            "retro": "A minor",
            "cinematic": "D minor",
            "bold": "G minor",
            "luxury": "F minor",
            "urban": "G minor",
            "energetic": "F# minor",
            "intense": "D# minor",
            "hypnotic": "E minor",
            "raw": "A minor",
            "neon": "G# minor",
            "editorial": "Bb minor",
            "chaos": "F# minor",
        }
        keyscale = mood_keyscale.get(mood, "C major")

        bright_words = ("fresh", "happy", "joy", "sun", "clean", "bright", "love", "smile", "new")
        dark_words = ("night", "dark", "desire", "danger", "mystery", "power", "drama", "luxury", "secret")
        if any(word in lyrics_lower for word in bright_words) and mood not in {"luxury", "cinematic", "intense", "neon"}:
            root = keyscale.split()[0]
            keyscale = f"{root} major" if f"{root} major" in ACE_KEY_SCALES else "C major"
        elif any(word in lyrics_lower for word in dark_words):
            root = keyscale.split()[0]
            keyscale = f"{root} minor" if f"{root} minor" in ACE_KEY_SCALES else "A minor"

        if mood in {"romantic"}:
            timesignature = 3
        elif mood in {"luxury", "cinematic", "elegant", "dreamy", "hypnotic", "editorial"} and bpm <= 120:
            timesignature = 6
        elif mood in {"energetic", "intense", "urban", "bold", "chaos"} and bpm >= 128:
            timesignature = 2
        else:
            timesignature = 4

        style_tags = {
            "dreamy": "dreamy polished pop, airy pads, soft percussion, floating hook",
            "romantic": "romantic elegant pop, warm chords, graceful rhythm, emotional melody",
            "calm": "clean wellness pop, soft groove, organic percussion, gentle premium mood",
            "nature": "organic cinematic pop, natural percussion, warm acoustic textures",
            "elegant": "elegant luxury commercial, silky bass, refined piano, glossy cinematic groove",
            "warm": "warm friendly pop, cozy chords, soft beat, inviting product ad",
            "minimal": "minimal modern commercial, clean synth pulse, crisp restrained beat",
            "playful": "playful upbeat pop, bright synths, catchy bounce, colorful ad hook",
            "retro": "retro modern disco pop, analog warmth, tasteful groove, nostalgic shine",
            "cinematic": "cinematic premium trailer pop, deep pulse, emotional rise, luxury reveal",
            "bold": "bold modern ad music, punchy drums, confident bass, sharp product hook",
            "luxury": "luxury cinematic pop, elegant strings, soft bloom, premium desire, glossy mix",
            "urban": "urban fashion beat, sleek bass, street-luxury drums, confident rhythm",
            "energetic": "energetic TikTok commercial pop, driving drums, memorable hook, sales lift",
            "intense": "intense action commercial, heavy pulse, aggressive rhythm, powerful drop",
            "hypnotic": "hypnotic premium electronic, pulsing synths, sensual loop, immersive groove",
            "raw": "raw documentary commercial, textured drums, authentic groove, analog grit",
            "neon": "neon electronic pop, glossy synths, club pulse, futuristic product energy",
            "editorial": "editorial fashion commercial, stylish bassline, luxury runway rhythm",
            "chaos": "controlled chaotic electronic ad, rapid accents, high energy, clean mix",
        }
        vocal_direction = "vocal-friendly with the provided lyrics" if (lyrics_text or "").strip() else "instrumental, no lead vocal"
        tags = (
            f"{style_tags.get(mood, style_tags['cinematic'])}, {vocal_direction}, "
            f"{bpm} bpm, {keyscale}, {timesignature}/4 feel, professional TikTok product advertisement, "
            "subtle film grain mood, polished commercial master, fresh variation"
        )
        return keyscale, timesignature, tags

    @classmethod
    def _finalize_music_direction(cls, config, bpm, lyrics_text):
        fallback_key, fallback_time, fallback_tags = cls._fallback_music_direction(
            config, bpm, lyrics_text
        )
        use_llm_music = bool(config.get("_music_direction_from_llm"))
        keyscale = cls._normalize_keyscale(
            config.get("recommended_keyscale") if use_llm_music else None,
            fallback_key,
        )
        timesignature = cls._normalize_timesignature(
            config.get("recommended_timesignature") if use_llm_music else None,
            fallback_time,
        )
        tags = config.get("recommended_music_tags") if use_llm_music else ""
        if isinstance(tags, list):
            tags = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
        tags = str(tags or "").strip()
        if not tags or tags.startswith("premium cinematic TikTok product ad, polished commercial"):
            tags = fallback_tags
        config["recommended_keyscale"] = keyscale
        config["recommended_timesignature"] = timesignature
        config["recommended_music_tags"] = tags[:900]
        return keyscale, timesignature, config["recommended_music_tags"]

    @staticmethod
    def _commercial_chunk_score(chunk, source_scores, fps):
        vid_idx, start, end = chunk
        dur = max(0.0, (end - start) / max(1, fps))
        if dur < 0.35:
            duration_score = 0.35
        elif dur <= 1.25:
            duration_score = 0.80
        elif dur <= 3.5:
            duration_score = 1.0
        else:
            duration_score = 0.75
        return source_scores.get(vid_idx, 0.5) * 0.75 + duration_score * 0.25

    @classmethod
    def _shape_premium_commercial_arc(cls, chunks, source_scores, fps, max_output_frames):
        """Create hook -> variation -> product hero ending while respecting memory limits."""
        if not chunks:
            return [], {"trimmed": False, "reason": "no chunks"}

        total_frames = sum(end - start for _, start, end in chunks)
        scored = sorted(
            chunks,
            key=lambda c: cls._commercial_chunk_score(c, source_scores, fps),
            reverse=True,
        )

        def chunk_frames(chunk):
            return max(0, chunk[2] - chunk[1])

        short_candidates = [
            c for c in scored
            if chunk_frames(c) <= int(max(1, fps) * 1.6)
        ]
        hook = short_candidates[0] if short_candidates else scored[0]
        hero = scored[0]
        if total_frames <= max_output_frames:
            ordered = list(chunks)
            if ordered and tuple(hook) in [tuple(c) for c in ordered]:
                ordered.remove(hook)
                ordered.insert(0, hook)
            if ordered and tuple(hero) in [tuple(c) for c in ordered] and tuple(hero) != tuple(hook):
                ordered.remove(hero)
                ordered.append(hero)
            ordered = cls._fix_consecutive(ordered)
            return ordered, {
                "trimmed": False,
                "source_frames": total_frames,
                "selected_frames": total_frames,
                "max_output_frames": max_output_frames,
                "hook_video": hook[0],
                "hero_video": hero[0],
            }

        selected = []
        selected_keys = set()

        def add_chunk(chunk, frame_budget):
            key = tuple(chunk)
            if key in selected_keys:
                return frame_budget
            n = chunk_frames(chunk)
            if n <= 0:
                return frame_budget
            if n > frame_budget:
                min_trim = min(frame_budget, max(4, int(max(1, fps) * 0.35)))
                if frame_budget >= min_trim and frame_budget > 0:
                    vid_idx, start, end = chunk
                    trimmed = (vid_idx, start, min(end, start + frame_budget))
                    selected.append(trimmed)
                    selected_keys.add(tuple(trimmed))
                    return 0
                return frame_budget
            selected.append(chunk)
            selected_keys.add(key)
            return frame_budget - n

        ending_frames = chunk_frames(hero) if tuple(hero) != tuple(hook) else 0
        budget = max_output_frames
        if ending_frames and ending_frames < budget:
            budget -= ending_frames

        budget = add_chunk(hook, budget)

        remaining = [c for c in chunks if tuple(c) not in selected_keys and tuple(c) != tuple(hero)]
        weighted_remaining = sorted(
            remaining,
            key=lambda c: (
                cls._commercial_chunk_score(c, source_scores, fps)
                + random.random() * 0.18
            ),
            reverse=True,
        )

        for chunk in weighted_remaining:
            if budget <= 0:
                break
            if selected and selected[-1][0] == chunk[0]:
                alternatives = [c for c in weighted_remaining if tuple(c) not in selected_keys and c[0] != selected[-1][0]]
                if alternatives:
                    chunk = alternatives[0]
            budget = add_chunk(chunk, budget)

        if tuple(hero) not in selected_keys:
            if chunk_frames(hero) <= budget:
                budget = add_chunk(hero, budget)
            elif selected:
                weakest_i = min(
                    range(len(selected)),
                    key=lambda i: cls._commercial_chunk_score(selected[i], source_scores, fps),
                )
                if chunk_frames(hero) <= chunk_frames(selected[weakest_i]):
                    selected_keys.discard(tuple(selected[weakest_i]))
                    selected[weakest_i] = hero
                    selected_keys.add(tuple(hero))

        selected = cls._fix_consecutive(selected)
        selected_frames = sum(chunk_frames(c) for c in selected)
        return selected, {
            "trimmed": selected_frames < total_frames,
            "source_frames": total_frames,
            "selected_frames": selected_frames,
            "max_output_frames": max_output_frames,
            "hook_video": hook[0],
            "hero_video": hero[0],
        }

    # ─── Fragmentation: chop a video into random-length chunks ───────────

    @staticmethod
    def _fragment_video(total_frames, fps, min_dur, max_dur):
        """
        Chop a video into dramatically varied fragments.
        Uses a bimodal distribution: favors either very short (snappy) or
        longer (breathing) cuts, avoiding boring uniform middle lengths.
        ALL frames are used — no gaps, no overlaps.
        """
        min_frames = max(1, int(min_dur * fps))
        max_frames = max(min_frames, int(max_dur * fps))
        frame_range = max_frames - min_frames
        chunks = []
        pos = 0
        cut_index = 0
        while pos < total_frames:
            remaining = total_frames - pos
            if remaining <= min_frames:
                # Last bit — absorb into previous chunk or make final chunk
                if chunks:
                    prev_start, prev_end = chunks[-1]
                    chunks[-1] = (prev_start, pos + remaining)
                else:
                    chunks.append((pos, pos + remaining))
                break

            # Bimodal distribution: alternate between short bursts and breathing room
            # Every 3-5 fast cuts, insert a longer "breathing" cut
            r = random.random()
            if cut_index > 0 and cut_index % random.randint(3, 5) == 0:
                # Breathing cut: use upper 60-100% of range
                t = 0.6 + random.random() * 0.4
                chunk_size = min_frames + int(frame_range * t)
            elif r < 0.4:
                # Short snappy cut: use lower 0-25% of range
                t = random.random() * 0.25
                chunk_size = min_frames + int(frame_range * t)
            elif r < 0.7:
                # Medium cut: 25-50% of range
                t = 0.25 + random.random() * 0.25
                chunk_size = min_frames + int(frame_range * t)
            else:
                # Longer cut: 50-100% of range
                t = 0.5 + random.random() * 0.5
                chunk_size = min_frames + int(frame_range * t)

            chunk_size = max(min_frames, min(chunk_size, remaining))
            # If leftover would be too small, absorb it
            if remaining - chunk_size < min_frames:
                chunk_size = remaining
            chunks.append((pos, pos + chunk_size))
            pos += chunk_size
            cut_index += 1
        return chunks

    # ─── Phased fragmentation: dramatic rhythm variation ─────────────────

    @staticmethod
    def _fragment_video_phased(total_frames, fps, phases=None):
        """
        Fragment a video using a multi-phase rhythm system.
        Each phase defines (min_dur, max_dur, n_chunks_range) for that section.
        Phases cycle across the timeline, creating dramatic cuts-per-second variation.

        Default phases: burst → breathing → medium → burst → ...
        """
        if phases is None:
            phases = [
                {"name": "burst", "min_dur": 0.2, "max_dur": 0.6, "min_cuts": 3, "max_cuts": 5},
                {"name": "breathing", "min_dur": 1.5, "max_dur": 4.0, "min_cuts": 1, "max_cuts": 2},
                {"name": "medium", "min_dur": 0.6, "max_dur": 1.5, "min_cuts": 2, "max_cuts": 4},
                {"name": "burst", "min_dur": 0.3, "max_dur": 0.8, "min_cuts": 2, "max_cuts": 4},
                {"name": "breathing", "min_dur": 2.0, "max_dur": 3.5, "min_cuts": 1, "max_cuts": 2},
                {"name": "medium", "min_dur": 0.5, "max_dur": 1.2, "min_cuts": 3, "max_cuts": 5},
            ]

        chunks = []
        pos = 0
        phase_idx = 0

        while pos < total_frames:
            remaining = total_frames - pos
            if remaining < max(3, int(0.15 * fps)):
                # Too small — absorb into last chunk
                if chunks:
                    prev_start, prev_end = chunks[-1]
                    chunks[-1] = (prev_start, pos + remaining)
                else:
                    chunks.append((pos, pos + remaining))
                break

            phase = phases[phase_idx % len(phases)]
            min_f = max(2, int(phase["min_dur"] * fps))
            max_f = max(min_f + 1, int(phase["max_dur"] * fps))
            n_cuts = random.randint(phase["min_cuts"], phase["max_cuts"])

            for _ in range(n_cuts):
                remaining = total_frames - pos
                if remaining < min_f:
                    if chunks:
                        prev_start, prev_end = chunks[-1]
                        chunks[-1] = (prev_start, pos + remaining)
                    else:
                        chunks.append((pos, pos + remaining))
                    pos = total_frames
                    break

                # Random size within phase range, with some jitter
                chunk_size = random.randint(min_f, min(max_f, remaining))
                # If leftover would be too small, absorb it
                if remaining - chunk_size < min_f:
                    chunk_size = remaining

                chunks.append((pos, pos + chunk_size))
                pos += chunk_size

            phase_idx += 1

        return chunks

    # ─── Vision-driven phase selection ────────────────────────────────────

    @staticmethod
    def _vision_to_phases(vision_descriptions, vid_idx, n_frames, fps):
        """
        Convert Florence-2 vision descriptions into fragmentation phases
        tailored to the visual content of a specific video.

        Returns a list of phase dicts, or None to use defaults.
        """
        if not vision_descriptions or vid_idx not in vision_descriptions:
            return None

        descs = vision_descriptions[vid_idx]
        if not descs:
            return None

        # Analyze descriptions for content cues
        all_text = " ".join(caption for _, caption in descs).lower()

        # Detect motion/action keywords
        motion_words = ["moving", "motion", "running", "spinning", "flying",
                        "flowing", "action", "dynamic", "fast", "speed",
                        "pouring", "splashing", "spraying", "shaking"]
        calm_words = ["static", "still", "product", "close-up", "closeup",
                      "detail", "packaging", "label", "text", "logo",
                      "bottle", "box", "display", "placed", "sitting"]

        motion_score = sum(1 for w in motion_words if w in all_text)
        calm_score = sum(1 for w in calm_words if w in all_text)

        if motion_score > calm_score + 1:
            # Motion-heavy footage: mostly bursts with brief breathing
            return [
                {"name": "burst", "min_dur": 0.2, "max_dur": 0.5, "min_cuts": 3, "max_cuts": 6},
                {"name": "breathing", "min_dur": 1.0, "max_dur": 2.0, "min_cuts": 1, "max_cuts": 1},
                {"name": "burst", "min_dur": 0.3, "max_dur": 0.7, "min_cuts": 2, "max_cuts": 4},
                {"name": "medium", "min_dur": 0.5, "max_dur": 1.0, "min_cuts": 2, "max_cuts": 3},
            ]
        elif calm_score > motion_score + 1:
            # Product/hero shot: longer breathing, fewer bursts
            return [
                {"name": "breathing", "min_dur": 2.0, "max_dur": 4.5, "min_cuts": 1, "max_cuts": 2},
                {"name": "medium", "min_dur": 0.8, "max_dur": 1.5, "min_cuts": 2, "max_cuts": 3},
                {"name": "breathing", "min_dur": 1.5, "max_dur": 3.5, "min_cuts": 1, "max_cuts": 2},
                {"name": "burst", "min_dur": 0.3, "max_dur": 0.6, "min_cuts": 2, "max_cuts": 3},
            ]
        else:
            # Balanced: use defaults
            return None

    # ─── Smart shuffle: avoid same-source back-to-back ───────────────────

    @staticmethod
    def _smart_shuffle(chunk_list, intensity):
        """
        Shuffle chunks with constraint: avoid same-source consecutive.
        intensity 0.0 = keep original order
        intensity 1.0 = maximum shuffle
        """
        if intensity <= 0.01 or len(chunk_list) <= 1:
            return chunk_list

        # Create a working copy
        pool = list(chunk_list)

        if intensity >= 0.99:
            # Full shuffle with back-to-back avoidance
            return DJ_AutoEditor._shuffle_avoid_consecutive(pool)

        # Partial shuffle: swap pairs with probability = intensity
        result = list(pool)
        n = len(result)
        for i in range(n - 1, 0, -1):
            if random.random() < intensity:
                j = random.randint(0, i)
                result[i], result[j] = result[j], result[i]

        # Post-process: fix consecutive same-source violations
        return DJ_AutoEditor._fix_consecutive(result)

    @staticmethod
    def _shuffle_avoid_consecutive(chunks):
        """Shuffle list avoiding same video_idx back-to-back when possible."""
        if len(chunks) <= 1:
            return chunks

        random.shuffle(chunks)
        result = DJ_AutoEditor._fix_consecutive(chunks)
        return result

    @staticmethod
    def _fix_consecutive(chunks):
        """Fix same-source back-to-back by swapping with nearest different source."""
        result = list(chunks)
        n = len(result)
        for i in range(1, n):
            if result[i][0] == result[i - 1][0]:  # Same source video
                # Find nearest chunk with different source to swap
                best_j = None
                best_dist = n
                for j in range(i + 1, n):
                    if result[j][0] != result[i - 1][0]:
                        # Check swapping won't create another violation
                        if j + 1 < n and result[j + 1][0] == result[i][0]:
                            continue
                        best_j = j
                        best_dist = j - i
                        break
                if best_j is not None:
                    result[i], result[best_j] = result[best_j], result[i]
        return result

    # ─── Burst pattern insertion ─────────────────────────────────────────

    @staticmethod
    def _insert_bursts(chunks, burst_frequency, fps, min_dur):
        """
        Insert burst patterns: split some chunks into 3-4 rapid sub-cuts
        with VARIED sizes (not equal splits) for natural rhythm.
        Only splits chunks that are long enough (> 1s).
        """
        if burst_frequency <= 0 or len(chunks) < 3:
            return chunks

        result = []
        min_burst_frames = max(6, int(0.45 * fps))  # Premium bursts should not feel like flicker

        for chunk in chunks:
            vid_idx, start, end = chunk
            chunk_frames = end - start

            # Only burst-split chunks that are long enough
            if chunk_frames > int(fps * 1.4) and random.random() < burst_frequency:
                # Split into 2-3 accented sub-cuts with VARIED sizes
                n_sub = random.randint(2, 3)
                # Generate random weights for unequal sub-cuts
                weights = [random.random() + 0.3 for _ in range(n_sub)]  # 0.3-1.3 range
                total_weight = sum(weights)
                sub_sizes = [max(min_burst_frames, int(chunk_frames * w / total_weight))
                             for w in weights]
                # Fix any rounding errors — adjust last sub-cut
                allocated = sum(sub_sizes[:-1])
                sub_sizes[-1] = chunk_frames - allocated
                if sub_sizes[-1] < min_burst_frames:
                    result.append(chunk)  # Can't split properly, keep original
                    continue
                pos = start
                for s_size in sub_sizes:
                    result.append((vid_idx, pos, pos + s_size))
                    pos += s_size
            else:
                result.append(chunk)

        return result

    # ─── Main execution ──────────────────────────────────────────────────

    def auto_edit(self, llm_model, llm_prompt,
                  images1, video_info1, images2, video_info2,
                  **kwargs):

        mood = "llm_auto"
        pacing_mode = "as_mood"
        transition_strength = "as_mood"
        audio_crossfade_ms = 80
        enable_reverse = False
        enable_punch_in = True
        enable_bursts = True
        enable_black_breath = False
        enable_hold = False
        enable_jump_cuts = False
        enable_micro_ramp = False
        video_understanding = str(kwargs.get("video_understanding", "FAST")).upper()
        if video_understanding not in {"ON", "FAST", "OFF"}:
            video_understanding = "FAST"
        enable_vision = video_understanding != "OFF"
        quality_mode = "premium"
        vision_quality = "detailed" if video_understanding == "ON" else "fast"
        distortion_mode = "skip_frames"
        enable_contrast_protect = True
        enable_phased = False
        lyrics_text = str(kwargs.get("lyrics_text", "") or "")

        stale_model_values = {"", "ON", "OFF", "(ollama offline)", "(no models found)"}
        if not isinstance(llm_model, str) or llm_model in stale_model_values or llm_model.startswith("("):
            fresh_models = list_ollama_models()
            usable_models = [m for m in fresh_models if isinstance(m, str) and m not in stale_model_values and not m.startswith("(")]
            if usable_models:
                preferred = next((m for m in usable_models if m == "gemma4:latest"), usable_models[0])
                print(f"[AutoEditor] Refreshed Ollama model: {llm_model!r} -> {preferred}")
                llm_model = preferred
            else:
                print(f"[AutoEditor] Ollama model refresh found no usable models: {fresh_models}")

        print(f"\n{'='*60}")
        print("[AutoEditor] AI DIRECTOR MODE: ALWAYS ON")
        print("[AutoEditor] The node will choose the mood/edit plan and apply professional settings.")
        print(f"[AutoEditor] Video understanding: {video_understanding} ({'Florence-2 ' + vision_quality if enable_vision else 'skipped'})")
        if llm_prompt and llm_prompt.strip():
            print(f"[AutoEditor] Creative direction: \"{llm_prompt.strip()[:140]}\"")
        else:
            print("[AutoEditor] No prompt given - fully automatic product-commercial direction.")
        print(f"{'='*60}")

        # Extract optional sources
        audio1 = kwargs.get("audio1")
        audio2 = kwargs.get("audio2")
        images3 = kwargs.get("images3")
        audio3 = kwargs.get("audio3")
        video_info3 = kwargs.get("video_info3")
        images4 = kwargs.get("images4")
        audio4 = kwargs.get("audio4")
        video_info4 = kwargs.get("video_info4")
        images5 = kwargs.get("images5")
        audio5 = kwargs.get("audio5")
        video_info5 = kwargs.get("video_info5")
        images6 = kwargs.get("images6")
        audio6 = kwargs.get("audio6")
        video_info6 = kwargs.get("video_info6")

        # ── Collect all videos ────────────────────────────────────────────
        all_images = {1: images1, 2: images2}
        all_audio = {1: audio1, 2: audio2}
        all_info = {1: video_info1, 2: video_info2}

        for idx, img, aud, info in [
            (3, images3, audio3, video_info3),
            (4, images4, audio4, video_info4),
            (5, images5, audio5, video_info5),
            (6, images6, audio6, video_info6),
        ]:
            if img is not None:
                all_images[idx] = img
                all_audio[idx] = aud
                all_info[idx] = info if info is not None else {
                    "loaded_fps": self._safe_float(
                        video_info1.get("loaded_fps", 25) if isinstance(video_info1, Mapping) else 25,
                        25.0,
                    ),
                    "loaded_frame_count": img.shape[0],
                }

        n_videos = len(all_images)

        # ── Get FPS ───────────────────────────────────────────────────────
        fps, fps_note = self._reliable_fps(all_info.get(1), images1)
        fps_notes = []
        if fps_note:
            fps_notes.append(fps_note)
            print(f"[AutoEditor] {fps_note}")

        # ── AI Distortion Detection & Removal ────────────────────────────
        if distortion_mode != "disable":
            mode_str = "skip" if distortion_mode == "skip_frames" else "freeze"
            total_removed = 0
            for idx in sorted(all_images.keys()):
                frames = all_images[idx]
                bad_indices, quality_scores = detect_distortions(frames, fps, sensitivity=2.5)
                if bad_indices:
                    n_before = frames.shape[0]
                    all_images[idx] = remove_distorted_frames(frames, bad_indices, mode=mode_str)
                    n_after = all_images[idx].shape[0]
                    total_removed += len(bad_indices)
                    print(f"[AutoEditor] 🧹 Video {idx}: {len(bad_indices)} distorted frames "
                          f"{'removed' if mode_str == 'skip' else 'frozen'} "
                          f"({n_before}→{n_after} frames)")
            if total_removed > 0:
                print(f"[AutoEditor] 🧹 Total distortion fixes: {total_removed} frames")
            else:
                print(f"[AutoEditor] ✨ No AI distortions detected")

        # ── Vision Analysis (Florence-2) ──────────────────────────────────
        vision_descriptions = None
        vision_context = ""
        vision_error = ""
        if enable_vision:
            print(f"[AutoEditor] 👁️ Running Florence-2 Vision Director (quality={vision_quality})...")
            try:
                vision_descriptions = analyze_videos(all_images, fps, quality=vision_quality)
                if vision_descriptions:
                    vision_context = format_descriptions_for_llm(
                        vision_descriptions, all_images, fps
                    )
                    print(f"[AutoEditor] 👁️ Vision analysis complete: "
                          f"{sum(len(v) for v in vision_descriptions.values())} descriptions")
                else:
                    vision_error = "analyze_videos returned None (model download or loading failed)"
                    print(f"[AutoEditor] ⚠️ Vision analysis unavailable: {vision_error}")
            except Exception as e:
                import traceback
                vision_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"[AutoEditor] ⚠️ Vision analysis failed: {e}")
                traceback.print_exc()
                vision_descriptions = None
        else:
            vision_error = "Vision analysis is disabled (video_understanding=OFF)"

        # ── Log source video info ─────────────────────────────────────────
        total_source_frames = 0
        print(f"\n{'='*60}")
        print(f"[AutoEditor] 🎬 Starting FRAGMENT & SHUFFLE edit with {n_videos} source videos")
        video_summary_lines = []
        for idx in sorted(all_images.keys()):
            n_frames = all_images[idx].shape[0]
            total_source_frames += n_frames
            vid_fps, vid_fps_note = self._reliable_fps(all_info.get(idx), all_images[idx], default=fps)
            if vid_fps_note:
                print(f"[AutoEditor] Video {idx} {vid_fps_note}")
            dur = n_frames / vid_fps
            line = f"  Video {idx}: {n_frames} frames ({dur:.1f}s)"
            print(f"[AutoEditor] {line}")
            video_summary_lines.append(line)
        print(f"[AutoEditor]   TOTAL: {total_source_frames} source frames")
        print(f"{'='*60}")

        # ── Get target resolution ─────────────────────────────────────────
        target_h = images1.shape[1]
        target_w = images1.shape[2]
        for idx in all_images:
            all_images[idx] = self._match_resolution(all_images[idx], target_h, target_w)

        memory_plan = self._premium_memory_budget(fps, target_h, target_w, total_source_frames)
        print(
            f"[AutoEditor] Premium mode: {memory_plan['guard_mode']} | "
            f"source {memory_plan['source_seconds']:.1f}s, target "
            f"{memory_plan['min_output_seconds']:.1f}-{memory_plan['max_output_seconds']:.1f}s "
            f"at {target_w}x{target_h}"
        )
        source_scores, source_details = self._score_source_videos(
            all_images, vision_descriptions
        )
        for idx in sorted(source_details):
            print(f"[AutoEditor] Source {idx} quality: {source_details[idx]}")

        # ── Get configuration ─────────────────────────────────────────────
        if mood == "llm_auto":
            config = self._get_llm_config(
                llm_model, llm_prompt, video_summary_lines, all_images,
                vision_context=vision_context,
                allow_keyframe_fallback=enable_vision,
                lyrics_text=lyrics_text,
            )
        else:
            config = MOODS.get(mood, MOODS["bold"]).copy()
            print(f"[AutoEditor] 🎭 Mood: '{mood}'")

        config, director_notes = self._apply_premium_director_defaults(
            config, llm_prompt, memory_plan
        )

        # ── Apply pacing override ─────────────────────────────────────────
        tweaks_applied = list(director_notes) + fps_notes

        PACING_PRESETS = {
            "as_mood": {"speed_mult": 1.0, "duration_mult": 1.0, "chunk_mult": 1.0},
            "slower": {"speed_mult": 0.8, "duration_mult": 1.3, "chunk_mult": 1.3},
            "much_slower": {"speed_mult": 0.6, "duration_mult": 1.6, "chunk_mult": 1.6},
            "faster": {"speed_mult": 1.3, "duration_mult": 0.75, "chunk_mult": 0.75},
            "much_faster": {"speed_mult": 1.6, "duration_mult": 0.5, "chunk_mult": 0.5},
        }

        TRANSITION_PRESETS = {
            "as_mood": {"frame_mult": 1.0, "intensity": None},
            "subtle": {"frame_mult": 0.5, "intensity": 0.5},
            "moderate": {"frame_mult": 0.8, "intensity": 0.75},
            "dramatic": {"frame_mult": 1.5, "intensity": 0.95},
            "extreme": {"frame_mult": 2.0, "intensity": 1.0},
        }

        # Pacing
        if pacing_mode != "as_mood":
            p = PACING_PRESETS[pacing_mode]
            original_sf = config.get("speed_factor", 1.0)
            config["speed_factor"] = max(0.25, min(4.0, original_sf * p["speed_mult"]))
            # Scale cut durations
            if config.get("cut_duration_mode") == "variable" and config.get("variable_durations"):
                parts = config["variable_durations"].split(",")
                new_parts = [str(round(float(x.strip()) * p["duration_mult"], 2)) for x in parts if x.strip()]
                config["variable_durations"] = ",".join(new_parts)
            else:
                original_fcd = config.get("fixed_cut_duration", 2.0)
                config["fixed_cut_duration"] = max(0.1, original_fcd * p["duration_mult"])
            # Scale chunk durations
            config["min_chunk_duration"] = max(0.2, config.get("min_chunk_duration", 0.5) * p["chunk_mult"])
            config["max_chunk_duration"] = max(0.5, config.get("max_chunk_duration", 2.5) * p["chunk_mult"])
            tweaks_applied.append(f"Pacing: {pacing_mode} (speed ×{p['speed_mult']}, cuts ×{p['duration_mult']})")

        # Transition strength
        if transition_strength != "as_mood":
            t = TRANSITION_PRESETS[transition_strength]
            original_tf = config.get("transition_frames", 5)
            config["transition_frames"] = max(1, int(original_tf * t["frame_mult"]))
            if t["intensity"] is not None:
                config["transition_intensity"] = t["intensity"]
            tweaks_applied.append(f"Transitions: {transition_strength} (frames ×{t['frame_mult']}, intensity={config['transition_intensity']})")

        if tweaks_applied:
            print(f"[AutoEditor] 🔧 Overrides applied:")
            for t in tweaks_applied:
                print(f"[AutoEditor]   {t}")

        # ── Log final config ──────────────────────────────────────────────
        print(f"[AutoEditor]   Transitions: {config.get('transitions', '?')}")
        print(f"[AutoEditor]   Speed: {config.get('speed_ramp', '?')} @ {config.get('speed_factor', '?')}x")
        print(f"[AutoEditor]   Color: {config.get('color_grade', '?')}")
        print(f"[AutoEditor]   Chunks: {config.get('min_chunk_duration', '?')}s - {config.get('max_chunk_duration', '?')}s")
        print(f"[AutoEditor]   Shuffle: {config.get('shuffle_intensity', '?')}")

        # Log feature toggles
        features_on = []
        if enable_reverse and config.get("reverse_chance", 0) > 0:
            features_on.append(f"reverse({config['reverse_chance']:.0%})")
        if enable_punch_in and config.get("punch_in_chance", 0) > 0:
            features_on.append(f"punch-in({config['punch_in_chance']:.0%})")
        if enable_bursts and config.get("burst_frequency", 0) > 0:
            features_on.append(f"bursts({config['burst_frequency']:.0%})")
        if enable_black_breath and config.get("black_breath_chance", 0) > 0:
            features_on.append(f"breaths({config['black_breath_chance']:.0%})")
        if enable_hold and config.get("hold_frame_chance", 0) > 0:
            features_on.append(f"hold({config['hold_frame_chance']:.0%})")
        if enable_micro_ramp and config.get("micro_ramp_chance", 0) > 0:
            features_on.append(f"micro-ramp({config['micro_ramp_chance']:.0%})")
        if enable_jump_cuts:
            features_on.append("jump-cuts")
        print(f"[AutoEditor]   Effects: {', '.join(features_on) if features_on else 'none'}")

        # ── Sample rate from first available audio ────────────────────────
        sample_rate = 44100
        for idx in sorted(all_audio.keys()):
            if all_audio[idx] is not None:
                _, sr = self._get_audio_data(all_audio[idx])
                if sr is not None:
                    sample_rate = sr
                    break

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: FRAGMENT ALL VIDEOS
        # ══════════════════════════════════════════════════════════════════
        min_chunk_dur = config.get("min_chunk_duration", 0.8)
        max_chunk_dur = config.get("max_chunk_duration", 2.5)
        shuffle_intensity = config.get("shuffle_intensity", 0.7)

        all_chunks = []  # List of (video_idx, start_frame, end_frame)

        # Get LLM-specified phases if available
        llm_phases = config.get("fragment_phases", None)

        for idx in sorted(all_images.keys()):
            total_vid_frames = all_images[idx].shape[0]

            if enable_phased:
                # Try vision-driven phases first, then LLM phases, then defaults
                vid_phases = None
                if vision_descriptions:
                    vid_phases = self._vision_to_phases(
                        vision_descriptions, idx, total_vid_frames, fps
                    )
                if vid_phases is None and llm_phases:
                    vid_phases = llm_phases
                vid_chunks = self._fragment_video_phased(
                    total_vid_frames, fps, phases=vid_phases
                )
                phase_label = "phased"
                if vid_phases and vision_descriptions and idx in vision_descriptions:
                    phase_label = "vision-driven"
            else:
                # Fallback: uniform fragmentation
                vid_chunks = self._fragment_video(
                    total_vid_frames, fps, min_chunk_dur, max_chunk_dur
                )
                phase_label = "uniform"

            for start, end in vid_chunks:
                all_chunks.append((idx, start, end))
            print(f"[AutoEditor] 🔪 Video {idx}: {len(vid_chunks)} chunks ({phase_label})")

        print(f"[AutoEditor] 📦 Total chunks in pool: {len(all_chunks)}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: SHUFFLE (with optional jump-cuts = within-video reorder)
        # ══════════════════════════════════════════════════════════════════

        if enable_jump_cuts:
            # Within each video, shuffle the chunk order before global shuffle
            # This creates jump-cut effects (non-contiguous playback within a source)
            from itertools import groupby

            by_video = {}
            for chunk in all_chunks:
                vid_idx = chunk[0]
                if vid_idx not in by_video:
                    by_video[vid_idx] = []
                by_video[vid_idx].append(chunk)

            # Shuffle within each video independently
            all_chunks = []
            for vid_idx in sorted(by_video.keys()):
                vid_chunks = by_video[vid_idx]
                if len(vid_chunks) > 2:
                    random.shuffle(vid_chunks)
                all_chunks.extend(vid_chunks)

        # Global shuffle across all videos
        shuffled_chunks = self._smart_shuffle(all_chunks, shuffle_intensity)
        shuffled_chunks, arc_plan = self._shape_premium_commercial_arc(
            shuffled_chunks, source_scores, fps, memory_plan["max_output_frames"]
        )
        if arc_plan.get("trimmed"):
            print(
                f"[AutoEditor] Premium memory guard selected "
                f"{arc_plan['selected_frames']}/{arc_plan['source_frames']} frames "
                f"for a safer commercial-length edit"
            )
            tweaks_applied.append(
                f"Memory-safe commercial arc: selected {arc_plan['selected_frames']} "
                f"of {arc_plan['source_frames']} candidate frames"
            )
        print(
            f"[AutoEditor] Commercial arc: hook from video {arc_plan.get('hook_video', '?')}, "
            f"hero ending from video {arc_plan.get('hero_video', '?')}"
        )

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3: BURST INSERTION (split long chunks into rapid sub-cuts)
        # ══════════════════════════════════════════════════════════════════

        if enable_bursts:
            burst_freq = config.get("burst_frequency", 0.3)
            shuffled_chunks = self._insert_bursts(shuffled_chunks, burst_freq, fps, min_chunk_dur)
            print(f"[AutoEditor] ⚡ After bursts: {len(shuffled_chunks)} chunks")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4: BUILD FRAME/AUDIO SEGMENTS WITH PER-CHUNK EFFECTS
        # ══════════════════════════════════════════════════════════════════

        cfg_speed_ramp = config.get("speed_ramp", "none")
        cfg_speed_factor = config.get("speed_factor", 1.0)
        total_cuts = len(shuffled_chunks)

        # Effect probabilities
        reverse_chance = config.get("reverse_chance", 0.1) if enable_reverse else 0.0
        punch_in_chance = config.get("punch_in_chance", 0.2) if enable_punch_in else 0.0
        black_breath_chance = config.get("black_breath_chance", 0.1) if enable_black_breath else 0.0
        hold_chance = config.get("hold_frame_chance", 0.1) if enable_hold else 0.0
        micro_ramp_chance = config.get("micro_ramp_chance", 0.15) if enable_micro_ramp else 0.0

        frame_segments = []
        audio_segments = []

        for cut_idx, (vid_idx, start, end) in enumerate(shuffled_chunks):
            n_source_frames = end - start
            if n_source_frames <= 0:
                continue

            # Extract frames. Keep this as a view until an effect really needs a copy.
            frames = all_images[vid_idx][start:end]

            # ── Per-chunk effects ─────────────────────────────────────────

            # Reverse
            if reverse_chance > 0 and random.random() < reverse_chance:
                frames = frames.flip(0)

            # Scale punch-in
            if punch_in_chance > 0 and random.random() < punch_in_chance:
                frames = self._apply_punch_in(frames)

            # Micro speed ramp (slow→fast within clip)
            if micro_ramp_chance > 0 and random.random() < micro_ramp_chance and frames.shape[0] >= 6:
                frames = self._micro_speed_ramp(frames)

            # Hold frame (freeze last frame)
            if hold_chance > 0 and random.random() < hold_chance:
                frames = self._apply_hold_frame(frames)

            # Global speed ramp
            speed = self._get_speed_for_cut(
                cut_idx, total_cuts, cfg_speed_ramp, cfg_speed_factor
            )
            frames = self._speed_ramp_frames(frames, speed)

            frame_segments.append(frames)

            # ── Audio ─────────────────────────────────────────────────────
            audio_wave, _ = self._get_audio_data(all_audio.get(vid_idx))
            if audio_wave is not None:
                start_sample = int(start / fps * sample_rate)
                n_samples = int(n_source_frames / fps * sample_rate)
                end_sample = start_sample + n_samples
                if end_sample > audio_wave.shape[-1]:
                    seg = audio_wave[..., start_sample:]
                    pad_len = end_sample - audio_wave.shape[-1]
                    if pad_len > 0:
                        pad = torch.zeros(*seg.shape[:-1], pad_len)
                        seg = torch.cat([seg, pad], dim=-1)
                else:
                    seg = audio_wave[..., start_sample:end_sample]
                seg = self._speed_ramp_audio(seg, speed)
            else:
                n_output_frames = frames.shape[0]
                n_samples = int(n_output_frames / fps * sample_rate)
                seg = torch.zeros(1, n_samples)

            audio_segments.append(seg)

            # Log
            actual_dur = frames.shape[0] / fps
            effects_str = ""
            if cut_idx < 30 or cut_idx % 20 == 0:
                print(f"[AutoEditor] Cut {cut_idx}: vid{vid_idx} "
                      f"[{start}:{end}] speed={speed:.2f}x → "
                      f"{frames.shape[0]}f ({actual_dur:.2f}s)")

        if not frame_segments:
            raise ValueError("[AutoEditor] No valid cut segments were produced!")

        # ── Match audio channels ──────────────────────────────────────────
        max_channels = max(s.shape[0] for s in audio_segments)
        for i in range(len(audio_segments)):
            if audio_segments[i].shape[0] < max_channels:
                audio_segments[i] = audio_segments[i].repeat(max_channels, 1)

        # ── Apply transitions ─────────────────────────────────────────────
        transitions_list = config.get("transitions", ["hard_cut"])
        cfg_transition_frames = config.get("transition_frames", 5)
        effective_transition_intensity = config.get("transition_intensity", 1.0)
        min_output_frames = int(memory_plan.get("min_output_frames", 0))
        selected_segment_frames = sum(seg.shape[0] for seg in frame_segments)
        transition_count = max(0, len(frame_segments) - 1)
        if transition_count and min_output_frames > 0:
            max_allowed_loss = max(0, selected_segment_frames - min_output_frames)
            max_overlap_each = max_allowed_loss // transition_count
            if cfg_transition_frames > max_overlap_each:
                old_transition_frames = cfg_transition_frames
                cfg_transition_frames = max(0, int(max_overlap_each))
                print(
                    f"[AutoEditor] Duration guard reduced transition frames "
                    f"{old_transition_frames} -> {cfg_transition_frames}"
                )
                tweaks_applied.append(
                    f"Duration guard: transitions reduced {old_transition_frames}->{cfg_transition_frames} frames"
                )

        print(f"[AutoEditor] Joining {len(frame_segments)} segments with transitions...")
        resolved_transitions = []
        for i in range(1, len(frame_segments)):
            t_type = transitions_list[(i - 1) % len(transitions_list)]
            # Black breath: randomly override transition with black_breath
            if black_breath_chance > 0 and random.random() < black_breath_chance:
                t_type = "black_breath"
            resolved_transitions.append(t_type)

        combined_frames = join_segment_sequence(
            frame_segments,
            resolved_transitions,
            cfg_transition_frames,
            effective_transition_intensity,
        )

        combined_audio = audio_segments[0]
        for i in range(1, len(audio_segments)):
            crossfade_samples = int(audio_crossfade_ms / 1000 * sample_rate)
            combined_audio = self._crossfade_audio(
                combined_audio, audio_segments[i], crossfade_samples
            )

        # ── Color grading ─────────────────────────────────────────────────
        grade = config.get("color_grade", "none")
        if grade != "none":
            print(f"[AutoEditor] 🎨 Color grade: {grade}")
            combined_frames = apply_color_grade(combined_frames, grade)

        # ── Visual effects (Hollywood post-processing) ───────────────────
        visual_effects = config.get("visual_effects", [])
        # Add contrast protection if enabled and not already in the list
        if enable_contrast_protect and "contrast_protect" not in visual_effects:
            visual_effects = list(visual_effects) + ["contrast_protect"]
        if visual_effects:
            print(f"[AutoEditor] ✨ Applying {len(visual_effects)} visual effects...")
            combined_frames = apply_visual_effects(combined_frames, visual_effects)

        if memory_plan.get("guard_mode") == "full_source_for_standard_tiktok_ad":
            target_final_frames = int(memory_plan.get("source_frames", total_source_frames))
            if target_final_frames > 0 and combined_frames.shape[0] != target_final_frames:
                before_frames = combined_frames.shape[0]
                combined_frames = self._match_frame_count(combined_frames, target_final_frames)
                print(
                    f"[AutoEditor] Duration lock restored standard ad length: "
                    f"{before_frames} -> {target_final_frames} frames"
                )
                tweaks_applied.append(
                    f"Duration lock: restored {before_frames}->{target_final_frames} frames"
                )

        # ── Sanitize audio ────────────────────────────────────────────────
        target_audio_samples = int(combined_frames.shape[0] / fps * sample_rate)
        if combined_audio.shape[-1] != target_audio_samples:
            before_samples = combined_audio.shape[-1]
            combined_audio = self._match_audio_samples(
                combined_audio, target_audio_samples
            )
            print(
                f"[AutoEditor] Audio/video sync: {before_samples} -> "
                f"{target_audio_samples} samples"
            )
        combined_audio = self._sanitize_audio(combined_audio, "final")

        # ── Build outputs ─────────────────────────────────────────────────
        audio_output = {
            "waveform": combined_audio.unsqueeze(0),
            "sample_rate": sample_rate,
        }

        final_frames = combined_frames.shape[0]
        final_duration = final_frames / fps
        recommended_bpm = self._recommend_bpm(config, all_images, source_scores)
        config["recommended_bpm"] = recommended_bpm
        recommended_keyscale, recommended_timesignature, recommended_music_tags = (
            self._finalize_music_direction(config, recommended_bpm, lyrics_text)
        )

        video_info_output = {
            "loaded_fps": fps,
            "loaded_frame_count": final_frames,
            "loaded_duration": final_duration,
            "loaded_width": target_w,
            "loaded_height": target_h,
            "source_fps": fps,
            "source_frame_count": final_frames,
            "source_duration": final_duration,
            "source_width": target_w,
            "source_height": target_h,
        }

        # ── Build edit report ─────────────────────────────────────────────
        edit_report = self._build_report(
            mood, llm_model, llm_prompt, config, grade,
            cfg_transition_frames, effective_transition_intensity,
            all_images, shuffled_chunks, frame_segments,
            final_frames, final_duration, fps,
            target_w, target_h, total_source_frames,
            tweaks_applied, features_on,
            quality_mode, memory_plan, source_details, arc_plan
        )

        # ── Console summary ───────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[AutoEditor] ✅ DONE — FRAGMENT & SHUFFLE")
        print(f"[AutoEditor]   Output: {final_frames} frames ({final_duration:.2f}s) at {fps}fps")
        print(f"[AutoEditor]   Recommended song BPM: {recommended_bpm}")
        print(f"[AutoEditor]   Recommended keyscale: {recommended_keyscale}")
        print(f"[AutoEditor]   Recommended time signature: {recommended_timesignature}")
        print(f"[AutoEditor]   Resolution: {target_w}x{target_h}")
        print(f"[AutoEditor]   Mood: {mood}")
        print(f"[AutoEditor]   Chunks: {len(shuffled_chunks)} fragments from {n_videos} videos")
        print(f"[AutoEditor]   Effects: {', '.join(features_on) if features_on else 'none'}")
        print(f"{'='*60}\n")

        # ── Build Florence-2 description outputs ─────────────────────────
        raw_descriptions = ""
        if vision_descriptions:
            lines = []
            for vid_idx in sorted(vision_descriptions.keys()):
                n_fr = all_images[vid_idx].shape[0] if vid_idx in all_images else 0
                dur_s = n_fr / fps if fps > 0 else 0
                lines.append(f"Video {vid_idx} ({n_fr} frames, {dur_s:.1f}s):")
                for label, caption in vision_descriptions[vid_idx]:
                    lines.append(f"  {label}: {caption}")
                lines.append("")
            raw_descriptions = "\n".join(lines)
        else:
            raw_descriptions = f"(Vision analysis was disabled or unavailable)\nError: {vision_error}"

        return (
            combined_frames, audio_output, video_info_output, edit_report,
            raw_descriptions, int(final_frames), int(recommended_bpm),
            recommended_keyscale, int(recommended_timesignature), recommended_music_tags
        )

    # ─── Report builder ──────────────────────────────────────────────────

    def _build_report(self, mood, llm_model, llm_prompt, config, grade,
                      cfg_transition_frames, effective_transition_intensity,
                      all_images, shuffled_chunks, frame_segments,
                      final_frames, final_duration, fps,
                      target_w, target_h, total_source_frames,
                      tweaks_applied, features_on,
                      quality_mode, memory_plan, source_details, arc_plan):
        r = []
        r.append("═" * 50)
        r.append(f"🎥 AUTO EDITOR {AUTOEDITOR_NODE_VERSION} — CREATIVE REPORT")
        r.append("═" * 50)
        r.append("")

        # AI Director notice
        if mood == "llm_auto":
            r.append("⚠️ AI DIRECTOR MODE: ON")
            r.append("─" * 40)
            r.append(f"Quality mode: {quality_mode.upper()} (always on)")
            r.append(f"Chosen mood: {config.get('selected_mood', 'cinematic')}")
            r.append("The prompt adds direction on top of the Auto Director's mood.")
            r.append(f"Strategy: {config.get('edit_strategy', 'premium commercial arc')}")
            r.append("")

        # If LLM provided a narrative, use it prominently
        llm_narrative = config.get("edit_narrative", "")
        if llm_narrative and mood == "llm_auto":
            r.append("🎨 CREATIVE DIRECTOR'S VISION")
            r.append("─" * 40)
            r.append(llm_narrative)
            r.append("")
            r.append(f'💬 Your brief: "{llm_prompt}"')
            r.append(f"🤖 Edited by: {llm_model}")
            r.append("")
        else:
            # Use mood description as narrative
            r.append("🎨 EDITING STYLE")
            r.append("─" * 40)
            r.append(self.MOOD_DESCRIPTIONS.get(mood, f"Mode: {mood}"))
            r.append("")

        r.append("🎯 COMMERCIAL ARC")
        r.append("─" * 40)
        r.append(f"  Hook source: video {arc_plan.get('hook_video', '?')}")
        r.append(f"  Hero ending source: video {arc_plan.get('hero_video', '?')}")
        r.append(f"  Guard mode: {memory_plan.get('guard_mode', 'unknown')}")
        r.append(
            f"  Target duration: {memory_plan.get('min_output_seconds', 0):.1f}s - "
            f"{memory_plan.get('max_output_seconds', 0):.1f}s "
            f"({memory_plan.get('min_output_frames', '?')}-"
            f"{memory_plan.get('max_output_frames', '?')} frames)"
        )
        if arc_plan.get("trimmed"):
            r.append(
                f"  Safety selection: {arc_plan.get('selected_frames', '?')} "
                f"of {arc_plan.get('source_frames', '?')} candidate frames"
            )
        else:
            r.append("  Safety selection: all candidate frames fit the premium budget")
        r.append("")

        if source_details:
            r.append("🔍 SOURCE QUALITY")
            r.append("─" * 40)
            for idx in sorted(source_details):
                r.append(f"  Video {idx}: {source_details[idx]}")
            r.append("")

        # Tweaks
        if tweaks_applied:
            r.append("🔧 YOUR ADJUSTMENTS")
            r.append("─" * 40)
            for t in tweaks_applied:
                r.append(f"  {t}")
            r.append("")

        # Fragmentation info
        r.append("🔪 FRAGMENTATION")
        r.append("─" * 40)
        r.append(f"  Chunk size: {config.get('min_chunk_duration', '?')}s - {config.get('max_chunk_duration', '?')}s")
        r.append(f"  Shuffle intensity: {config.get('shuffle_intensity', '?'):.0%}")
        r.append(f"  Total fragments: {len(shuffled_chunks)}")
        if features_on:
            r.append(f"  Active effects: {', '.join(features_on)}")
        r.append("")

        # The look
        grade_descs = {
            'none': 'Raw footage — untouched original colors.',
            'cinematic_warm': 'Cinematic Warm — golden hour glow with theatrical vignette.',
            'cinematic_cool': 'Cinematic Cool — cold steel tones, thriller atmosphere.',
            'high_contrast': 'High Contrast — crushed blacks, blazing whites, maximum punch.',
            'vivid_pop': 'Vivid Pop — saturated colors that leap off the screen.',
            'moody_dark': 'Moody Dark — noir shadows, lifted fog, brooding atmosphere.',
            'clean_bright': 'Clean Bright — fresh morning light, crisp clarity.',
            'teal_orange': 'Teal & Orange — Hollywood split-tone, cinema standard.',
            'pastel_soft': 'Pastel Soft — dreamlike haze, lifted shadows, ethereal.',
            'vintage_warm': 'Vintage Warm — golden nostalgia with analog film grain.',
            'hollywood': 'Hollywood — Full blockbuster pipeline: teal-orange, bloom, lens dispersion, anamorphic streaks, film grain.',
            'blockbuster': 'Blockbuster — High-impact action grade: razor contrast, chromatic aberration, cinematic grain.',
        }
        vfx = config.get("visual_effects", [])
        r.append("✨ THE LOOK")
        r.append("─" * 40)
        r.append(f"  Color: {grade_descs.get(grade, grade)}")
        if vfx:
            vfx_names = {
                'chromatic_aberration': 'Lens Dispersion',
                'bloom': 'Highlight Bloom',
                'film_grain': 'Film Grain',
                'anamorphic_streak': 'Anamorphic Lens Streaks',
                'heavy_vignette': 'Cinematic Vignette',
            }
            vfx_readable = [vfx_names.get(v, v) for v in vfx]
            r.append(f"  Effects: {', '.join(vfx_readable)}")
        r.append("")

        r.append("MUSIC DIRECTION")
        r.append("-" * 40)
        r.append(f"  BPM: {config.get('recommended_bpm', 'see INT output')}")
        r.append(f"  Keyscale: {config.get('recommended_keyscale', 'C major')}")
        r.append(f"  Time signature: {config.get('recommended_timesignature', 4)}")
        r.append(f"  ACE-Step tags: {config.get('recommended_music_tags', '')}")
        r.append("")

        # Results (concise)
        total_source_dur = sum(all_images[idx].shape[0] / fps for idx in all_images)

        r.append("📊 RESULT")
        r.append("─" * 40)
        r.append(f"  {final_duration:.1f}s output from {total_source_dur:.1f}s of source footage")
        r.append(f"  {len(frame_segments)} cuts at {fps}fps • {target_w}×{target_h}")
        r.append(f"  Output frames: {final_frames}")
        r.append(f"  Duration policy: {config.get('duration_policy', 'flexible_punchy_sales_edit')}")
        r.append(f"  Recommended song BPM: {config.get('recommended_bpm', 'see INT output')}")
        if arc_plan.get("trimmed"):
            r.append("  Large source protected by memory-safe commercial selection")
        else:
            r.append(f"  All selected frames from all {len(all_images)} videos used")
        r.append("")
        r.append("═" * 50)

        return "\n".join(r)

    # ─── LLM config helper ───────────────────────────────────────────────

    @staticmethod
    def _apply_premium_director_defaults(config, llm_prompt, memory_plan):
        """Premium is always on: make LLM settings safer, cleaner, and more commercial."""
        config = dict(config or {})
        notes = ["Quality mode: premium commercial director"]

        config["quality_mode"] = "premium"
        config["edit_strategy"] = "premium TikTok sales arc: hook -> proof/detail -> desire -> hero CTA"
        config["duration_policy"] = "flexible_punchy_sales_edit"
        config["max_output_frames"] = memory_plan.get("max_output_frames")
        config["max_output_seconds"] = memory_plan.get("max_output_seconds")

        if config.get("color_grade") in ("none", "high_contrast"):
            config["color_grade"] = "hollywood"
            notes.append("Premium color safety: using hollywood grade instead of harsh/none")

        transitions = config.get("transitions", [])
        if isinstance(transitions, str):
            transitions = [transitions]
        polished_transitions = [
            t for t in transitions
            if t not in ("glitch_cut", "shake_cut", "flash_black")
        ]
        if not polished_transitions:
            polished_transitions = [
                "hard_cut", "cross_dissolve", "luma_fade", "zoom_punch_in"
            ]
            notes.append("Premium transitions: replaced noisy cuts with clean commercial transitions")
        clean_priority = ["hard_cut", "zoom_punch_in", "cross_dissolve", "luma_fade", "swipe_left", "swipe_up"]
        ordered_transitions = [t for t in clean_priority if t in polished_transitions]
        ordered_transitions += [t for t in polished_transitions if t not in ordered_transitions]
        config["transitions"] = ordered_transitions[:5] or ["hard_cut", "zoom_punch_in", "cross_dissolve"]

        vfx = config.get("visual_effects", [])
        if isinstance(vfx, str):
            vfx = [vfx]
        vfx = [e for e in vfx if e not in ("heavy_vignette",)]
        for required in ("bloom", "film_grain", "contrast_protect"):
            if required not in vfx:
                vfx.append(required)
        config["visual_effects"] = vfx[:5]
        config["contrast_protect"] = True

        config["cut_duration_mode"] = "variable"
        config["speed_ramp"] = "none"
        config["speed_factor"] = 1.0
        if not config.get("variable_durations"):
            config["variable_durations"] = "0.55,1.15,0.75,1.6,0.65,2.2,0.9,1.35,2.6"

        def clamp_float(key, lo, hi, default):
            try:
                value = float(config.get(key, default))
            except (TypeError, ValueError):
                value = default
            config[key] = max(lo, min(hi, value))

        clamp_float("min_chunk_duration", 0.38, 0.9, 0.55)
        clamp_float("max_chunk_duration", max(config["min_chunk_duration"], 1.4), 2.8, 2.2)
        clamp_float("shuffle_intensity", 0.38, 0.72, 0.58)
        clamp_float("transition_intensity", 0.45, 0.82, 0.68)
        clamp_float("reverse_chance", 0.0, 0.05, 0.0)
        clamp_float("punch_in_chance", 0.12, 0.32, 0.24)
        clamp_float("burst_frequency", 0.02, 0.16, 0.10)
        clamp_float("black_breath_chance", 0.0, 0.02, 0.0)
        clamp_float("hold_frame_chance", 0.0, 0.04, 0.0)
        clamp_float("micro_ramp_chance", 0.0, 0.18, 0.10)

        if config["max_chunk_duration"] < config["min_chunk_duration"]:
            config["max_chunk_duration"] = config["min_chunk_duration"]

        config["reverse_chance"] = 0.0
        config["black_breath_chance"] = 0.0
        config["hold_frame_chance"] = 0.0
        config["micro_ramp_chance"] = 0.0

        narrative = config.get("edit_narrative", "")
        if not narrative:
            prompt_note = (
                f" The user's extra direction is: {llm_prompt.strip()}"
                if llm_prompt and llm_prompt.strip()
                else ""
            )
            narrative = (
                "Premium TikTok sales director mode is building a punchy persuasive arc: "
                "open fast with the strongest hook, remove bad AI/morphing motion, use the middle "
                "for product texture and proof, and finish on the most desirable cinematic hero angle." + prompt_note
            )
        config["edit_narrative"] = str(narrative)
        return config, notes

    @staticmethod
    def _professional_fallback_config():
        """Premium automatic fallback when the LLM is unavailable."""
        config = MOODS.get("cinematic", MOODS["bold"]).copy()
        config.update({
            "selected_mood": "bold",
            "cut_duration_mode": "variable",
            "variable_durations": "0.55,1.15,0.75,1.6,0.65,2.2,0.9,1.35,2.6",
            "transitions": ["hard_cut", "zoom_punch_in", "cross_dissolve", "luma_fade", "swipe_left"],
            "transition_frames": 6,
            "transition_intensity": 0.68,
            "speed_ramp": "none",
            "speed_factor": 1.0,
            "color_grade": "hollywood",
            "visual_effects": ["bloom", "film_grain", "contrast_protect"],
            "contrast_protect": True,
            "min_chunk_duration": 0.55,
            "max_chunk_duration": 2.2,
            "shuffle_intensity": 0.58,
            "reverse_chance": 0.0,
            "punch_in_chance": 0.24,
            "burst_frequency": 0.10,
            "black_breath_chance": 0.0,
            "hold_frame_chance": 0.0,
            "micro_ramp_chance": 0.0,
            "edit_narrative": (
                "Automatic premium TikTok sales edit: fast hook, cinematic product detail, "
                "clean premium transitions, believable movement, and a desirable hero ending. "
                "Bad AI/morphing motion is removed when detected instead of frozen."
            ),
            "recommended_keyscale": "G minor",
            "recommended_timesignature": 4,
            "recommended_music_tags": (
                "bold modern ad music, punchy drums, confident bass, sharp product hook, "
                "instrumental, no lead vocal, professional TikTok product advertisement, "
                "subtle film grain mood, polished commercial master"
            ),
            "_music_direction_from_llm": False,
        })
        return config

    def _get_llm_config(self, llm_model, llm_prompt, video_summary_lines, all_images,
                        vision_context="", allow_keyframe_fallback=True, lyrics_text=""):
        """Ask the LLM for editing configuration.
        If vision_context is provided, uses text-only mode (Florence-2 descriptions).
        Otherwise falls back to sending keyframe images."""
        default_prompt = "Optional: guide the AI (e.g. 'make it energetic and fast-paced'). Leave empty for fully automatic editing."

        if not llm_prompt or not llm_prompt.strip() or llm_prompt.strip() == default_prompt:
            if vision_context:
                # AI Director mode with no prompt — generate from content
                llm_prompt = ("You are the AI Director. Analyze the video content descriptions below "
                              "and decide the BEST editing style for this product advertisement. "
                              "Choose pacing, transitions, color grading, and effects that will "
                              "maximize sales impact based on what you see in the footage.")
                print("[AutoEditor] 🤖 AI Director: No prompt given — LLM deciding from video content alone")
            else:
                print("[AutoEditor] LLM has no prompt or vision context - using premium fallback director config")
                return self._professional_fallback_config()

        if "(ollama" in llm_model or "(no model" in llm_model:
            print(f"[AutoEditor] Ollama unavailable ({llm_model}) - using premium fallback director config")
            return self._professional_fallback_config()

        video_info_str = "\n".join(video_summary_lines)

        # ── PREFERRED: Vision-directed (text-only, no images to LLM) ──────
        if vision_context:
            print(f"[AutoEditor] 👁️ Using Vision Director mode (Florence-2 descriptions)")
            config = ask_ollama_with_descriptions(
                llm_model, llm_prompt, video_info_str, vision_context,
                lyrics_text=lyrics_text,
            )
            if config is not None:
                if "cut_pattern" not in config:
                    config["cut_pattern"] = "1,2,3,4,5,6"
                print(f"[AutoEditor] 🤖 LLM config received (vision-directed):")
                for k, v in config.items():
                    print(f"[AutoEditor]   {k}: {v}")
                return config
            print(f"[AutoEditor] ⚠️ Vision-directed query failed — falling back to image mode")

        if not allow_keyframe_fallback:
            print("[AutoEditor] Video understanding OFF - using text-only LLM direction")
            config = ask_ollama(
                llm_model, llm_prompt, video_info_str, lyrics_text=lyrics_text
            )
            if config is None:
                print("[AutoEditor] Text-only LLM returned no valid config - using premium fallback director config")
                return self._professional_fallback_config()
            if "cut_pattern" not in config:
                config["cut_pattern"] = "1,2,3,4,5,6"
            print(f"[AutoEditor] 🤖 LLM config received (text-only):")
            for k, v in config.items():
                print(f"[AutoEditor]   {k}: {v}")
            return config

        # ── FALLBACK: Send keyframe images to multimodal LLM ──────────────
        keyframe_tensors = []
        keyframe_descriptions = []
        for idx in sorted(all_images.keys()):
            frames = all_images[idx]
            n = frames.shape[0]
            keyframe_tensors.append(frames[0])
            keyframe_descriptions.append(f"Video {idx} — FIRST frame (frame 1/{n})")
            if n > 1:
                keyframe_tensors.append(frames[-1])
                keyframe_descriptions.append(f"Video {idx} — LAST frame (frame {n}/{n})")

        print(f"[AutoEditor] 📷 Extracted {len(keyframe_tensors)} keyframes for visual analysis")
        keyframe_b64 = frames_to_base64(keyframe_tensors, max_size=256)

        config = ask_ollama(
            llm_model, llm_prompt, video_info_str,
            keyframe_images=keyframe_b64 if keyframe_b64 else None,
            keyframe_descriptions=keyframe_descriptions if keyframe_b64 else None,
            lyrics_text=lyrics_text,
        )

        if config is None:
            print("[AutoEditor] LLM returned no valid config - using premium fallback director config")
            return self._professional_fallback_config()

        if "cut_pattern" not in config:
            config["cut_pattern"] = "1,2,3,4,5,6"

        print(f"[AutoEditor] 🤖 LLM config received:")
        for k, v in config.items():
            print(f"[AutoEditor]   {k}: {v}")

        return config

    # ─── Audio crossfade ─────────────────────────────────────────────────

    @staticmethod
    def _crossfade_audio(audio_a, audio_b, crossfade_samples):
        if crossfade_samples <= 0:
            return torch.cat([audio_a, audio_b], dim=-1)
        cf = min(crossfade_samples, audio_a.shape[-1], audio_b.shape[-1])
        if cf <= 0:
            return torch.cat([audio_a, audio_b], dim=-1)
        fade_out = torch.linspace(1, 0, cf)
        fade_in = torch.linspace(0, 1, cf)
        blended = audio_a[..., -cf:] * fade_out + audio_b[..., :cf] * fade_in
        return torch.cat([audio_a[..., :-cf], blended, audio_b[..., cf:]], dim=-1)


# ─── Node Registration ───────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DJ_AutoEditor": DJ_AutoEditor,
    # Compatibility alias for workflows saved while the node was named Auto Director.
    "DJ_AutoDirector": DJ_AutoEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_AutoEditor": f"Auto Editor {AUTOEDITOR_NODE_VERSION} 🎬",
    "DJ_AutoDirector": f"Auto Editor {AUTOEDITOR_NODE_VERSION} 🎬",
}
