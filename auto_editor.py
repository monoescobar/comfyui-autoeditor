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

from .presets import MOODS, get_mood_names
from .transitions import join_segments
from .color_grading import apply_color_grade, get_grade_names, apply_visual_effects
from .ollama_bridge import list_ollama_models, ask_ollama, ask_ollama_with_descriptions, frames_to_base64
from .vision_analysis import analyze_videos, format_descriptions_for_llm, detect_distortions, remove_distorted_frames, get_vision_quality_names


class DJ_AutoEditor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        mood_list = get_mood_names() + ["llm_auto"]

        try:
            ollama_models = list_ollama_models()
        except Exception:
            ollama_models = ["(ollama offline)"]

        return {
            "required": {
                "ai_director": (["ON", "OFF"], {
                    "default": "ON",
                    "tooltip": "\u26a0\ufe0f AI DIRECTOR: When ON, the AI controls ALL editing decisions based on video content. All manual settings below are IGNORED. Set to OFF for manual control.",
                }),
                "llm_model": (ollama_models, {"default": ollama_models[0]}),
                "mood": (mood_list, {"default": "bold"}),
                "llm_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: guide the AI (e.g. 'make it energetic and fast-paced'). Leave empty for fully automatic editing.",
                }),
                "images1": ("IMAGE",),
                "video_info1": ("VHS_VIDEOINFO",),
                "images2": ("IMAGE",),
                "video_info2": ("VHS_VIDEOINFO",),
            },
            "optional": {
                # Pacing & transition overrides
                "pacing": (["as_mood", "slower", "much_slower", "faster", "much_faster"], {
                    "default": "as_mood",
                    "tooltip": "Override the pacing: affects speed and cut durations together",
                }),
                "transition_strength": (["as_mood", "subtle", "moderate", "dramatic", "extreme"], {
                    "default": "as_mood",
                    "tooltip": "Override transition intensity and duration",
                }),
                "audio_crossfade_ms": ("INT", {
                    "default": 0, "min": 0, "max": 500, "step": 10,
                    "tooltip": "Audio blend between cuts in milliseconds. 0=sharp cut",
                }),
                # ── Feature toggles ──────────────────────────────────
                "reverse_clips": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Randomly reverse some fragment playback",
                }),
                "scale_punch_ins": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Random subtle zoom crop (102-108%) on fragments",
                }),
                "rhythm_bursts": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Insert rapid-fire 3-4 cut burst sequences",
                }),
                "black_breaths": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Insert 2-4 black frames between burst segments",
                }),
                "hold_frames": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Freeze last frame of some clips for impact",
                }),
                "jump_cuts": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Reorder fragments within same video for jump-cut feel",
                }),
                "micro_speed_ramps": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Per-clip slow-motion → snap speed ramps",
                }),
                "vision_analysis": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Use Florence-2 AI to analyze video content for smarter editing decisions",
                }),
                "vision_quality": (get_vision_quality_names(), {
                    "default": "fast",
                    "tooltip": "Vision analysis speed/quality: detailed (slowest) → balanced → fast (recommended) → turbo (fastest)",
                }),
                "distortion_removal": (["skip_frames", "freeze_frames", "disable"], {
                    "default": "freeze_frames",
                    "tooltip": "Detect and handle AI-generated distortion frames. Skip=shorter video, Freeze=replace with clean frame",
                }),
                "contrast_protect": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Soft contrast limiter — prevents crushed blacks and blown highlights in the final video",
                }),
                "phased_fragmentation": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Multi-phase rhythm: alternates between rapid bursts and breathing moments for dramatic frame count variation",
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
    RETURN_NAMES = ("images_output", "audio_output", "video_info_output", "edit_report", "vision_descriptions")
    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_VIDEOINFO", "STRING", "STRING")

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
        indices = torch.linspace(0, n_in - 1, n_out).long()
        return frames[indices]

    @staticmethod
    def _speed_ramp_audio(audio, speed):
        if audio is None or abs(speed - 1.0) < 0.01:
            return audio
        n_in = audio.shape[-1]
        n_out = max(1, int(n_in / speed))
        indices = torch.linspace(0, n_in - 1, n_out).long()
        return audio[..., indices]

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
        slow_indices = torch.linspace(0, split - 1, n_slow_out).long()
        slow_stretched = slow_part[slow_indices]
        # Compress fast part to maintain total ≈ original
        n_fast_out = max(1, n - n_slow_out)
        fast_indices = torch.linspace(0, fast_part.shape[0] - 1, n_fast_out).long()
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
        cropped = frames[:, y0:y0 + crop_h, x0:x0 + crop_w, :]
        # Resize back to original
        cropped_p = cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(cropped_p, size=(h, w), mode='bilinear', align_corners=False)
        return resized.permute(0, 2, 3, 1)

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
        s = images.permute(0, 3, 1, 2)
        s = comfy.utils.common_upscale(s, target_w, target_h, "bilinear", "center")
        return s.permute(0, 2, 3, 1)

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
        min_burst_frames = max(3, int(0.3 * fps))  # Minimum burst sub-cut: 0.3s

        for chunk in chunks:
            vid_idx, start, end = chunk
            chunk_frames = end - start

            # Only burst-split chunks that are long enough
            if chunk_frames > int(fps * 1.0) and random.random() < burst_frequency:
                # Split into 3-4 rapid sub-cuts with VARIED sizes
                n_sub = random.randint(3, 4)
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

    def auto_edit(self, ai_director, llm_model, mood, llm_prompt,
                  images1, video_info1, images2, video_info2,
                  **kwargs):

        # ── AI Director Mode ──────────────────────────────────────────────
        ai_director_on = (ai_director == "ON")
        if ai_director_on:
            print(f"\n{'='*60}")
            print(f"[AutoEditor] \u26a0\ufe0f  AI DIRECTOR MODE: ON")
            print(f"[AutoEditor]    All manual settings (mood, pacing, toggles) are IGNORED.")
            print(f"[AutoEditor]    The AI will analyze your videos and decide everything.")
            if llm_prompt and llm_prompt.strip():
                print(f"[AutoEditor]    Your guidance: \"{llm_prompt.strip()[:100]}\"")
            else:
                print(f"[AutoEditor]    No prompt given \u2014 fully automatic mode.")
            print(f"{'='*60}")
            # Force LLM auto mode
            mood = "llm_auto"

        # Extract tweaks with safe defaults
        if ai_director_on:
            # AI Director overrides: ignore manual settings, enable everything
            pacing_mode = "as_mood"  # LLM controls pacing directly
            transition_strength = "as_mood"  # LLM controls transitions directly
            audio_crossfade_ms = 50  # Always use a small crossfade for clean audio
            enable_reverse = True
            enable_punch_in = True
            enable_bursts = True
            enable_black_breath = True
            enable_hold = True
            enable_jump_cuts = True
            enable_micro_ramp = True
            enable_vision = True
            vision_quality = kwargs.get("vision_quality", "fast")
            distortion_mode = "freeze_frames"
            enable_contrast_protect = True
            enable_phased = True
        else:
            pacing_mode = kwargs.get("pacing", "as_mood")
            transition_strength = kwargs.get("transition_strength", "as_mood")
            audio_crossfade_ms = kwargs.get("audio_crossfade_ms", 0)
            enable_reverse = kwargs.get("reverse_clips", "enable") == "enable"
            enable_punch_in = kwargs.get("scale_punch_ins", "enable") == "enable"
            enable_bursts = kwargs.get("rhythm_bursts", "enable") == "enable"
            enable_black_breath = kwargs.get("black_breaths", "enable") == "enable"
            enable_hold = kwargs.get("hold_frames", "enable") == "enable"
            enable_jump_cuts = kwargs.get("jump_cuts", "enable") == "enable"
            enable_micro_ramp = kwargs.get("micro_speed_ramps", "enable") == "enable"
            enable_vision = kwargs.get("vision_analysis", "enable") == "enable"
            vision_quality = kwargs.get("vision_quality", "fast")
            distortion_mode = kwargs.get("distortion_removal", "freeze_frames")
            enable_contrast_protect = kwargs.get("contrast_protect", "enable") == "enable"
            enable_phased = kwargs.get("phased_fragmentation", "enable") == "enable"

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
                    "loaded_fps": video_info1.get("loaded_fps", 25),
                    "loaded_frame_count": img.shape[0],
                }

        n_videos = len(all_images)

        # ── Get FPS ───────────────────────────────────────────────────────
        fps = all_info[1].get("loaded_fps", 25)
        if isinstance(fps, str):
            fps = float(fps)

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
            vision_error = "Vision analysis is disabled (toggle is OFF)"

        # ── Log source video info ─────────────────────────────────────────
        total_source_frames = 0
        print(f"\n{'='*60}")
        print(f"[AutoEditor] 🎬 Starting FRAGMENT & SHUFFLE edit with {n_videos} source videos")
        video_summary_lines = []
        for idx in sorted(all_images.keys()):
            n_frames = all_images[idx].shape[0]
            total_source_frames += n_frames
            vid_fps = all_info[idx].get("loaded_fps", 25) if idx in all_info else 25
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

        # ── Get configuration ─────────────────────────────────────────────
        if mood == "llm_auto":
            config = self._get_llm_config(
                llm_model, llm_prompt, video_summary_lines, all_images,
                vision_context=vision_context
            )
        else:
            config = MOODS.get(mood, MOODS["bold"]).copy()
            print(f"[AutoEditor] 🎭 Mood: '{mood}'")

        # ── Apply pacing override ─────────────────────────────────────────
        tweaks_applied = []

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

            # Extract frames
            frames = all_images[vid_idx][start:end].clone()

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

        print(f"[AutoEditor] Joining {len(frame_segments)} segments with transitions...")
        combined_frames = frame_segments[0]
        combined_audio = audio_segments[0]

        for i in range(1, len(frame_segments)):
            t_type = transitions_list[(i - 1) % len(transitions_list)]
            # Black breath: randomly override transition with black_breath
            if black_breath_chance > 0 and random.random() < black_breath_chance:
                t_type = "black_breath"
            combined_frames = join_segments(
                combined_frames, frame_segments[i],
                t_type, cfg_transition_frames, effective_transition_intensity
            )
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

        # ── Sanitize audio ────────────────────────────────────────────────
        combined_audio = self._sanitize_audio(combined_audio, "final")

        # ── Build outputs ─────────────────────────────────────────────────
        audio_output = {
            "waveform": combined_audio.unsqueeze(0),
            "sample_rate": sample_rate,
        }

        final_frames = combined_frames.shape[0]
        final_duration = final_frames / fps

        video_info_output = {
            "loaded_fps": fps,
            "loaded_frame_count": final_frames,
            "loaded_duration": final_duration,
            "loaded_width": target_w,
            "loaded_height": target_h,
            "source_fps": video_info1.get("source_fps", fps),
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
            tweaks_applied, features_on
        )

        # ── Console summary ───────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[AutoEditor] ✅ DONE — FRAGMENT & SHUFFLE")
        print(f"[AutoEditor]   Output: {final_frames} frames ({final_duration:.2f}s) at {fps}fps")
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

        return (combined_frames, audio_output, video_info_output, edit_report,
                raw_descriptions)

    # ─── Report builder ──────────────────────────────────────────────────

    def _build_report(self, mood, llm_model, llm_prompt, config, grade,
                      cfg_transition_frames, effective_transition_intensity,
                      all_images, shuffled_chunks, frame_segments,
                      final_frames, final_duration, fps,
                      target_w, target_h, total_source_frames,
                      tweaks_applied, features_on):
        r = []
        r.append("═" * 50)
        r.append("🎥 AUTO EDITOR — CREATIVE REPORT")
        r.append("═" * 50)
        r.append("")

        # AI Director notice
        if mood == "llm_auto":
            r.append("⚠️ AI DIRECTOR MODE: ON")
            r.append("─" * 40)
            r.append("All editing decisions were made automatically")
            r.append("by the AI based on Florence-2 video analysis.")
            r.append("Manual settings (mood, pacing, toggles) were IGNORED.")
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

        # Results (concise)
        total_source_dur = sum(all_images[idx].shape[0] / fps for idx in all_images)

        r.append("📊 RESULT")
        r.append("─" * 40)
        r.append(f"  {final_duration:.1f}s output from {total_source_dur:.1f}s of source footage")
        r.append(f"  {len(frame_segments)} cuts at {fps}fps • {target_w}×{target_h}")
        r.append(f"  ✅ All frames from all {len(all_images)} videos used")
        r.append("")
        r.append("═" * 50)

        return "\n".join(r)

    # ─── LLM config helper ───────────────────────────────────────────────

    def _get_llm_config(self, llm_model, llm_prompt, video_summary_lines, all_images,
                        vision_context=""):
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
                print("[AutoEditor] ⚠️  LLM mode but no custom prompt — falling back to 'bold' mood")
                return MOODS["bold"].copy()

        if "(ollama" in llm_model or "(no model" in llm_model:
            print(f"[AutoEditor] ⚠️  Ollama unavailable ({llm_model}) — falling back to 'bold'")
            return MOODS["bold"].copy()

        video_info_str = "\n".join(video_summary_lines)

        # ── PREFERRED: Vision-directed (text-only, no images to LLM) ──────
        if vision_context:
            print(f"[AutoEditor] 👁️ Using Vision Director mode (Florence-2 descriptions)")
            config = ask_ollama_with_descriptions(
                llm_model, llm_prompt, video_info_str, vision_context
            )
            if config is not None:
                if "cut_pattern" not in config:
                    config["cut_pattern"] = "1,2,3,4,5,6"
                print(f"[AutoEditor] 🤖 LLM config received (vision-directed):")
                for k, v in config.items():
                    print(f"[AutoEditor]   {k}: {v}")
                return config
            print(f"[AutoEditor] ⚠️ Vision-directed query failed — falling back to image mode")

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
        )

        if config is None:
            print("[AutoEditor] ⚠️  LLM returned no valid config — falling back to 'bold'")
            return MOODS["bold"].copy()

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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_AutoEditor": "Auto Editor 🎬",
}
