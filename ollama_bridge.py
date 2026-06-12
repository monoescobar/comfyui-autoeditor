"""
Ollama LLM bridge for DJ_AutoEditor — with vision support.

Queries a local Ollama instance to generate edit configurations
including fragmentation, shuffle, and per-chunk effect parameters.
Can send keyframe images from source videos so multimodal models
(like Gemma 4) can analyze visual content and make smarter edits.
"""

import io
import json
import base64
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"


def list_ollama_models():
    """Query Ollama REST API for installed models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            names = sorted(set(
                m.get("name", m.get("model", ""))
                for m in data.get("models", [])
            ))
            return [n for n in names if n] or ["(no models found)"]
    except Exception:
        return ["(ollama offline)"]


def frames_to_base64(frame_tensors, max_size=256):
    """
    Convert a list of torch tensors [H,W,C] to base64 PNG strings.
    Resizes to max_size for bandwidth efficiency.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("[AutoEditor-LLM] PIL not available — skipping image analysis")
        return []

    images_b64 = []
    for tensor in frame_tensors:
        try:
            np_arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            img = Image.fromarray(np_arr)
            # Resize to save bandwidth
            w, h = img.size
            scale = min(max_size / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        except Exception as e:
            print(f"[AutoEditor-LLM] Failed to encode frame: {e}")
    return images_b64


def ask_ollama(model, user_prompt, video_summaries, keyframe_images=None, keyframe_descriptions=None):
    """
    Send a prompt + optional keyframe images to Ollama for edit config.

    Args:
        model: Ollama model name
        user_prompt: User's editing style description
        video_summaries: Text describing source videos
        keyframe_images: List of base64 PNG strings (first/last frames)
        keyframe_descriptions: List of strings describing each image

    Returns:
        dict with edit configuration, or None on failure.
    """

    available_transitions = [
        "hard_cut", "flash_white", "flash_black", "zoom_punch_in",
        "zoom_punch_out", "glitch_cut", "whip_pan", "cross_dissolve",
        "luma_fade", "swipe_left", "swipe_up", "shake_cut",
    ]
    available_grades = [
        "none", "cinematic_warm", "cinematic_cool", "high_contrast",
        "vivid_pop", "moody_dark", "clean_bright", "teal_orange",
        "pastel_soft", "vintage_warm", "hollywood", "blockbuster",
    ]
    available_effects = [
        "chromatic_aberration", "bloom", "film_grain",
        "anamorphic_streak", "heavy_vignette", "contrast_protect",
    ]
    available_speeds = [
        "none", "slow_motion", "speed_up", "fast_start",
        "speed_up_end", "accelerate",
    ]
    available_moods = [
        "dreamy", "romantic", "calm", "nature", "elegant", "warm",
        "minimal", "playful", "retro", "cinematic", "bold", "luxury",
        "urban", "energetic", "intense", "hypnotic", "raw", "neon",
        "editorial", "chaos",
    ]

    # Build visual context if keyframes are available
    visual_context = ""
    if keyframe_descriptions:
        visual_context = "\nVISUAL ANALYSIS OF SOURCE FOOTAGE:\n"
        visual_context += "(I'm showing you the first and last frame of each video)\n"
        for desc in keyframe_descriptions:
            visual_context += f"  {desc}\n"
        visual_context += (
            "\nUse this visual information to choose appropriate transitions:\n"
            "  - Similar brightness/color between clips → cross_dissolve or luma_fade\n"
            "  - High contrast between clips → flash_white or flash_black\n"
            "  - Motion in the footage → whip_pan or swipe transitions\n"
            "  - Static/product shots → zoom_punch_in for impact\n"
            "  - Dark/moody footage → luma_fade or flash_black\n"
            "  - Bright/energetic footage → flash_white or swipe transitions\n"
        )

    system = f"""You are a world-class product commercial director, film editor, and advertising creative director. You create premium edits that sell the product while keeping it believable, clean, and cinematic.

SOURCE VIDEOS:
{video_summaries}
{visual_context}
DIRECTOR BRIEF:
1. This is ALWAYS a TikTok sales ad, but it must also feel luxury, cinematic, premium, and professional.
2. First choose the best hidden mood from this list: {json.dumps(available_moods)}.
3. Use the selected mood as the base sentiment/edit language. The user's prompt is ADDITIVE DIRECTION that refines and elevates the mood, not a replacement.
4. Decide how much TikTok sales energy, luxury, cinematic drama, and premium polish to apply from 25% to 100% each, based on the footage.
5. Detect video-model problems: bad morphing, weird animation, ugly movement, distorted hands/product, fake-looking motion, or frames that feel like still images. Avoid or minimize those moments.
6. The edit must make the product look desirable, real, premium, and easy to buy.
7. Think like a commercial director: build hook -> product proof/detail -> emotional desire -> hero ending.

PROFESSIONAL EDITING PRINCIPLES:
1. Editing should feel intentional and mostly invisible. Avoid random effects that call attention to the edit instead of the product.
2. Vary shot lengths musically and keep the edit punchy. Use short accents around motion/reveals and brief breathing cuts on product hero shots.
3. INTERLEAVE videos creatively — don't just go 1,2,3,4,5,6. Create callbacks and reveals.
3. Each transition serves a purpose:
   - cross_dissolve: Elegant blend. Best when shots share similar lighting.
   - luma_fade: Sophisticated brightness reveal. Premium luxury look.
   - flash_white: Energy burst. Use sparingly for key impact moments.
   - flash_black: Dramatic breath. Weight and separation between scenes.
   - zoom_punch_in: Product reveal impact. Punching into the next shot.
   - zoom_punch_out: Subtle pull-back for establishing context.
   - whip_pan: Fast horizontal blur. Urgency and action.
   - swipe_left / swipe_up: Clean directional wipes. Modern and polished.
   - shake_cut: Camera impact. Action and sports energy.
   - glitch_cut: Digital distortion. Tech, gaming, edgy content.
   - hard_cut: Clean cut. Use between naturally flowing compositions.
4. Prefer clean transitions: hard_cut, cross_dissolve, luma_fade, zoom_punch_in, subtle swipe. Use flash/glitch/shake only for rare emphasis.
5. Choose cinematic color grades that preserve product detail and skin/material believability.
6. Visual effects must be restrained: bloom, film_grain, contrast_protect are usually enough. Use chromatic aberration lightly.

FRAGMENT & SHUFFLE SYSTEM:
This editor chops each source video into small fragments (chunks) and shuffles them across the timeline.
- min/max_chunk_duration controls fragment size (smaller = more dynamic, bigger = calmer)
- shuffle_intensity controls how mixed the fragments are (0 = sequential, 1 = fully random)
- For FAST/ENERGETIC edits: min 0.3-0.8s, max 1.0-2.0s, shuffle 0.7-1.0
- For CALM/LUXURY edits: min 1.5-3.0s, max 3.0-5.0s, shuffle 0.2-0.4
- For BALANCED edits: min 0.8-1.5s, max 2.0-3.5s, shuffle 0.5-0.7
- ALWAYS use some shuffle (at least 0.3) — pure sequential looks amateurish

CRITICAL CONTRAST RULES:
- NEVER use "high_contrast" as color_grade — it looks harsh and amateur.
- Prefer "hollywood", "cinematic_warm", or "teal_orange" for premium look.
- ALWAYS include "contrast_protect" in visual_effects to prevent crushed blacks/blown highlights.
- Keep the final look SOFT and CINEMATIC, never harsh or blown out.

RHYTHM VARIATION:
- Every 3-5 seconds, the rhythm should evolve, not stutter.
- A professional rhythm is: reveal -> product detail -> motion accent -> breathing hero shot -> supporting angle -> memorable close.
- Use short cuts around action (0.5-1.0s), medium cuts for context (1.2-2.0s), and hero cuts for desirability (2.5-4.0s).
- For TikTok sales, prefer energetic commercial rhythm: most cuts 0.45-1.8s, with only occasional longer hero moments.
- Avoid slow edits that feel like still images. Never stretch weak footage just to preserve duration.
- Avoid excessive 0.2-0.4s cutting unless the footage is truly music/action driven.
- The variable_durations string should have smooth variation (e.g. "1.0,2.6,0.8,1.6,3.0,0.9,2.2,1.2,3.5")

PER-CHUNK EFFECTS (use sparingly for professional results):
- reverse_chance: usually 0.0; only use when it still looks believable.
- punch_in_chance: 0.08-0.30 for subtle camera movement and product emphasis.
- burst_frequency: 0.05-0.35; bursts should be accents, not the whole edit.
- black_breath_chance: usually 0.0; black frames often feel choppy in product ads.
- hold_frame_chance: usually 0.0-0.08; freezes can make AI footage feel fake.
- micro_ramp_chance: 0.05-0.25 for polished energy.

ALL PARAMETERS (include EVERY key in your JSON response):

0. "selected_mood": one of {json.dumps(available_moods)}
1. "cut_duration_mode": "variable" (always use variable for professional rhythm)
2. "fixed_cut_duration": float 0.5-10.0
3. "variable_durations": comma-separated floats with polished variation (e.g. "1.0,2.6,0.8,1.6,3.0,0.9,2.2,1.2,3.5")
4. "cut_pattern": comma-separated video numbers — interleave CREATIVELY with callbacks
5. "transitions": JSON array of 4-6 different transitions to cycle through
   Available: {json.dumps(available_transitions)}
6. "transition_frames": int 3-14 (bigger = smoother, 8-12 is cinematic)
7. "speed_ramp": one of {json.dumps(available_speeds)}
8. "speed_factor": float 0.25-4.0
9. "color_grade": one of {json.dumps(available_grades)} — prefer "hollywood" or "cinematic_warm", NEVER "high_contrast"
10. "transition_intensity": float 0.7-1.0 (professional range)
11. "visual_effects": JSON array — ALWAYS include ["bloom", "chromatic_aberration", "contrast_protect"]
    Available: {json.dumps(available_effects)}
12. "min_chunk_duration": float 0.2-5.0 (minimum fragment size in seconds)
13. "max_chunk_duration": float 0.5-5.0 (maximum fragment size in seconds, must be >= min)
14. "shuffle_intensity": float 0.0-1.0 (0=sequential, 1=fully random interleaving)
15. "reverse_chance": float 0.0-0.15
16. "punch_in_chance": float 0.0-0.35
17. "burst_frequency": float 0.0-0.35
18. "black_breath_chance": float 0.0-0.08
19. "hold_frame_chance": float 0.0-0.12
20. "micro_ramp_chance": float 0.0-0.25
21. "contrast_protect": boolean true/false — enable soft contrast protection (recommend true)
22. "edit_narrative": A 3-5 sentence PERSONAL explanation of your creative vision. Explain the chosen mood, how the user's prompt modified it, how much TikTok/luxury/cinematic/premium energy you applied, what weak AI-video motion should be avoided, the hook, the proof/detail section, and the hero ending. Write as a creative director explaining their vision to a client.

RESPOND WITH ONLY A VALID JSON OBJECT. No markdown, no explanation outside the JSON."""

    try:
        payload_dict = {
            "model": model,
            "system": system,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "num_predict": 1500,
            },
        }

        # Add keyframe images for multimodal models
        if keyframe_images:
            payload_dict["images"] = keyframe_images
            print(f"[AutoEditor-LLM] 📷 Sending {len(keyframe_images)} keyframe images to model")

        payload = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"[AutoEditor-LLM] 🤖 Querying '{model}'...")
        print(f"[AutoEditor-LLM] Prompt: \"{user_prompt[:200]}\"")

        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            response_text = raw.get("response", "")
            print(f"[AutoEditor-LLM] Response: {response_text[:500]}")

            config = json.loads(response_text)
            config = _sanitize_config(
                config, available_transitions, available_grades, available_speeds
            )

            print(f"[AutoEditor-LLM] ✅ Config parsed successfully")
            return config

    except json.JSONDecodeError as e:
        print(f"[AutoEditor-LLM] ❌ Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"[AutoEditor-LLM] ❌ Ollama error: {e}")
        return None


def _sanitize_config(config, valid_transitions, valid_grades, valid_speeds):
    """Validate and clamp LLM output to valid ranges."""

    if config.get("cut_duration_mode") not in ("fixed", "variable"):
        config["cut_duration_mode"] = "variable"

    try:
        config["fixed_cut_duration"] = max(0.3, min(30.0, float(config.get("fixed_cut_duration", 2.0))))
    except (ValueError, TypeError):
        config["fixed_cut_duration"] = 2.0

    vd = config.get("variable_durations", "2.0,3.0,1.5,2.5")
    if isinstance(vd, list):
        vd = ",".join(str(x) for x in vd)
    try:
        parts = [str(max(0.3, min(30.0, float(x.strip())))) for x in str(vd).split(",") if x.strip()]
        config["variable_durations"] = ",".join(parts) if parts else "2.0,3.0,1.5"
    except Exception:
        config["variable_durations"] = "2.0,3.0,1.5"

    cp = config.get("cut_pattern", "1,2,3,4,5,6")
    if isinstance(cp, list):
        cp = ",".join(str(x) for x in cp)
    config["cut_pattern"] = str(cp)

    trans = config.get("transitions", ["cross_dissolve"])
    if isinstance(trans, str):
        trans = [trans]
    if isinstance(trans, list):
        trans = [t for t in trans if t in valid_transitions]
    config["transitions"] = trans if trans else ["cross_dissolve"]

    try:
        tf = int(config.get("transition_frames", 6))
        config["transition_frames"] = max(0, min(30, tf))
    except (ValueError, TypeError):
        config["transition_frames"] = 6

    sr = config.get("speed_ramp", "none")
    config["speed_ramp"] = sr if sr in valid_speeds else "none"

    try:
        config["speed_factor"] = max(0.25, min(4.0, float(config.get("speed_factor", 1.0))))
    except (ValueError, TypeError):
        config["speed_factor"] = 1.0

    cg = config.get("color_grade", "none")
    config["color_grade"] = cg if cg in valid_grades else "none"

    try:
        config["transition_intensity"] = max(0.0, min(1.0, float(config.get("transition_intensity", 0.85))))
    except (ValueError, TypeError):
        config["transition_intensity"] = 0.85

    # Visual effects
    vfx = config.get("visual_effects", ["bloom", "chromatic_aberration", "contrast_protect"])
    if isinstance(vfx, str):
        vfx = [vfx]
    valid_effects = ["chromatic_aberration", "bloom", "film_grain", "anamorphic_streak", "heavy_vignette", "contrast_protect"]
    vfx = [e for e in vfx if e in valid_effects]
    config["visual_effects"] = vfx if vfx else ["bloom", "chromatic_aberration", "contrast_protect"]

    # ── Fragmentation parameters ──────────────────────────────────────
    try:
        config["min_chunk_duration"] = max(0.2, min(5.0, float(config.get("min_chunk_duration", 0.8))))
    except (ValueError, TypeError):
        config["min_chunk_duration"] = 0.8

    try:
        config["max_chunk_duration"] = max(0.5, min(10.0, float(config.get("max_chunk_duration", 2.5))))
    except (ValueError, TypeError):
        config["max_chunk_duration"] = 2.5

    # Ensure max >= min
    if config["max_chunk_duration"] < config["min_chunk_duration"]:
        config["max_chunk_duration"] = config["min_chunk_duration"]

    try:
        config["shuffle_intensity"] = max(0.0, min(1.0, float(config.get("shuffle_intensity", 0.7))))
    except (ValueError, TypeError):
        config["shuffle_intensity"] = 0.7

    # ── Per-chunk effect probabilities ────────────────────────────────
    _clamp_chance(config, "reverse_chance", 0.0, 0.15, 0.0)
    _clamp_chance(config, "punch_in_chance", 0.0, 0.35, 0.18)
    _clamp_chance(config, "burst_frequency", 0.0, 0.35, 0.12)
    _clamp_chance(config, "black_breath_chance", 0.0, 0.08, 0.0)
    _clamp_chance(config, "hold_frame_chance", 0.0, 0.12, 0.0)
    _clamp_chance(config, "micro_ramp_chance", 0.0, 0.25, 0.12)

    # Edit narrative (keep as-is, just ensure it's a string)
    narrative = config.get("edit_narrative", "")
    if not isinstance(narrative, str):
        narrative = str(narrative)
    config["edit_narrative"] = narrative

    selected_mood = config.get("selected_mood", "cinematic")
    if not isinstance(selected_mood, str):
        selected_mood = str(selected_mood)
    config["selected_mood"] = selected_mood

    # ── Contrast protect flag ─────────────────────────────────────────
    cp = config.get("contrast_protect", True)
    if isinstance(cp, str):
        cp = cp.lower() in ("true", "1", "yes", "on")
    config["contrast_protect"] = bool(cp)

    # ── Fragment phases (optional LLM-specified rhythm) ───────────────
    fp = config.get("fragment_phases", None)
    if fp is not None:
        if isinstance(fp, list):
            # Validate each phase dict
            valid_phases = []
            for phase in fp:
                if isinstance(phase, dict) and all(k in phase for k in ["min_dur", "max_dur", "min_cuts", "max_cuts"]):
                    try:
                        valid_phases.append({
                            "name": str(phase.get("name", "custom")),
                            "min_dur": max(0.1, min(10.0, float(phase["min_dur"]))),
                            "max_dur": max(0.2, min(10.0, float(phase["max_dur"]))),
                            "min_cuts": max(1, min(10, int(phase["min_cuts"]))),
                            "max_cuts": max(1, min(10, int(phase["max_cuts"]))),
                        })
                    except (ValueError, TypeError):
                        pass
            config["fragment_phases"] = valid_phases if valid_phases else None
        else:
            config["fragment_phases"] = None

    return config


def _clamp_chance(config, key, lo, hi, default):
    """Clamp a probability value to [lo, hi] with a fallback default."""
    try:
        config[key] = max(lo, min(hi, float(config.get(key, default))))
    except (ValueError, TypeError):
        config[key] = default


def ask_ollama_with_descriptions(model, user_prompt, video_summaries, vision_context):
    """
    Send a prompt + Florence-2 text descriptions to Ollama (text-only, no images).

    This is the preferred method when VisionDirector has analyzed the videos.
    Works with ANY Ollama model (including small text-only models like gemma3:4b).

    Args:
        model: Ollama model name
        user_prompt: User's editing style description
        video_summaries: Text describing source videos (frame counts, durations)
        vision_context: Rich text descriptions from VisionDirector

    Returns:
        dict with edit configuration, or None on failure.
    """
    available_transitions = [
        "hard_cut", "flash_white", "flash_black", "zoom_punch_in",
        "zoom_punch_out", "glitch_cut", "whip_pan", "cross_dissolve",
        "luma_fade", "swipe_left", "swipe_up", "shake_cut",
    ]
    available_grades = [
        "none", "cinematic_warm", "cinematic_cool", "high_contrast",
        "vivid_pop", "moody_dark", "clean_bright", "teal_orange",
        "pastel_soft", "vintage_warm", "hollywood", "blockbuster",
    ]
    available_effects = [
        "chromatic_aberration", "bloom", "film_grain",
        "anamorphic_streak", "heavy_vignette", "contrast_protect",
    ]
    available_speeds = [
        "none", "slow_motion", "speed_up", "fast_start",
        "speed_up_end", "accelerate",
    ]
    available_moods = [
        "dreamy", "romantic", "calm", "nature", "elegant", "warm",
        "minimal", "playful", "retro", "cinematic", "bold", "luxury",
        "urban", "energetic", "intense", "hypnotic", "raw", "neon",
        "editorial", "chaos",
    ]

    system = f"""You are a world-class product commercial director, film editor, and advertising creative director. You create premium edits that sell the product while keeping it believable, clean, and cinematic.

SOURCE VIDEOS:
{video_summaries}

{vision_context}

PROFESSIONAL EDITING RULES:
OVERRIDING SALES-DIRECTOR RULES:
- This is ALWAYS a TikTok sales ad, but it must also feel luxury, cinematic, premium, and professional.
- Choose one of the 20 hidden moods as the base sentiment/edit language.
- The user's prompt is additive direction: refine and elevate the chosen mood, do not replace it.
- Decide how much TikTok sales energy, luxury, cinematic drama, and premium polish to apply from 25% to 100% each, based on the footage.
- Detect video-model problems: bad morphing, weird animation, ugly movement, distorted hands/product, fake-looking motion, or frames that feel like still images. Avoid or minimize those moments.
- Keep the edit punchy and alive. Do not make a slow sequence of almost-still images.

1. Use the VISUAL ANALYSIS above to choose the best mood from this list: {json.dumps(available_moods)}.
2. Use that mood as the base edit language, then adapt it using the user's prompt.
3. Lead with the STRONGEST believable product shot. Close with a memorable product angle.
4. INTERLEAVE videos creatively — contrast wide with close-up, static with motion.
5. Editing should feel intentional and mostly invisible. Avoid random effects that call attention away from the product.
6. Vary shot lengths musically, but do not make the whole edit frantic. Use short accents around motion/reveals and longer breathing cuts on product hero shots.
7. The user's prompt is ADDITIVE DIRECTION. It should refine and elevate your chosen mood, not replace the mood.
8. Build a commercial arc: hook -> product proof/detail -> emotional desire -> hero ending.
9. Each transition serves a purpose:
   - hard_cut: Clean, sharp. Use for matching motion or rhythm.
   - flash_white: Energy burst. Use sparingly at key product reveals.
   - flash_black: Dramatic weight. Use rarely; too much feels choppy.
   - zoom_punch_in: Product impact. Punch INTO the hero shot.
   - whip_pan: Fast motion blur. Creates urgency.
   - cross_dissolve: Elegant blend. Best for similar lighting.
   - luma_fade: Luxury premium reveal through brightness.
   - glitch_cut: Digital edge. Only for tech/gaming/intentional style.
   - shake_cut: Camera impact. Use rarely.
10. Choose color grades that enhance the PRODUCT and preserve real material detail.

CRITICAL CONTRAST RULES:
- NEVER use "high_contrast" — it crushed blacks and looks harsh.
- Prefer "hollywood", "cinematic_warm", or "teal_orange" for premium look.
- ALWAYS include "contrast_protect" in visual_effects.
- The final look must be SOFT, CINEMATIC, and PROFESSIONAL — never harsh.

CONTENT-DRIVEN RHYTHM:
- Map the VISUAL CONTENT to the editing rhythm:
  * Motion/action shots -> short accents (0.5-1.0s)
  * Product hero/close-up shots -> lingering desirability cuts (2.5-4.0s)
  * Scene/detail transitions -> medium cuts (1.2-2.0s)
- Every 3-5 seconds, the rhythm should evolve, not stutter.
- variable_durations should show polished variation: "1.0,2.6,0.8,1.6,3.0,0.9,2.2,1.2,3.5"
- NEVER use uniform cut lengths — that looks robotic.

11. Visual effects must be restrained: bloom, film_grain, contrast_protect are usually enough. Use chromatic aberration lightly.

ALL PARAMETERS (include EVERY key in your JSON):

0. "selected_mood": one of {json.dumps(available_moods)}
1. "cut_duration_mode": "variable"
2. "fixed_cut_duration": float 0.5-10.0
3. "variable_durations": comma-separated floats with polished variation (e.g. "1.0,2.6,0.8,1.6,3.0,0.9,2.2,1.2,3.5")
4. "cut_pattern": comma-separated video numbers — interleave CREATIVELY based on content analysis
5. "transitions": JSON array of 4-6 transitions
   Available: {json.dumps(available_transitions)}
6. "transition_frames": int 3-14
7. "speed_ramp": one of {json.dumps(available_speeds)}
8. "speed_factor": float 0.25-4.0
9. "color_grade": one of {json.dumps(available_grades)} — NEVER "high_contrast", prefer "hollywood" or "cinematic_warm"
10. "transition_intensity": float 0.7-1.0
11. "visual_effects": JSON array — ALWAYS include ["bloom", "chromatic_aberration", "contrast_protect"]
    Available: {json.dumps(available_effects)}
12. "min_chunk_duration": float 0.2-5.0
13. "max_chunk_duration": float 0.5-5.0 (must be >= min)
14. "shuffle_intensity": float 0.0-1.0 (0=sequential, 1=fully random)
15. "reverse_chance": float 0.0-0.15
16. "punch_in_chance": float 0.0-0.35
17. "burst_frequency": float 0.0-0.35
18. "black_breath_chance": float 0.0-0.08
19. "hold_frame_chance": float 0.0-0.12
20. "micro_ramp_chance": float 0.0-0.25
21. "contrast_protect": boolean true/false (recommend true)
22. "edit_narrative": 3-5 sentences explaining your creative vision. Reference SPECIFIC shots from the analysis. Explain the chosen mood, how the user's prompt modified it, the hook, the middle proof/detail section, and the hero ending.

RESPOND WITH ONLY A VALID JSON OBJECT."""

    try:
        payload_dict = {
            "model": model,
            "system": system,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "num_predict": 1500,
            },
        }

        payload = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"[AutoEditor-LLM] 🤖 Querying '{model}' with vision descriptions (text-only)...")
        print(f"[AutoEditor-LLM] Prompt: \"{user_prompt[:200]}\"")

        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            response_text = raw.get("response", "")
            print(f"[AutoEditor-LLM] Response: {response_text[:500]}")

            config = json.loads(response_text)
            config = _sanitize_config(
                config, available_transitions, available_grades, available_speeds
            )

            print(f"[AutoEditor-LLM] ✅ Config parsed (vision-directed)")
            return config

    except json.JSONDecodeError as e:
        print(f"[AutoEditor-LLM] ❌ Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"[AutoEditor-LLM] ❌ Ollama error: {e}")
        return None


def ask_ollama_lyrics_style(model, lyrics_text, bpm):
    """
    Ask Ollama to auto-pick lyrics overlay style based on song content.

    Returns dict with display_style, colors, font, position, etc.
    """
    available_styles = [
        "karaoke", "subtitles", "word_pop", "typewriter", "word_wave",
        "glow_pulse", "slide_in", "bounce_drop", "fade_flow", "neon_flash",
    ]
    available_fonts = [
        "arial", "impact", "roboto", "montserrat", "bebas_neue",
        "comic_sans", "times", "courier",
    ]
    available_positions = ["top", "upper_third", "center", "lower_third", "bottom"]
    available_backgrounds = ["none", "solid_bar", "gradient_bar", "rounded_box", "blur_box", "shadow_only"]
    available_line_modes = ["single_line", "two_lines", "three_lines", "word_by_word", "full_verse"]

    # Truncate lyrics for prompt efficiency
    lyrics_preview = lyrics_text[:800] if len(lyrics_text) > 800 else lyrics_text

    system = f"""You are a TikTok creative director specializing in viral music content. Choose the perfect text overlay style for song lyrics on a TikTok video.

SONG INFO:
- BPM: {bpm}
- Lyrics preview:
{lyrics_preview}

STYLE OPTIONS:
1. karaoke — Words highlight as sung (classic sing-along)
2. subtitles — Clean lines appear/disappear (professional)
3. word_pop — Words pop in with bounce (viral TikTok style)
4. typewriter — Characters type in synced (ASMR/aesthetic)
5. word_wave — Words bounce in wave pattern (fun/playful)
6. glow_pulse — Active words glow and pulse (neon/night)
7. slide_in — Lines slide from sides (dynamic/energetic)
8. bounce_drop — Words drop from above (upbeat/playful)
9. fade_flow — Smooth crossfade between lines (chill/ambient)
10. neon_flash — Neon glow with flash bursts (EDM/club)

Choose style based on:
- Fast BPM (>130) → word_pop, bounce_drop, neon_flash, slide_in
- Medium BPM (90-130) → karaoke, word_wave, glow_pulse, typewriter
- Slow BPM (<90) → subtitles, fade_flow, karaoke
- Emotional lyrics → fade_flow, subtitles, karaoke
- Hype/party lyrics → neon_flash, word_pop, bounce_drop
- Aesthetic/artistic → typewriter, glow_pulse

RESPOND WITH ONLY JSON containing:
1. "display_style": one of {json.dumps(available_styles)}
2. "text_color": hex color for inactive text (e.g. "#FFFFFF")
3. "highlight_color": hex color for active word (e.g. "#FFD700")
4. "font_family": one of {json.dumps(available_fonts)}
5. "text_position": one of {json.dumps(available_positions)}
6. "background_style": one of {json.dumps(available_backgrounds)}
7. "background_opacity": float 0.0-1.0
8. "line_display": one of {json.dumps(available_line_modes)}
9. "style_narrative": 2-3 sentences explaining your creative choice"""

    try:
        payload = json.dumps({
            "model": model,
            "system": system,
            "prompt": f"Choose the best lyrics overlay style for this song (BPM: {bpm}). Analyze the mood and energy.",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.7, "num_predict": 500},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"[LyricsOverlay-LLM] 🤖 Querying '{model}' for style...")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            config = json.loads(raw.get("response", "{}"))

            # Validate
            if config.get("display_style") not in available_styles:
                config["display_style"] = "word_pop"
            if config.get("font_family") not in available_fonts:
                config["font_family"] = "arial"
            if config.get("text_position") not in available_positions:
                config["text_position"] = "bottom"
            if config.get("background_style") not in available_backgrounds:
                config["background_style"] = "none"
            if config.get("line_display") not in available_line_modes:
                config["line_display"] = "single_line"

            narrative = config.get("style_narrative", "")
            print(f"[LyricsOverlay-LLM] ✅ Style: {config['display_style']}")
            if narrative:
                print(f"[LyricsOverlay-LLM] 💡 {narrative[:200]}")
            return config

    except Exception as e:
        print(f"[LyricsOverlay-LLM] ❌ Error: {e}")
        return None
