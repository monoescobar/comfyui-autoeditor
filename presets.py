"""
Mood-based editing presets for DJ_AutoEditor.

20 Moods — all with Hollywood-grade visual effects and fragment-shuffle editing.
Each mood defines fragmentation intensity, shuffle behavior, and per-chunk
effect probabilities alongside traditional cut/transition/grade settings.
"""

MOODS = {
    # ── 1. Dreamy 🌙 ───────────────────────────────────────────────────
    "dreamy": {
        "cut_pattern": "1,2,1,3,2,4,3,5,4,6,5,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 5.0,
        "variable_durations": "5.5,3.0,6.0,2.0,5.0,7.0,3.5,5.5,2.5",
        "transitions": ["cross_dissolve", "luma_fade", "cross_dissolve", "luma_fade"],
        "transition_frames": 14,
        "transition_intensity": 0.85,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.72,
        "color_grade": "pastel_soft",
        "visual_effects": ["bloom", "chromatic_aberration", "film_grain"],
        "contrast_protect": False,
        # Fragmentation — gentle, long fragments
        "min_chunk_duration": 2.0,
        "max_chunk_duration": 5.0,
        "shuffle_intensity": 0.3,
        # Per-chunk effects — mostly off for dreamy
        "reverse_chance": 0.0,
        "punch_in_chance": 0.05,
        "burst_frequency": 0.0,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.15,
        "micro_ramp_chance": 0.05,
    },

    # ── 2. Romantic 💕 ─────────────────────────────────────────────────
    "romantic": {
        "cut_pattern": "1,2,3,1,4,2,5,3,6,4",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 4.5,
        "variable_durations": "4.5,5.5,3.5,6.0,4.0,5.0,3.5,7.0",
        "transitions": ["cross_dissolve", "luma_fade", "cross_dissolve"],
        "transition_frames": 12,
        "transition_intensity": 0.9,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.78,
        "color_grade": "cinematic_warm",
        "visual_effects": ["bloom", "film_grain"],
        # Fragmentation — tender, mostly sequential
        "min_chunk_duration": 2.0,
        "max_chunk_duration": 4.0,
        "shuffle_intensity": 0.3,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.05,
        "burst_frequency": 0.0,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.05,
    },

    # ── 3. Calm 🍃 ────────────────────────────────────────────────────
    "calm": {
        "cut_pattern": "1,2,3,4,5,6,3,1,5,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 4.5,
        "variable_durations": "4.5,5.0,3.5,5.5,4.0,6.0,4.5,5.0",
        "transitions": ["cross_dissolve", "luma_fade", "swipe_left", "cross_dissolve"],
        "transition_frames": 10,
        "transition_intensity": 0.75,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "clean_bright",
        "visual_effects": ["bloom"],
        # Fragmentation — peaceful, subtle shuffle
        "min_chunk_duration": 1.5,
        "max_chunk_duration": 4.0,
        "shuffle_intensity": 0.4,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.0,
        "burst_frequency": 0.0,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.05,
        "micro_ramp_chance": 0.0,
    },

    # ── 4. Nature 🌿 ──────────────────────────────────────────────────
    "nature": {
        "cut_pattern": "1,3,5,2,6,4,1,5,3,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 4.0,
        "variable_durations": "4.5,3.5,5.5,3.0,6.0,4.0,5.0,3.5",
        "transitions": ["cross_dissolve", "luma_fade", "swipe_up", "cross_dissolve"],
        "transition_frames": 10,
        "transition_intensity": 0.75,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "clean_bright",
        "visual_effects": ["bloom", "film_grain"],
        # Fragmentation — organic, flowing
        "min_chunk_duration": 1.5,
        "max_chunk_duration": 4.0,
        "shuffle_intensity": 0.4,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.05,
        "burst_frequency": 0.0,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.05,
        "micro_ramp_chance": 0.0,
    },

    # ── 5. Elegant ✨ ─────────────────────────────────────────────────
    "elegant": {
        "cut_pattern": "1,3,2,5,4,6,1,4,3,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 4.0,
        "variable_durations": "4.0,3.0,5.0,2.5,4.5,3.5,5.5,3.0",
        "transitions": ["luma_fade", "cross_dissolve", "zoom_punch_in", "luma_fade"],
        "transition_frames": 10,
        "transition_intensity": 0.85,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.82,
        "color_grade": "hollywood",
        "visual_effects": ["bloom", "chromatic_aberration", "film_grain"],
        # Fragmentation — refined, measured
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 3.5,
        "shuffle_intensity": 0.5,
        "reverse_chance": 0.05,
        "punch_in_chance": 0.15,
        "burst_frequency": 0.1,
        "black_breath_chance": 0.05,
        "hold_frame_chance": 0.15,
        "micro_ramp_chance": 0.1,
    },

    # ── 6. Warm 🌅 ────────────────────────────────────────────────────
    "warm": {
        "cut_pattern": "1,2,4,3,6,5,2,1,5,3",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 3.5,
        "variable_durations": "3.5,4.5,2.5,4.0,3.0,5.0,3.5,4.5",
        "transitions": ["cross_dissolve", "swipe_left", "luma_fade", "cross_dissolve"],
        "transition_frames": 8,
        "transition_intensity": 0.8,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "vintage_warm",
        "visual_effects": ["bloom", "film_grain"],
        # Fragmentation — friendly, comfortable
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 3.5,
        "shuffle_intensity": 0.5,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.1,
        "burst_frequency": 0.1,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.05,
    },

    # ── 7. Minimal ⬜ ─────────────────────────────────────────────────
    "minimal": {
        "cut_pattern": "1,2,3,4,5,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 3.5,
        "variable_durations": "3.5,4.0,3.0,3.5,4.5,3.0",
        "transitions": ["hard_cut", "cross_dissolve", "hard_cut"],
        "transition_frames": 4,
        "transition_intensity": 0.5,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "none",
        "visual_effects": [],
        # Fragmentation — clean, simple
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 3.0,
        "shuffle_intensity": 0.4,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.0,
        "burst_frequency": 0.0,
        "black_breath_chance": 0.0,
        "hold_frame_chance": 0.0,
        "micro_ramp_chance": 0.0,
    },

    # ── 8. Playful 🎨 ─────────────────────────────────────────────────
    "playful": {
        "cut_pattern": "1,4,2,5,3,6,2,4,1,6,3,5",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 2.0,
        "variable_durations": "1.5,2.5,1.0,3.0,1.8,2.0,1.2,2.8",
        "transitions": ["swipe_left", "zoom_punch_in", "swipe_up", "flash_white", "hard_cut", "whip_pan"],
        "transition_frames": 6,
        "transition_intensity": 0.9,
        "speed_ramp": "none",
        "speed_factor": 1.1,
        "color_grade": "vivid_pop",
        "visual_effects": ["bloom", "chromatic_aberration"],
        # Fragmentation — fun, bouncy, unpredictable
        "min_chunk_duration": 0.5,
        "max_chunk_duration": 2.5,
        "shuffle_intensity": 0.7,
        "reverse_chance": 0.1,
        "punch_in_chance": 0.25,
        "burst_frequency": 0.4,
        "black_breath_chance": 0.1,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.15,
    },

    # ── 9. Retro 📼 ───────────────────────────────────────────────────
    "retro": {
        "cut_pattern": "1,3,2,5,4,6,3,1,6,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 2.5,
        "variable_durations": "3.0,2.0,3.5,1.5,2.5,4.0,2.0,3.0",
        "transitions": ["hard_cut", "flash_black", "hard_cut", "cross_dissolve"],
        "transition_frames": 5,
        "transition_intensity": 0.7,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "vintage_warm",
        "visual_effects": ["film_grain", "heavy_vignette"],
        # Fragmentation — vintage, analog feel
        "min_chunk_duration": 0.8,
        "max_chunk_duration": 3.0,
        "shuffle_intensity": 0.5,
        "reverse_chance": 0.1,
        "punch_in_chance": 0.1,
        "burst_frequency": 0.2,
        "black_breath_chance": 0.15,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.05,
    },

    # ── 10. Cinematic 🎬 ──────────────────────────────────────────────
    "cinematic": {
        "cut_pattern": "1,3,5,2,6,4,1,5,6,3,2,4",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 4.0,
        "variable_durations": "0.5,4.5,0.4,3.0,0.6,5.0,0.3,2.5,4.0,0.5,3.5",
        "transitions": ["luma_fade", "flash_black", "cross_dissolve", "luma_fade", "zoom_punch_in"],
        "transition_frames": 10,
        "transition_intensity": 0.9,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.78,
        "color_grade": "hollywood",
        "visual_effects": ["bloom", "chromatic_aberration", "anamorphic_streak", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — theatrical, dramatic with extreme variation
        "min_chunk_duration": 0.3,
        "max_chunk_duration": 4.0,
        "shuffle_intensity": 0.6,
        "reverse_chance": 0.05,
        "punch_in_chance": 0.2,
        "burst_frequency": 0.2,
        "black_breath_chance": 0.1,
        "hold_frame_chance": 0.15,
        "micro_ramp_chance": 0.15,
    },

    # ── 11. Bold 🔥 ───────────────────────────────────────────────────
    "bold": {
        "cut_pattern": "1,4,2,5,3,6,1,5,3,6,2,4",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 2.5,
        "variable_durations": "0.3,2.5,0.4,1.5,0.3,3.0,0.5,1.0,0.3,3.5,0.4,2.0",
        "transitions": ["flash_white", "zoom_punch_in", "hard_cut", "whip_pan", "flash_black", "hard_cut"],
        "transition_frames": 6,
        "transition_intensity": 1.0,
        "speed_ramp": "speed_up_end",
        "speed_factor": 1.2,
        "color_grade": "blockbuster",
        "visual_effects": ["bloom", "chromatic_aberration", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — confident, striking with dramatic variation
        "min_chunk_duration": 0.2,
        "max_chunk_duration": 3.5,
        "shuffle_intensity": 0.7,
        "reverse_chance": 0.1,
        "punch_in_chance": 0.3,
        "burst_frequency": 0.4,
        "black_breath_chance": 0.1,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.2,
    },

    # ── 12. Luxury 💎 ─────────────────────────────────────────────────
    "luxury": {
        "cut_pattern": "1,2,3,1,4,5,2,6,3,5",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 5.5,
        "variable_durations": "5.5,2.0,6.5,4.0,0.8,5.0,7.0,0.5,4.5,6.0",
        "transitions": ["luma_fade", "cross_dissolve", "zoom_punch_in", "luma_fade"],
        "transition_frames": 14,
        "transition_intensity": 0.85,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.68,
        "color_grade": "hollywood",
        "visual_effects": ["bloom", "chromatic_aberration", "anamorphic_streak", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — ultra-slow, premium, minimal shuffle
        "min_chunk_duration": 1.5,
        "max_chunk_duration": 5.0,
        "shuffle_intensity": 0.3,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.1,
        "burst_frequency": 0.05,
        "black_breath_chance": 0.05,
        "hold_frame_chance": 0.2,
        "micro_ramp_chance": 0.1,
    },

    # ── 13. Urban 🏙️ ──────────────────────────────────────────────────
    "urban": {
        "cut_pattern": "1,5,3,6,2,4,6,1,3,5,4,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 1.5,
        "variable_durations": "0.3,1.8,0.4,2.0,0.3,1.5,0.5,2.2,0.3,1.2",
        "transitions": ["whip_pan", "glitch_cut", "hard_cut", "shake_cut", "flash_white"],
        "transition_frames": 4,
        "transition_intensity": 1.0,
        "speed_ramp": "speed_up",
        "speed_factor": 1.35,
        "color_grade": "teal_orange",
        "visual_effects": ["chromatic_aberration", "bloom", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — fast, gritty, street
        "min_chunk_duration": 0.2,
        "max_chunk_duration": 2.5,
        "shuffle_intensity": 0.9,
        "reverse_chance": 0.15,
        "punch_in_chance": 0.3,
        "burst_frequency": 0.6,
        "black_breath_chance": 0.15,
        "hold_frame_chance": 0.05,
        "micro_ramp_chance": 0.2,
    },

    # ── 14. Energetic ⚡ ──────────────────────────────────────────────
    "energetic": {
        "cut_pattern": "1,4,6,2,5,3,1,6,4,3,5,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 1.5,
        "variable_durations": "0.3,1.8,0.4,2.0,0.3,1.5,0.5,2.2,0.3,0.8",
        "transitions": ["flash_white", "whip_pan", "shake_cut", "zoom_punch_in", "hard_cut", "swipe_left"],
        "transition_frames": 5,
        "transition_intensity": 1.0,
        "speed_ramp": "accelerate",
        "speed_factor": 1.3,
        "color_grade": "blockbuster",
        "visual_effects": ["bloom", "chromatic_aberration", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — high-energy, dynamic
        "min_chunk_duration": 0.2,
        "max_chunk_duration": 3.0,
        "shuffle_intensity": 0.85,
        "reverse_chance": 0.1,
        "punch_in_chance": 0.3,
        "burst_frequency": 0.5,
        "black_breath_chance": 0.1,
        "hold_frame_chance": 0.05,
        "micro_ramp_chance": 0.25,
    },

    # ── 15. Intense 💥 ────────────────────────────────────────────────
    "intense": {
        "cut_pattern": "1,6,3,5,2,4,6,1,5,3,4,2,1,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 1.0,
        "variable_durations": "0.2,1.2,0.3,0.6,0.2,1.4,0.3,0.7,0.2,0.8,1.0,0.3,0.5,0.9",
        "transitions": ["glitch_cut", "flash_white", "shake_cut", "hard_cut", "whip_pan", "flash_black"],
        "transition_frames": 4,
        "transition_intensity": 1.0,
        "speed_ramp": "speed_up",
        "speed_factor": 1.4,
        "color_grade": "blockbuster",
        "visual_effects": ["chromatic_aberration", "bloom", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — aggressive, maximal
        "min_chunk_duration": 0.2,
        "max_chunk_duration": 2.0,
        "shuffle_intensity": 0.95,
        "reverse_chance": 0.2,
        "punch_in_chance": 0.4,
        "burst_frequency": 0.7,
        "black_breath_chance": 0.2,
        "hold_frame_chance": 0.1,
        "micro_ramp_chance": 0.3,
    },

    # ── 16. Hypnotic 🌀 ──────────────────────────────────────────────
    "hypnotic": {
        "cut_pattern": "1,3,2,4,1,5,3,6,2,5,4,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 2.0,
        "variable_durations": "2.0,1.5,2.5,1.2,2.0,1.8,2.5,1.5",
        "transitions": ["cross_dissolve", "luma_fade", "hard_cut", "cross_dissolve"],
        "transition_frames": 10,
        "transition_intensity": 0.8,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.8,
        "color_grade": "moody_dark",
        "visual_effects": ["bloom", "chromatic_aberration", "heavy_vignette"],
        # Fragmentation — trance-like, repetitive rhythm
        "min_chunk_duration": 1.0,
        "max_chunk_duration": 2.5,
        "shuffle_intensity": 0.6,
        "reverse_chance": 0.2,
        "punch_in_chance": 0.15,
        "burst_frequency": 0.1,
        "black_breath_chance": 0.2,
        "hold_frame_chance": 0.25,
        "micro_ramp_chance": 0.3,
    },

    # ── 17. Raw 📹 ────────────────────────────────────────────────────
    "raw": {
        "cut_pattern": "1,2,3,4,5,6,2,4,1,3,5,6",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 2.5,
        "variable_durations": "2.5,1.5,3.0,1.0,2.0,3.5,1.5,2.0",
        "transitions": ["hard_cut", "shake_cut", "hard_cut", "whip_pan"],
        "transition_frames": 4,
        "transition_intensity": 0.7,
        "speed_ramp": "none",
        "speed_factor": 1.0,
        "color_grade": "none",
        "visual_effects": ["film_grain", "heavy_vignette"],
        # Fragmentation — documentary/BTS, authentic
        "min_chunk_duration": 0.8,
        "max_chunk_duration": 3.0,
        "shuffle_intensity": 0.5,
        "reverse_chance": 0.0,
        "punch_in_chance": 0.1,
        "burst_frequency": 0.2,
        "black_breath_chance": 0.1,
        "hold_frame_chance": 0.05,
        "micro_ramp_chance": 0.05,
    },

    # ── 18. Neon 💜 ───────────────────────────────────────────────────
    "neon": {
        "cut_pattern": "1,5,3,6,2,4,6,3,1,5,4,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 1.5,
        "variable_durations": "0.3,1.2,0.4,1.8,0.3,0.5,1.5,0.3,2.0,0.4,0.7",
        "transitions": ["glitch_cut", "flash_white", "hard_cut", "flash_black", "shake_cut"],
        "transition_frames": 4,
        "transition_intensity": 1.0,
        "speed_ramp": "speed_up",
        "speed_factor": 1.2,
        "color_grade": "vivid_pop",
        "visual_effects": ["chromatic_aberration", "bloom", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — cyberpunk, neon-soaked, glitch-forward
        "min_chunk_duration": 0.2,
        "max_chunk_duration": 3.0,
        "shuffle_intensity": 0.85,
        "reverse_chance": 0.15,
        "punch_in_chance": 0.35,
        "burst_frequency": 0.5,
        "black_breath_chance": 0.2,
        "hold_frame_chance": 0.15,
        "micro_ramp_chance": 0.25,
    },

    # ── 19. Editorial 📰 ─────────────────────────────────────────────
    "editorial": {
        "cut_pattern": "1,3,5,2,6,4,1,4,6,3",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 3.0,
        "variable_durations": "0.4,3.0,0.3,1.5,4.0,0.3,1.0,2.5,0.5,3.5,0.4,1.5,3.0",
        "transitions": ["hard_cut", "luma_fade", "flash_black", "hard_cut", "zoom_punch_in"],
        "transition_frames": 8,
        "transition_intensity": 0.9,
        "speed_ramp": "slow_motion",
        "speed_factor": 0.85,
        "color_grade": "hollywood",
        "visual_effects": ["bloom", "chromatic_aberration", "film_grain", "anamorphic_streak", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — Vogue/fashion, deliberate with sharp impacts
        "min_chunk_duration": 0.3,
        "max_chunk_duration": 4.0,
        "shuffle_intensity": 0.55,
        "reverse_chance": 0.05,
        "punch_in_chance": 0.3,
        "burst_frequency": 0.25,
        "black_breath_chance": 0.15,
        "hold_frame_chance": 0.2,
        "micro_ramp_chance": 0.2,
    },

    # ── 20. Chaos 🔥 ─────────────────────────────────────────────────
    "chaos": {
        "cut_pattern": "1,6,3,5,2,4,6,1,5,3,4,2,1,6,5,2",
        "cut_duration_mode": "variable",
        "fixed_cut_duration": 0.8,
        "variable_durations": "0.2,0.8,0.3,0.4,1.2,0.2,0.3,1.0,0.2,0.5,0.6,1.0,0.2,0.3,0.7",
        "transitions": ["glitch_cut", "flash_white", "shake_cut", "hard_cut", "flash_black", "whip_pan"],
        "transition_frames": 3,
        "transition_intensity": 1.0,
        "speed_ramp": "accelerate",
        "speed_factor": 1.5,
        "color_grade": "blockbuster",
        "visual_effects": ["chromatic_aberration", "bloom", "film_grain", "contrast_protect"],
        "contrast_protect": True,
        # Fragmentation — maximum, strobe-like, experimental
        "min_chunk_duration": 0.15,
        "max_chunk_duration": 2.0,
        "shuffle_intensity": 1.0,
        "reverse_chance": 0.3,
        "punch_in_chance": 0.5,
        "burst_frequency": 0.8,
        "black_breath_chance": 0.3,
        "hold_frame_chance": 0.2,
        "micro_ramp_chance": 0.4,
    },
}

PRESETS = MOODS


def get_mood(name):
    if name in MOODS:
        return MOODS[name].copy()
    return None


def get_mood_names():
    return list(MOODS.keys())


def get_preset(name):
    return get_mood(name)

def get_preset_names():
    return get_mood_names()
