"""
Color grading & visual effects for DJ_AutoEditor.
All operations work on torch tensors with shape [B, H, W, C] in range [0, 1].

Hollywood-grade post-processing pipeline:
  Color grades → Chromatic aberration → Bloom → Film grain → Vignette
"""

import torch
import torch.nn.functional as F


# ─── Core adjustment functions ────────────────────────────────────────────────

def _brightness(frames, offset):
    return torch.clamp(frames + offset, 0, 1)


def _contrast(frames, factor):
    return torch.clamp((frames - 0.5) * factor + 0.5, 0, 1)


def _saturation(frames, factor):
    gray = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    return torch.clamp(gray + (frames - gray) * factor, 0, 1)


def _tint(frames, r, g, b):
    result = frames.clone()
    result[..., 0] *= r
    result[..., 1] *= g
    result[..., 2] *= b
    return torch.clamp(result, 0, 1)


def _vignette(frames, strength):
    if strength <= 0:
        return frames
    b, h, w, c = frames.shape
    y = torch.linspace(-1, 1, h, device=frames.device)
    x = torch.linspace(-1, 1, w, device=frames.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx * xx + yy * yy) / 1.414
    falloff = torch.clamp(r * 2.0 - 0.5, 0, 1)
    mask = (1.0 - strength * falloff).unsqueeze(0).unsqueeze(-1)
    return torch.clamp(frames * mask, 0, 1)


def _lift_blacks(frames, floor_value):
    """Lift the darkest values to create a matte/faded look."""
    return torch.clamp(frames * (1.0 - floor_value) + floor_value, 0, 1)


def _limit_contrast(frames, black_floor=0.03, white_ceil=0.97):
    """
    Soft contrast limiter — prevents crushed blacks and blown highlights.
    Uses a sigmoid-style soft roll-off instead of hard clamps for natural falloff.
    Applied AFTER color grading as a safety net.
    """
    # Soft roll-off zone widths
    soft_zone = 0.08

    # Protect shadows: smoothly lift anything below black_floor + soft_zone
    shadow_threshold = black_floor + soft_zone
    shadow_mask = torch.clamp((shadow_threshold - frames) / soft_zone, 0, 1)
    # Sigmoid-like smooth lift in shadow zone
    shadow_lift = shadow_mask * shadow_mask * (3.0 - 2.0 * shadow_mask)  # smoothstep
    frames = frames + shadow_lift * black_floor

    # Protect highlights: smoothly pull down anything above white_ceil - soft_zone
    highlight_threshold = white_ceil - soft_zone
    highlight_mask = torch.clamp((frames - highlight_threshold) / soft_zone, 0, 1)
    highlight_pull = highlight_mask * highlight_mask * (3.0 - 2.0 * highlight_mask)
    frames = frames - highlight_pull * (1.0 - white_ceil)

    return torch.clamp(frames, 0, 1)


def _adaptive_contrast(frames, target_range=0.85):
    """
    Adaptive per-frame contrast compression.
    Analyzes luminance range per frame and gently compresses if it exceeds
    the target range. Prevents jarring contrast jumps between cuts.
    """
    # Compute per-frame luminance
    lum = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114

    # Per-frame min/max (across H, W dimensions)
    B = frames.shape[0]
    lum_flat = lum.reshape(B, -1)  # [B, H*W]
    lum_min = lum_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
    lum_max = lum_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    lum_range = (lum_max - lum_min).clamp(min=0.01)

    # Only compress frames where range exceeds target
    needs_compression = (lum_range > target_range).float()
    compression_factor = target_range / lum_range
    compression_factor = compression_factor.clamp(min=0.5, max=1.0)

    # Gentle blend: compress toward midpoint
    midpoint = (lum_min + lum_max) / 2.0
    compressed = midpoint + (frames - midpoint) * compression_factor
    # Only apply to frames that need it, with soft blend
    blend_strength = needs_compression * 0.7  # Don't fully compress, keep some punch
    result = frames * (1.0 - blend_strength) + compressed * blend_strength

    return torch.clamp(result, 0, 1)


# ─── Hollywood Visual Effects ────────────────────────────────────────────────

def _chromatic_aberration(frames, intensity=0.003):
    """
    Lens dispersion / chromatic aberration.
    Shifts R channel right and B channel left, creating subtle color fringing
    that mimics real anamorphic lens imperfections.
    """
    offset = max(1, int(frames.shape[2] * intensity))
    result = frames.clone()
    # Red shifts right, Blue shifts left
    result[..., 0] = torch.roll(frames[..., 0], offset, dims=2)
    result[..., 2] = torch.roll(frames[..., 2], -offset, dims=2)
    return torch.clamp(result, 0, 1)


def _bloom(frames, threshold=0.78, intensity=0.25):
    """
    Highlight bloom / glow. Bright areas bleed light outward,
    creating that cinematic halation look from anamorphic lenses.
    """
    luminance = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    # Soft highlight mask with smooth falloff
    bright_mask = torch.clamp((luminance - threshold) / (1.0 - threshold + 0.01), 0, 1)
    # Bright-area glow (approximate blur via downscale/upscale)
    bloom_contribution = frames * bright_mask * intensity
    return torch.clamp(frames + bloom_contribution, 0, 1)


def _film_grain(frames, intensity=0.025):
    """
    Photographic film grain. Adds organic, textured noise
    that makes digital footage feel analog and cinematic.
    """
    noise = torch.randn_like(frames) * intensity
    # Luminance-weighted: more grain in midtones, less in shadows/highlights
    luminance = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    midtone_weight = 4.0 * luminance * (1.0 - luminance)  # Peaks at 0.5
    return torch.clamp(frames + noise * midtone_weight, 0, 1)


def _anamorphic_streak(frames, intensity=0.08):
    """
    Horizontal light streaks on bright areas, simulating
    anamorphic lens flare characteristics.
    """
    luminance = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    bright = torch.clamp((luminance - 0.85) / 0.15, 0, 1)
    # Horizontal streak via 1D box blur
    B, H, W, C = frames.shape
    streak = bright.permute(0, 3, 1, 2)  # [B, 1, H, W]
    k_size = max(3, int(W * 0.08)) | 1  # Ensure odd
    pad = k_size // 2
    kernel = torch.ones(1, 1, 1, k_size, device=frames.device, dtype=frames.dtype) / k_size
    streak = F.pad(streak, (pad, pad, 0, 0), mode='reflect')
    streak = F.conv2d(streak.reshape(B, 1, H, W + 2 * pad), kernel, padding=0)
    streak = streak.reshape(B, 1, H, W).permute(0, 2, 3, 1)
    # Blue-white tint for the streak
    result = frames.clone()
    result[..., 0:1] += streak * intensity * 0.7
    result[..., 1:2] += streak * intensity * 0.85
    result[..., 2:3] += streak * intensity * 1.0
    return torch.clamp(result, 0, 1)


# ─── Color grade definitions ─────────────────────────────────────────────────

def _cinematic_warm(frames):
    frames = _brightness(frames, 0.03)
    frames = _contrast(frames, 1.2)
    frames = _saturation(frames, 1.15)
    frames = _tint(frames, 1.06, 1.0, 0.90)
    frames = _bloom(frames, 0.82, 0.15)
    frames = _vignette(frames, 0.3)
    return frames


def _cinematic_cool(frames):
    frames = _contrast(frames, 1.25)
    frames = _saturation(frames, 0.85)
    frames = _tint(frames, 0.90, 0.96, 1.10)
    frames = _bloom(frames, 0.80, 0.12)
    frames = _vignette(frames, 0.25)
    return frames


def _high_contrast(frames):
    frames = _contrast(frames, 1.35)  # Reduced from 1.5 for safer output
    frames = _saturation(frames, 1.15)
    frames = _bloom(frames, 0.85, 0.10)
    frames = _vignette(frames, 0.2)
    frames = _limit_contrast(frames)  # Safety net
    return frames


def _vivid_pop(frames):
    frames = _brightness(frames, 0.05)
    frames = _contrast(frames, 1.25)
    frames = _saturation(frames, 1.5)
    frames = _tint(frames, 1.02, 1.0, 1.02)
    frames = _bloom(frames, 0.80, 0.12)
    frames = _vignette(frames, 0.12)
    return frames


def _moody_dark(frames):
    frames = _brightness(frames, -0.06)
    frames = _contrast(frames, 0.85)
    frames = _saturation(frames, 0.80)
    frames = _tint(frames, 0.93, 0.93, 1.02)
    frames = _lift_blacks(frames, 0.10)
    frames = _bloom(frames, 0.75, 0.20)
    frames = _vignette(frames, 0.45)
    return frames


def _clean_bright(frames):
    frames = _brightness(frames, 0.08)
    frames = _contrast(frames, 0.95)
    frames = _saturation(frames, 1.08)
    frames = _tint(frames, 1.0, 1.02, 1.05)
    return frames


def _teal_orange(frames):
    frames = _contrast(frames, 1.25)
    frames = _saturation(frames, 1.2)
    lum = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    shadow_mask = (1.0 - lum).clamp(0, 1)
    highlight_mask = lum.clamp(0, 1)
    result = frames.clone()
    result[..., 0:1] -= 0.10 * shadow_mask
    result[..., 2:3] += 0.10 * shadow_mask
    result[..., 0:1] += 0.10 * highlight_mask
    result[..., 2:3] -= 0.06 * highlight_mask
    result = torch.clamp(result, 0, 1)
    result = _bloom(result, 0.80, 0.15)
    result = _vignette(result, 0.22)
    return result


def _pastel_soft(frames):
    frames = _brightness(frames, 0.06)
    frames = _contrast(frames, 0.85)
    frames = _saturation(frames, 0.75)
    frames = _tint(frames, 1.04, 0.98, 1.02)
    frames = _lift_blacks(frames, 0.12)
    frames = _bloom(frames, 0.70, 0.20)
    return frames


def _vintage_warm(frames):
    frames = _brightness(frames, 0.04)
    frames = _contrast(frames, 1.1)
    frames = _saturation(frames, 0.90)
    frames = _tint(frames, 1.10, 1.02, 0.85)
    frames = _lift_blacks(frames, 0.08)
    frames = _film_grain(frames, 0.015)
    frames = _vignette(frames, 0.22)
    return frames


def _hollywood(frames):
    """
    Full Hollywood blockbuster look.
    Teal-orange split tone + lifted blacks + highlight bloom +
    chromatic aberration + film grain + anamorphic streak + strong vignette.
    Contrast-safe: no crushed blacks, no blown highlights.
    """
    # Base color — reduced from 1.35 for safer contrast
    frames = _contrast(frames, 1.25)
    frames = _saturation(frames, 1.15)
    # Teal-orange split tone (slightly reduced intensity)
    lum = frames[..., 0:1] * 0.299 + frames[..., 1:2] * 0.587 + frames[..., 2:3] * 0.114
    shadow_mask = (1.0 - lum).clamp(0, 1)
    highlight_mask = lum.clamp(0, 1)
    result = frames.clone()
    result[..., 0:1] -= 0.06 * shadow_mask
    result[..., 1:2] += 0.02 * shadow_mask
    result[..., 2:3] += 0.08 * shadow_mask
    result[..., 0:1] += 0.08 * highlight_mask
    result[..., 1:2] += 0.02 * highlight_mask
    result[..., 2:3] -= 0.05 * highlight_mask
    result = torch.clamp(result, 0, 1)
    # Lift blacks for cinematic matte — slightly more for safety
    result = _lift_blacks(result, 0.05)
    # Hollywood effects stack
    result = _bloom(result, 0.78, 0.20)
    result = _chromatic_aberration(result, 0.002)
    result = _anamorphic_streak(result, 0.06)
    result = _film_grain(result, 0.018)
    result = _vignette(result, 0.32)
    # Contrast safety net
    result = _limit_contrast(result)
    return result


def _blockbuster(frames):
    """
    Michael Bay / action movie grade.
    Punchy contrast with sharp color, but contrast-safe.
    """
    frames = _contrast(frames, 1.3)  # Reduced from 1.5
    frames = _saturation(frames, 1.25)
    frames = _tint(frames, 1.04, 0.98, 0.96)
    frames = _lift_blacks(frames, 0.03)  # Prevent full black crush
    frames = _bloom(frames, 0.80, 0.18)
    frames = _chromatic_aberration(frames, 0.003)
    frames = _film_grain(frames, 0.012)
    frames = _vignette(frames, 0.25)
    # Contrast safety net
    frames = _limit_contrast(frames)
    return frames


# ─── Post-processing pipeline ────────────────────────────────────────────────

VISUAL_EFFECTS = {
    "chromatic_aberration": lambda f: _chromatic_aberration(f, 0.003),
    "bloom": lambda f: _bloom(f, 0.78, 0.20),
    "film_grain": lambda f: _film_grain(f, 0.020),
    "anamorphic_streak": lambda f: _anamorphic_streak(f, 0.06),
    "heavy_vignette": lambda f: _vignette(f, 0.35),
    "contrast_protect": lambda f: _adaptive_contrast(_limit_contrast(f), target_range=0.85),
}


def apply_visual_effects(frames, effects_list):
    """Apply a list of visual effects to frames."""
    if not effects_list:
        return frames
    for effect_name in effects_list:
        fn = VISUAL_EFFECTS.get(effect_name)
        if fn:
            frames = fn(frames)
            print(f"[AutoEditor] ✨ Applied: {effect_name}")
    return frames


# ─── Public API ───────────────────────────────────────────────────────────────

GRADE_FUNCTIONS = {
    "none": None,
    "cinematic_warm": _cinematic_warm,
    "cinematic_cool": _cinematic_cool,
    "high_contrast": _high_contrast,
    "vivid_pop": _vivid_pop,
    "moody_dark": _moody_dark,
    "clean_bright": _clean_bright,
    "teal_orange": _teal_orange,
    "pastel_soft": _pastel_soft,
    "vintage_warm": _vintage_warm,
    "hollywood": _hollywood,
    "blockbuster": _blockbuster,
}


def apply_color_grade(frames, grade_name):
    """Apply a color grade to frames [B, H, W, C]. Returns graded frames."""
    fn = GRADE_FUNCTIONS.get(grade_name)
    if fn is None:
        return frames
    return fn(frames)


def get_grade_names():
    return list(GRADE_FUNCTIONS.keys())
