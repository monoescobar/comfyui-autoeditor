"""
Transition effects for DJ_AutoEditor.
All functions work on torch tensors with shape [B, H, W, C] in range [0, 1].

Professional-grade transitions with smooth easing curves.

Transition types:
  - Overlapping (blend tail/head):  cross_dissolve, luma_fade, swipe_left, swipe_up, whip_pan
  - Insertive (add extra frames):   flash_white, flash_black
  - Modifying (same frame count):   zoom_punch_in, zoom_punch_out, glitch_cut, shake_cut
  - None:                           hard_cut
"""

import torch
import torch.nn.functional as F


# ─── Easing curves ────────────────────────────────────────────────────────────

def _smoothstep(t):
    """Smooth ease-in-out: t*t*(3 - 2t). Input/output [0,1]."""
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_tensor(n, device, dtype):
    """Generate n smoothstep-eased values from 0 to 1."""
    t = torch.linspace(0, 1, n, device=device, dtype=dtype)
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t):
    """Ease-out curve: decelerating. t in [0,1]."""
    return 1.0 - (1.0 - t) * (1.0 - t)


# ─── Main dispatch ───────────────────────────────────────────────────────────

def join_segments(seg_a, seg_b, transition_type, n_transition_frames, intensity=1.0):
    """
    Join two video segments with a transition effect.

    Args:
        seg_a: [N_A, H, W, C] frames from first segment
        seg_b: [N_B, H, W, C] frames from second segment
        transition_type: str, one of the 12 transition types
        n_transition_frames: int, frames used for the transition
        intensity: float 0-1, strength of the effect

    Returns:
        combined: [N_out, H, W, C] combined frames with transition
    """
    if intensity <= 0 or n_transition_frames <= 0:
        transition_type = "hard_cut"

    # Effective transition frames (scaled by intensity)
    n_eff = max(1, int(n_transition_frames * intensity))

    dispatch = {
        "hard_cut": _hard_cut,
        "flash_white": _flash_white,
        "flash_black": _flash_black,
        "zoom_punch_in": _zoom_punch_in,
        "zoom_punch_out": _zoom_punch_out,
        "glitch_cut": _glitch_cut,
        "whip_pan": _whip_pan,
        "cross_dissolve": _cross_dissolve,
        "luma_fade": _luma_fade,
        "swipe_left": _swipe_left,
        "swipe_up": _swipe_up,
        "shake_cut": _shake_cut,
        "black_breath": _black_breath,
    }

    fn = dispatch.get(transition_type, _hard_cut)
    return fn(seg_a, seg_b, n_eff, intensity)


# ─── Transition implementations ──────────────────────────────────────────────

def _hard_cut(seg_a, seg_b, n_frames, intensity):
    return torch.cat([seg_a, seg_b], dim=0)


def _flash_white(seg_a, seg_b, n_frames, intensity):
    """Smooth flash to white with eased blend on both sides."""
    h, w, c = seg_a.shape[1], seg_a.shape[2], seg_a.shape[3]
    n = max(min(n_frames, 5), 3)  # At least 3 frames for smooth flash
    frames = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        # Bell curve: peaks at center, gentle edges
        flash_amount = _smoothstep(1.0 - abs(2.0 * t - 1.0)) * intensity
        if t < 0.5:
            base = seg_a[-1]
        else:
            base = seg_b[0]
        frame = base * (1.0 - flash_amount) + flash_amount
        frames.append(frame)
    return torch.cat([seg_a, torch.stack(frames), seg_b], dim=0)


def _flash_black(seg_a, seg_b, n_frames, intensity):
    """Smooth dip to black with eased transitions."""
    h, w, c = seg_a.shape[1], seg_a.shape[2], seg_a.shape[3]
    n = max(min(n_frames, 7), 3)
    frames = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        # Bell curve for black intensity
        black_amount = _smoothstep(1.0 - abs(2.0 * t - 1.0)) * intensity
        if t < 0.5:
            base = seg_a[-1]
        else:
            base = seg_b[0]
        frame = base * (1.0 - black_amount)
        frames.append(frame)
    return torch.cat([seg_a, torch.stack(frames), seg_b], dim=0)


def _cross_dissolve(seg_a, seg_b, n_frames, intensity):
    """Smooth cross-dissolve with ease-in-out blending curve."""
    n = min(n_frames, seg_a.shape[0], seg_b.shape[0])
    if n <= 0:
        return _hard_cut(seg_a, seg_b, n_frames, intensity)
    tail = seg_a[-n:]
    head = seg_b[:n]
    # Smoothstep easing instead of linear — much smoother blend
    alphas = _smoothstep_tensor(n, seg_a.device, seg_a.dtype).view(-1, 1, 1, 1)
    blended = tail * (1 - alphas) + head * alphas
    return torch.cat([seg_a[:-n], blended, seg_b[n:]], dim=0)


def _luma_fade(seg_a, seg_b, n_frames, intensity):
    """Luma-based dissolve with smooth easing — bright areas transition first."""
    n = min(n_frames, seg_a.shape[0], seg_b.shape[0])
    if n <= 0:
        return _hard_cut(seg_a, seg_b, n_frames, intensity)
    tail = seg_a[-n:]
    head = seg_b[:n]
    frames = []
    for i in range(n):
        t_raw = (i + 1) / (n + 1)
        t = _smoothstep(t_raw)  # Eased timing
        lum_a = tail[i, ..., 0:1] * 0.299 + tail[i, ..., 1:2] * 0.587 + tail[i, ..., 2:3] * 0.114
        # Smooth mask with soft edge
        threshold = 1.0 - t * intensity
        mask_raw = (lum_a - threshold) / max(0.15, 0.3 * (1.0 - abs(2.0 * t_raw - 1.0)))
        mask_a = torch.clamp(mask_raw, 0, 1)
        mask_b = 1.0 - mask_a
        frame = tail[i] * mask_b + head[i] * mask_a
        frames.append(frame)
    return torch.cat([seg_a[:-n], torch.stack(frames), seg_b[n:]], dim=0)


def _swipe_left(seg_a, seg_b, n_frames, intensity):
    """Smooth left wipe with eased motion."""
    n = min(n_frames, seg_a.shape[0], seg_b.shape[0])
    if n <= 0:
        return _hard_cut(seg_a, seg_b, n_frames, intensity)
    w = seg_a.shape[2]
    frames = []
    for i in range(n):
        t_raw = (i + 1) / (n + 1) * intensity
        t = _smoothstep(t_raw)  # Eased position
        split = int(w * (1 - t))
        frame = seg_a[-n + i].clone()
        if split < w:
            # Soft edge blend (4 pixel gradient)
            edge_width = max(1, min(8, int(w * 0.02)))
            frame[:, split:] = seg_b[i, :, :w - split]
            # Blend at the seam
            if edge_width > 0 and split > edge_width and split < w - edge_width:
                for e in range(edge_width):
                    blend = (e + 1) / (edge_width + 1)
                    col = split - edge_width + e
                    frame[:, col] = frame[:, col] * (1 - blend) + seg_b[i, :, w - split - edge_width + e] * blend
        frames.append(frame)
    return torch.cat([seg_a[:-n], torch.stack(frames), seg_b[n:]], dim=0)


def _swipe_up(seg_a, seg_b, n_frames, intensity):
    """Smooth upward wipe with eased motion."""
    n = min(n_frames, seg_a.shape[0], seg_b.shape[0])
    if n <= 0:
        return _hard_cut(seg_a, seg_b, n_frames, intensity)
    h = seg_a.shape[1]
    frames = []
    for i in range(n):
        t_raw = (i + 1) / (n + 1) * intensity
        t = _smoothstep(t_raw)
        split = int(h * (1 - t))
        frame = seg_a[-n + i].clone()
        if split < h:
            edge_width = max(1, min(8, int(h * 0.02)))
            frame[split:, :] = seg_b[i, :h - split, :]
            # Blend at the seam
            if edge_width > 0 and split > edge_width and split < h - edge_width:
                for e in range(edge_width):
                    blend = (e + 1) / (edge_width + 1)
                    row = split - edge_width + e
                    frame[row, :] = frame[row, :] * (1 - blend) + seg_b[i, h - split - edge_width + e, :] * blend
        frames.append(frame)
    return torch.cat([seg_a[:-n], torch.stack(frames), seg_b[n:]], dim=0)


def _whip_pan(seg_a, seg_b, n_frames, intensity):
    """Smooth whip pan with motion blur, eased movement."""
    n = min(n_frames, seg_a.shape[0], seg_b.shape[0])
    if n <= 0:
        return _hard_cut(seg_a, seg_b, n_frames, intensity)
    h, w, c = seg_a.shape[1], seg_a.shape[2], seg_a.shape[3]
    last_a = seg_a[-1]
    first_b = seg_b[0]
    frames = []
    for i in range(n):
        t_raw = (i + 1) / (n + 1)
        t = _smoothstep(t_raw)  # Eased movement
        split = int(w * (1 - t))
        frame = torch.zeros(h, w, c, device=seg_a.device, dtype=seg_a.dtype)
        if split > 0:
            src_start = min(int(w * t), w - 1)
            copy_w = min(split, w - src_start)
            frame[:, :copy_w] = last_a[:, src_start:src_start + copy_w]
        if split < w:
            copy_w = w - split
            frame[:, split:split + copy_w] = first_b[:, :copy_w]
        # Motion blur peaks at center of transition
        blur_strength = 1.0 - abs(2.0 * t_raw - 1.0)  # Bell curve
        blur_r = int(w * 0.06 * intensity * blur_strength)
        if blur_r > 1:
            frame = _horizontal_blur(frame, blur_r)
        frames.append(frame)
    return torch.cat([seg_a[:-1], torch.stack(frames), seg_b[1:]], dim=0)


def _zoom_punch_in(seg_a, seg_b, n_frames, intensity):
    """Zoom in settle with overshoot easing."""
    n = min(n_frames, seg_b.shape[0])
    start_scale = 1.0 + 0.12 * intensity  # Slightly less aggressive
    modified_head = _apply_zoom_settle(seg_b[:n], start_scale=start_scale, end_scale=1.0)
    return torch.cat([seg_a, modified_head, seg_b[n:]], dim=0)


def _zoom_punch_out(seg_a, seg_b, n_frames, intensity):
    """Zoom out settle with smooth easing."""
    n = min(n_frames, seg_b.shape[0])
    start_scale = 1.0 - 0.08 * intensity
    modified_head = _apply_zoom_settle(seg_b[:n], start_scale=start_scale, end_scale=1.0)
    return torch.cat([seg_a, modified_head, seg_b[n:]], dim=0)


def _glitch_cut(seg_a, seg_b, n_frames, intensity):
    """Digital glitch transition with decaying effect."""
    n = min(n_frames, 5)
    h, w, c = seg_a.shape[1], seg_a.shape[2], seg_a.shape[3]
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Ease out: effect decays
        decay = _ease_out(1.0 - t)
        effective_intensity = intensity * decay

        base = seg_a[-1] if t < 0.4 else seg_b[0]
        frame = base.clone()
        # RGB channel offset (decaying)
        offset = int(w * 0.03 * effective_intensity)
        if offset > 0 and c >= 3:
            frame[..., 0] = torch.roll(base[..., 0], offset, dims=1)
            frame[..., 2] = torch.roll(base[..., 2], -offset, dims=1)
        # Horizontal slice displacement (fewer, more controlled)
        n_slices = int(2 + 4 * effective_intensity)
        for _ in range(n_slices):
            y_start = torch.randint(0, max(h - 8, 1), (1,)).item()
            slice_h = torch.randint(2, 6, (1,)).item()
            y_end = min(y_start + slice_h, h)
            shift = torch.randint(-int(w * 0.05 * effective_intensity) - 1,
                                   int(w * 0.05 * effective_intensity) + 1, (1,)).item()
            if shift != 0:
                frame[y_start:y_end] = torch.roll(frame[y_start:y_end], shift, dims=1)
        # Subtle noise (reduced)
        noise = torch.randn_like(frame) * 0.04 * effective_intensity
        frame = torch.clamp(frame + noise, 0, 1)
        frames.append(frame)
    return torch.cat([seg_a, torch.stack(frames), seg_b], dim=0)


def _shake_cut(seg_a, seg_b, n_frames, intensity):
    """Camera shake with exponential decay — settles quickly."""
    n = min(n_frames, seg_b.shape[0], 6)
    h, w = seg_b.shape[1], seg_b.shape[2]
    max_shift = max(1, int(min(h, w) * 0.025 * intensity))
    modified = seg_b[:n].clone()
    for i in range(n):
        # Exponential decay — shake settles fast
        decay = (0.3 ** i)  # Faster decay than linear
        sx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item() * decay)
        sy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item() * decay)
        if sx != 0 or sy != 0:
            modified[i] = torch.roll(modified[i], shifts=(sy, sx), dims=(0, 1))
    return torch.cat([seg_a, modified, seg_b[n:]], dim=0)


def _black_breath(seg_a, seg_b, n_frames, intensity):
    """Smooth dip-to-black breathing gap between segments.
    Fades last frames of seg_a to black, holds black, fades first frames of seg_b from black.
    """
    n_hold = max(1, min(int(n_frames * intensity * 0.5), 4))  # Black hold frames
    n_fade = max(1, min(3, seg_a.shape[0] // 2, seg_b.shape[0] // 2))  # Fade frames
    h, w, c = seg_a.shape[1], seg_a.shape[2], seg_a.shape[3]

    # Fade-out: last n_fade frames of seg_a dim to black
    fade_out_part = seg_a[-n_fade:].clone()
    for i in range(n_fade):
        t = (i + 1) / (n_fade + 1)
        fade_out_part[i] = fade_out_part[i] * (1.0 - _smoothstep(t) * intensity)

    # Pure black hold
    black = torch.zeros(n_hold, h, w, c, device=seg_a.device, dtype=seg_a.dtype)

    # Fade-in: first n_fade frames of seg_b rise from black
    fade_in_part = seg_b[:n_fade].clone()
    for i in range(n_fade):
        t = (i + 1) / (n_fade + 1)
        fade_in_part[i] = fade_in_part[i] * _smoothstep(t) * intensity + fade_in_part[i] * (1.0 - intensity)

    return torch.cat([seg_a[:-n_fade], fade_out_part, black, fade_in_part, seg_b[n_fade:]], dim=0)


# ─── Helper functions ─────────────────────────────────────────────────────────

def _apply_zoom_settle(frames, start_scale, end_scale):
    """Apply a zoom animation with smooth ease-out settling."""
    n = frames.shape[0]
    h, w = frames.shape[1], frames.shape[2]
    result = frames.clone()
    for i in range(n):
        t_raw = i / max(n - 1, 1)
        t = _smoothstep(t_raw)  # Smooth settling
        scale = start_scale + (end_scale - start_scale) * t
        if abs(scale - 1.0) < 0.001:
            continue
        crop_h = max(1, int(h / scale))
        crop_w = max(1, int(w / scale))
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        cropped = frames[i:i + 1, y0:y0 + crop_h, x0:x0 + crop_w, :]
        cropped_p = cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(cropped_p, size=(h, w), mode='bilinear', align_corners=False)
        result[i] = resized.permute(0, 2, 3, 1)[0]
    return result


def _horizontal_blur(frame, radius):
    """Apply horizontal box blur to a single frame [H, W, C]."""
    if radius <= 0:
        return frame
    h, w, c = frame.shape
    frame_p = frame.unsqueeze(0).permute(0, 3, 1, 2)
    kernel = torch.ones(1, 1, 1, radius * 2 + 1, device=frame.device, dtype=frame.dtype) / (radius * 2 + 1)
    padded = F.pad(frame_p, (radius, radius, 0, 0), mode='reflect')
    blurred = F.conv2d(
        padded.reshape(c, 1, h, w + radius * 2),
        kernel,
        padding=0,
    ).reshape(1, c, h, w)
    return torch.clamp(blurred.permute(0, 2, 3, 1)[0], 0, 1)
