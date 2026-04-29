"""
DJ_AudioMixer — Professional Two-Track Audio Mixer for ComfyUI.

Agency-grade audio mixing node that combines two audio tracks
with precise dB-based volume control, alignment options,
and broadcast-quality post-processing (limiter, normalization,
fade-in/out, DC offset removal).

All operations work on torch tensors matching ComfyUI's
{"waveform": Tensor, "sample_rate": int} audio format.
"""

import torch
import math


class DJ_AudioMixer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "mix_balance": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 1,
                    "tooltip": (
                        "Mix balance between tracks. 50 = equal mix. "
                        "90 = 90% Track 1 + 10% Track 2. "
                        "0 = Track 2 only. 100 = Track 1 only."
                    ),
                }),
                "alignment": (["start", "end"], {
                    "default": "start",
                    "tooltip": (
                        "How to align the shorter track. "
                        "'start' = both begin together. "
                        "'end' = both end together (shorter track is padded at the start)."
                    ),
                }),
            },
            "optional": {
                "audio1_volume_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Track 1 volume adjustment in dB. 0 = unchanged. -6 = half amplitude. +6 = double.",
                }),
                "audio2_volume_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Track 2 volume adjustment in dB. 0 = unchanged. -6 = half amplitude. +6 = double.",
                }),
                "fade_in_ms": ("INT", {
                    "default": 0, "min": 0, "max": 5000, "step": 10,
                    "tooltip": "Fade-in duration on Track 2 in milliseconds. Prevents clicks at the start.",
                }),
                "fade_out_ms": ("INT", {
                    "default": 0, "min": 0, "max": 5000, "step": 10,
                    "tooltip": "Fade-out duration on Track 2 in milliseconds. Smooth tail-off.",
                }),
                "crossfade_ms": ("INT", {
                    "default": 0, "min": 0, "max": 2000, "step": 10,
                    "tooltip": (
                        "Crossfade zone at the edges of the shorter track. "
                        "Creates a smooth blend where Track 2 enters and exits. "
                        "Applied in addition to fade_in/fade_out."
                    ),
                }),
                "limiter": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": (
                        "Broadcast-quality brick-wall limiter. Prevents clipping by "
                        "soft-compressing peaks above -1 dBFS. Always recommended."
                    ),
                }),
                "limiter_ceiling_db": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Peak ceiling for the limiter in dBFS. -1.0 is broadcast standard.",
                }),
                "normalize": (["off", "peak", "rms"], {
                    "default": "off",
                    "tooltip": (
                        "Output normalization. 'peak' normalizes to peak ceiling. "
                        "'rms' normalizes to target RMS loudness (-14 LUFS approx). "
                        "'off' = no normalization."
                    ),
                }),
                "dc_offset_removal": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Remove DC offset from the mixed output. Prevents speaker damage and improves headroom.",
                }),
                "stereo_mode": (["stereo", "mono_sum", "keep_original"], {
                    "default": "keep_original",
                    "tooltip": (
                        "'keep_original' preserves channel count. "
                        "'stereo' forces stereo output. "
                        "'mono_sum' sums to mono."
                    ),
                }),
            },
        }

    CATEGORY = "audio/video processing"
    FUNCTION = "mix_audio"
    RETURN_NAMES = ("audio_output", "mix_report")
    RETURN_TYPES = ("AUDIO", "STRING")

    # ─── dB conversion ────────────────────────────────────────────────

    @staticmethod
    def _db_to_linear(db):
        """Convert decibels to linear amplitude multiplier."""
        return math.pow(10.0, db / 20.0)

    @staticmethod
    def _linear_to_db(linear):
        """Convert linear amplitude to decibels."""
        if linear <= 0:
            return -120.0
        return 20.0 * math.log10(linear)

    # ─── Audio extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract_audio(audio_input):
        """Extract waveform tensor and sample rate from ComfyUI audio dict."""
        if audio_input is None:
            return None, None
        if isinstance(audio_input, dict) and "waveform" in audio_input:
            waveform = audio_input["waveform"]
            # Remove batch dim if present: [B, C, S] → [C, S]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            return waveform, audio_input["sample_rate"]
        return None, None

    # ─── Resampling ───────────────────────────────────────────────────

    @staticmethod
    def _resample(audio, src_rate, dst_rate):
        """Resample audio from src_rate to dst_rate using linear interpolation."""
        if src_rate == dst_rate:
            return audio
        ratio = dst_rate / src_rate
        n_in = audio.shape[-1]
        n_out = max(1, int(n_in * ratio))
        indices = torch.linspace(0, n_in - 1, n_out, device=audio.device)
        # Linear interpolation
        idx_floor = indices.long().clamp(0, n_in - 1)
        idx_ceil = (idx_floor + 1).clamp(0, n_in - 1)
        frac = indices - idx_floor.float()
        resampled = audio[..., idx_floor] * (1.0 - frac) + audio[..., idx_ceil] * frac
        print(f"[AudioMixer] 🔄 Resampled {src_rate}Hz → {dst_rate}Hz ({n_in} → {n_out} samples)")
        return resampled

    # ─── Channel matching ─────────────────────────────────────────────

    @staticmethod
    def _match_channels(audio, target_channels):
        """Match channel count to target."""
        current = audio.shape[0]
        if current == target_channels:
            return audio
        if target_channels == 1:
            # Sum to mono with proper gain compensation
            return audio.mean(dim=0, keepdim=True)
        if target_channels == 2 and current == 1:
            # Mono to stereo: duplicate
            return audio.repeat(2, 1)
        # General case: truncate or pad channels
        if current > target_channels:
            return audio[:target_channels]
        # Pad with zeros
        pad = torch.zeros(target_channels - current, audio.shape[-1], device=audio.device)
        return torch.cat([audio, pad], dim=0)

    # ─── Fade curves ──────────────────────────────────────────────────

    @staticmethod
    def _apply_fade(audio, fade_in_samples, fade_out_samples):
        """Apply smooth fade-in and fade-out using raised-cosine curves."""
        n_samples = audio.shape[-1]

        if fade_in_samples > 0:
            fade_in_samples = min(fade_in_samples, n_samples)
            # Raised cosine: smoother than linear, no clicks
            t = torch.linspace(0, math.pi / 2, fade_in_samples, device=audio.device)
            fade_in_curve = torch.sin(t) ** 2  # Equal-power fade
            audio = audio.clone()
            audio[..., :fade_in_samples] *= fade_in_curve

        if fade_out_samples > 0:
            fade_out_samples = min(fade_out_samples, n_samples)
            t = torch.linspace(math.pi / 2, 0, fade_out_samples, device=audio.device)
            fade_out_curve = torch.sin(t) ** 2
            audio = audio.clone()
            audio[..., -fade_out_samples:] *= fade_out_curve

        return audio

    # ─── Brick-wall limiter ───────────────────────────────────────────

    @staticmethod
    def _soft_clip(audio, ceiling_linear, knee_db=3.0):
        """
        Broadcast-quality soft-knee limiter.
        Uses a smooth tanh-based soft clipping curve that preserves
        transient character while preventing clipping.
        """
        # Soft knee: compress gradually as signal approaches ceiling
        knee_linear = DJ_AudioMixer._db_to_linear(-knee_db)
        threshold = ceiling_linear * knee_linear

        result = audio.clone()
        abs_audio = audio.abs()

        # Below threshold: pass through
        # Above threshold: apply soft compression
        above_mask = abs_audio > threshold

        if above_mask.any():
            # Normalize above-threshold portion
            overshoot = (abs_audio[above_mask] - threshold) / (ceiling_linear - threshold + 1e-10)
            # Tanh compression: smooth, musical limiting
            compressed = threshold + (ceiling_linear - threshold) * torch.tanh(overshoot)
            result[above_mask] = compressed * audio[above_mask].sign()

        return result

    # ─── DC offset removal (high-pass at ~10Hz) ──────────────────────

    @staticmethod
    def _remove_dc_offset(audio):
        """Remove DC offset using a simple running mean subtraction."""
        # Per-channel mean subtraction (simple but effective)
        mean = audio.mean(dim=-1, keepdim=True)
        return audio - mean

    # ─── RMS measurement ──────────────────────────────────────────────

    @staticmethod
    def _rms(audio):
        """Calculate RMS level of audio."""
        return torch.sqrt(torch.mean(audio ** 2) + 1e-10)

    @staticmethod
    def _peak(audio):
        """Get peak absolute level."""
        return audio.abs().max()

    # ─── Main mix function ────────────────────────────────────────────

    def mix_audio(self, audio1, audio2, mix_balance, alignment, **kwargs):
        # Extract options
        vol1_db = kwargs.get("audio1_volume_db", 0.0)
        vol2_db = kwargs.get("audio2_volume_db", 0.0)
        fade_in_ms = kwargs.get("fade_in_ms", 0)
        fade_out_ms = kwargs.get("fade_out_ms", 0)
        crossfade_ms = kwargs.get("crossfade_ms", 0)
        enable_limiter = kwargs.get("limiter", "enable") == "enable"
        limiter_ceil_db = kwargs.get("limiter_ceiling_db", -1.0)
        normalize_mode = kwargs.get("normalize", "off")
        enable_dc_removal = kwargs.get("dc_offset_removal", "enable") == "enable"
        stereo_mode = kwargs.get("stereo_mode", "keep_original")

        print(f"\n{'='*60}")
        print(f"[AudioMixer] 🎚️ Professional Audio Mixer")
        print(f"{'='*60}")

        # ── Extract waveforms ─────────────────────────────────────────
        wave1, sr1 = self._extract_audio(audio1)
        wave2, sr2 = self._extract_audio(audio2)

        if wave1 is None or wave2 is None:
            raise ValueError("[AudioMixer] Both audio inputs must be valid!")

        print(f"[AudioMixer] Track 1: {wave1.shape[0]}ch, {wave1.shape[-1]} samples @ {sr1}Hz "
              f"({wave1.shape[-1]/sr1:.2f}s)")
        print(f"[AudioMixer] Track 2: {wave2.shape[0]}ch, {wave2.shape[-1]} samples @ {sr2}Hz "
              f"({wave2.shape[-1]/sr2:.2f}s)")

        # ── Pre-mix analysis ──────────────────────────────────────────
        pre_peak1 = self._peak(wave1)
        pre_peak2 = self._peak(wave2)
        pre_rms1 = self._rms(wave1)
        pre_rms2 = self._rms(wave2)
        print(f"[AudioMixer] Track 1 levels: peak={self._linear_to_db(pre_peak1.item()):.1f}dB, "
              f"RMS={self._linear_to_db(pre_rms1.item()):.1f}dB")
        print(f"[AudioMixer] Track 2 levels: peak={self._linear_to_db(pre_peak2.item()):.1f}dB, "
              f"RMS={self._linear_to_db(pre_rms2.item()):.1f}dB")

        # ── Match sample rates ────────────────────────────────────────
        # Use the higher sample rate as target for quality preservation
        target_sr = max(sr1, sr2)
        if sr1 != target_sr:
            wave1 = self._resample(wave1, sr1, target_sr)
        if sr2 != target_sr:
            wave2 = self._resample(wave2, sr2, target_sr)

        # ── Determine target channels ────────────────────────────────
        if stereo_mode == "stereo":
            target_channels = 2
        elif stereo_mode == "mono_sum":
            target_channels = 1
        else:
            target_channels = max(wave1.shape[0], wave2.shape[0])

        wave1 = self._match_channels(wave1, target_channels)
        wave2 = self._match_channels(wave2, target_channels)

        # ── Apply volume adjustments (dB → linear) ───────────────────
        gain1 = self._db_to_linear(vol1_db)
        gain2 = self._db_to_linear(vol2_db)
        wave1 = wave1 * gain1
        wave2 = wave2 * gain2
        print(f"[AudioMixer] Volume: Track1 {vol1_db:+.1f}dB (×{gain1:.3f}), "
              f"Track2 {vol2_db:+.1f}dB (×{gain2:.3f})")

        # ── Apply mix balance ─────────────────────────────────────────
        # Convert percentage to equal-power crossfade coefficients
        # This sounds more natural than linear mixing
        balance = mix_balance / 100.0  # 0.0 = all track2, 1.0 = all track1
        # Equal-power panning law: preserves perceived loudness
        coeff1 = math.cos((1.0 - balance) * math.pi / 2)
        coeff2 = math.cos(balance * math.pi / 2)
        print(f"[AudioMixer] Balance: {mix_balance}% "
              f"(Track1 ×{coeff1:.3f} / Track2 ×{coeff2:.3f}, equal-power)")

        wave1 = wave1 * coeff1
        wave2 = wave2 * coeff2

        # ── Apply fade-in/out on Track 2 ──────────────────────────────
        fade_in_samples = int(fade_in_ms / 1000.0 * target_sr)
        fade_out_samples = int(fade_out_ms / 1000.0 * target_sr)
        if fade_in_samples > 0 or fade_out_samples > 0:
            wave2 = self._apply_fade(wave2, fade_in_samples, fade_out_samples)
            print(f"[AudioMixer] Fades on Track 2: in={fade_in_ms}ms, out={fade_out_ms}ms "
                  f"(raised-cosine)")

        # ── Align and pad to same length ──────────────────────────────
        len1 = wave1.shape[-1]
        len2 = wave2.shape[-1]
        max_len = max(len1, len2)

        if len1 < max_len:
            pad_amount = max_len - len1
            if alignment == "end":
                # Pad at the start
                wave1 = torch.cat([
                    torch.zeros(target_channels, pad_amount, device=wave1.device),
                    wave1
                ], dim=-1)
            else:
                # Pad at the end
                wave1 = torch.cat([
                    wave1,
                    torch.zeros(target_channels, pad_amount, device=wave1.device)
                ], dim=-1)
            print(f"[AudioMixer] Padded Track 1: {len1} → {max_len} samples "
                  f"(aligned to {alignment})")

        if len2 < max_len:
            pad_amount = max_len - len2
            if alignment == "end":
                wave2 = torch.cat([
                    torch.zeros(target_channels, pad_amount, device=wave2.device),
                    wave2
                ], dim=-1)
            else:
                wave2 = torch.cat([
                    wave2,
                    torch.zeros(target_channels, pad_amount, device=wave2.device)
                ], dim=-1)
            print(f"[AudioMixer] Padded Track 2: {len2} → {max_len} samples "
                  f"(aligned to {alignment})")

        # ── Apply crossfade at Track 2 edges ──────────────────────────
        if crossfade_ms > 0:
            cf_samples = min(int(crossfade_ms / 1000.0 * target_sr), max_len // 4)
            if cf_samples > 0:
                wave2 = self._apply_fade(wave2, cf_samples, cf_samples)
                print(f"[AudioMixer] Crossfade: {crossfade_ms}ms ({cf_samples} samples) "
                      f"at Track 2 edges")

        # ── Mix ───────────────────────────────────────────────────────
        mixed = wave1 + wave2

        # ── DC offset removal ─────────────────────────────────────────
        if enable_dc_removal:
            dc_before = mixed.mean(dim=-1).abs().max().item()
            mixed = self._remove_dc_offset(mixed)
            dc_after = mixed.mean(dim=-1).abs().max().item()
            if dc_before > 0.001:
                print(f"[AudioMixer] 🔧 DC offset removed: {dc_before:.6f} → {dc_after:.6f}")

        # ── Normalization ─────────────────────────────────────────────
        if normalize_mode == "peak":
            ceiling_linear = self._db_to_linear(limiter_ceil_db)
            current_peak = self._peak(mixed)
            if current_peak > 1e-6:
                normalize_gain = ceiling_linear / current_peak
                mixed = mixed * normalize_gain
                print(f"[AudioMixer] 📊 Peak normalized: "
                      f"×{normalize_gain:.3f} → {limiter_ceil_db:.1f}dBFS")
        elif normalize_mode == "rms":
            # Target: approximately -14 LUFS (broadcast standard)
            target_rms = self._db_to_linear(-14.0)
            current_rms = self._rms(mixed)
            if current_rms > 1e-6:
                normalize_gain = target_rms / current_rms
                # Safety cap to prevent extreme amplification
                normalize_gain = min(normalize_gain, 20.0)
                mixed = mixed * normalize_gain
                print(f"[AudioMixer] 📊 RMS normalized: "
                      f"×{normalize_gain:.3f} → target RMS {self._linear_to_db(target_rms):.1f}dB")

        # ── Brick-wall limiter ────────────────────────────────────────
        if enable_limiter:
            ceiling_linear = self._db_to_linear(limiter_ceil_db)
            pre_limit_peak = self._peak(mixed)
            mixed = self._soft_clip(mixed, ceiling_linear)
            post_limit_peak = self._peak(mixed)

            if pre_limit_peak > ceiling_linear:
                gr = self._linear_to_db(pre_limit_peak.item()) - limiter_ceil_db
                print(f"[AudioMixer] 🔊 Limiter engaged: {gr:.1f}dB gain reduction "
                      f"(ceiling {limiter_ceil_db:.1f}dBFS)")
            else:
                print(f"[AudioMixer] 🔊 Limiter active (no reduction needed)")

        # ── Final safety clamp ────────────────────────────────────────
        mixed = torch.clamp(mixed, -1.0, 1.0)

        # ── Final sanitize ────────────────────────────────────────────
        mixed = torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Post-mix analysis ─────────────────────────────────────────
        post_peak = self._peak(mixed)
        post_rms = self._rms(mixed)
        final_duration = mixed.shape[-1] / target_sr

        print(f"\n[AudioMixer] ✅ Mix complete:")
        print(f"[AudioMixer]   Duration: {final_duration:.2f}s")
        print(f"[AudioMixer]   Channels: {mixed.shape[0]}")
        print(f"[AudioMixer]   Sample rate: {target_sr}Hz")
        print(f"[AudioMixer]   Peak: {self._linear_to_db(post_peak.item()):.1f}dBFS")
        print(f"[AudioMixer]   RMS: {self._linear_to_db(post_rms.item()):.1f}dB")
        print(f"{'='*60}\n")

        # ── Build output ──────────────────────────────────────────────
        audio_output = {
            "waveform": mixed.unsqueeze(0),  # Add batch dim [1, C, S]
            "sample_rate": target_sr,
        }

        # ── Build report ──────────────────────────────────────────────
        report = self._build_report(
            sr1, sr2, target_sr,
            len1, len2, max_len,
            vol1_db, vol2_db,
            mix_balance, coeff1, coeff2,
            alignment, fade_in_ms, fade_out_ms, crossfade_ms,
            enable_limiter, limiter_ceil_db,
            normalize_mode,
            enable_dc_removal,
            pre_peak1.item(), pre_peak2.item(),
            pre_rms1.item(), pre_rms2.item(),
            post_peak.item(), post_rms.item(),
            final_duration, target_channels,
        )

        return (audio_output, report)

    # ─── Report builder ──────────────────────────────────────────────

    def _build_report(self, sr1, sr2, target_sr,
                      len1, len2, max_len,
                      vol1_db, vol2_db,
                      mix_balance, coeff1, coeff2,
                      alignment, fade_in_ms, fade_out_ms, crossfade_ms,
                      enable_limiter, limiter_ceil_db,
                      normalize_mode, enable_dc_removal,
                      pre_peak1, pre_peak2,
                      pre_rms1, pre_rms2,
                      post_peak, post_rms,
                      final_duration, channels):
        r = []
        r.append("═" * 50)
        r.append("🎚️ AUDIO MIXER — MIX REPORT")
        r.append("═" * 50)
        r.append("")

        # Input analysis
        r.append("📥 INPUT TRACKS")
        r.append("─" * 40)
        r.append(f"  Track 1: {len1/target_sr:.2f}s @ {sr1}Hz")
        r.append(f"    Peak: {self._linear_to_db(pre_peak1):.1f}dBFS  "
                 f"RMS: {self._linear_to_db(pre_rms1):.1f}dB")
        r.append(f"    Volume: {vol1_db:+.1f}dB")
        r.append(f"  Track 2: {len2/target_sr:.2f}s @ {sr2}Hz")
        r.append(f"    Peak: {self._linear_to_db(pre_peak2):.1f}dBFS  "
                 f"RMS: {self._linear_to_db(pre_rms2):.1f}dB")
        r.append(f"    Volume: {vol2_db:+.1f}dB")
        r.append("")

        # Mix settings
        r.append("🎛️ MIX SETTINGS")
        r.append("─" * 40)
        r.append(f"  Balance: {mix_balance}% (Track1 ×{coeff1:.3f} / Track2 ×{coeff2:.3f})")
        r.append(f"  Alignment: {alignment}")
        if fade_in_ms > 0 or fade_out_ms > 0:
            r.append(f"  Fades: in={fade_in_ms}ms, out={fade_out_ms}ms (raised-cosine)")
        if crossfade_ms > 0:
            r.append(f"  Crossfade: {crossfade_ms}ms at Track 2 edges")
        r.append("")

        # Processing
        r.append("⚙️ PROCESSING")
        r.append("─" * 40)
        if sr1 != sr2:
            r.append(f"  Resampled: {min(sr1,sr2)}Hz → {target_sr}Hz")
        r.append(f"  DC offset removal: {'ON' if enable_dc_removal else 'OFF'}")
        r.append(f"  Normalization: {normalize_mode}")
        r.append(f"  Limiter: {'ON' if enable_limiter else 'OFF'}"
                 + (f" (ceiling {limiter_ceil_db:.1f}dBFS)" if enable_limiter else ""))
        r.append("")

        # Output
        r.append("📤 OUTPUT")
        r.append("─" * 40)
        r.append(f"  Duration: {final_duration:.2f}s")
        r.append(f"  Channels: {channels}")
        r.append(f"  Sample rate: {target_sr}Hz")
        r.append(f"  Peak: {self._linear_to_db(post_peak):.1f}dBFS")
        r.append(f"  RMS: {self._linear_to_db(post_rms):.1f}dB")

        # Headroom
        headroom = -self._linear_to_db(post_peak)
        if headroom < 1.0:
            r.append(f"  ⚠️ Headroom: {headroom:.1f}dB (very hot!)")
        elif headroom < 3.0:
            r.append(f"  Headroom: {headroom:.1f}dB (warm)")
        else:
            r.append(f"  ✅ Headroom: {headroom:.1f}dB (healthy)")
        r.append("")
        r.append("═" * 50)

        return "\n".join(r)


# ─── Node Registration ───────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DJ_AudioMixer": DJ_AudioMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJ_AudioMixer": "Audio Mixer 🎚️",
}
