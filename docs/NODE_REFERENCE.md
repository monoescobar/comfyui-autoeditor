# Node reference

## Auto Editor

Internal types: `DJ_AutoEditor` and compatibility alias `DJ_AutoDirector`.

Category: `🎬 Escobarte/Video`.

Required inputs are the local Ollama model selector, optional creative text,
and two `IMAGE`/`VHS_VIDEOINFO` source pairs. Up to four additional source
pairs and their audio may be connected. Source-one and source-two audio are
also optional.

`target_duration_seconds` is a string intentionally: this preserves exact
values and compatibility with previously saved widget layouts. Zero or blank
uses the available-footage behavior. The target frame count is calculated from
the selected duration and source-one FPS.

Outputs, in stable order:

1. `images_output` (`IMAGE`)
2. `audio_output` (`AUDIO`)
3. `video_info_output` (`VHS_VIDEOINFO`)
4. `edit_report` (`STRING`)
5. `vision_descriptions` (`STRING`)
6. `output_frame_count` (`INT`)
7. `recommended_bpm` (`INT`)
8. `recommended_keyscale` (`STRING`)
9. `recommended_timesignature` (`INT`)
10. `recommended_music_tags` (`STRING`)

The compatibility alias must retain the identical interface. Removing it would
turn earlier saved workflows red.

## Audio Mixer

Internal type: `DJ_AudioMixer`.

Category: `🎬 Escobarte/Audio`.

Two audio inputs are required. Controls cover equal-power balance, start/end
alignment, per-track dB gain, fades on track two, edge crossfade, limiter,
normalization, DC-offset removal, and channel handling.

The output sample rate is the higher input rate. The output length is the longer
input length after resampling. The result is clamped and sanitized to eliminate
NaN and infinity values. `mix_report` records the resolved processing contract.

## Lyrics Overlay

Internal type: `DJ_LyricsOverlay`.

Category: `🎬 Escobarte/Video`.

Required inputs are song audio, supplied lyrics, a video frame batch, display
style, and Whisper model size. `video_info` is recommended. `fps_override`
takes precedence when greater than zero.

The node returns overlaid frames, the original audio, video information, and a
sync report. With no lyrics it passes frames/audio through and reports that no
lyrics were supplied. If Whisper alignment fails, the deterministic alignment
fallback is used before the node gives up.

## Stable workflow contract

Internal type names, required-input order, output order, and the Auto Director
alias are compatibility boundaries. Any breaking change requires a migration
script and a separately documented major release.
