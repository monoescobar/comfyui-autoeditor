# ComfyUI AutoEditor

Three production-oriented ComfyUI nodes for assembling video frame batches,
mixing audio, and rendering synchronized lyrics. The package works on ComfyUI
`IMAGE`, `AUDIO`, and Video Helper Suite `VHS_VIDEOINFO` values; it never edits
the source video or audio payloads in place.

Current package version: `2026.8.25.1`.

## Registered nodes

| Node | Internal type | Purpose |
| --- | --- | --- |
| Auto Editor | `DJ_AutoEditor` | Build an editorial sequence from two to six source frame batches and return an audit report plus music direction. |
| Auto Editor compatibility alias | `DJ_AutoDirector` | Preserves workflows saved under the earlier internal name. It uses the same implementation as `DJ_AutoEditor`. |
| Audio Mixer | `DJ_AudioMixer` | Mix two audio payloads with level, alignment, fade, channel, normalization, and limiter controls. |
| Lyrics Overlay | `DJ_LyricsOverlay` | Align supplied lyrics to song audio and render animated text over a video frame batch. |

All user-facing inputs and outputs now include ComfyUI tooltips. Each node also
has a node-level description visible in ComfyUI's help and node information.

## Installation

Clone the repository into the active ComfyUI `custom_nodes` folder and restart
ComfyUI:

```powershell
git clone https://github.com/monoescobar/comfyui-autoeditor.git
```

Video workflows normally also use
[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
to load frames/audio and save the returned frame batch with synchronized audio.

PyTorch, NumPy, and Pillow are expected to come from ComfyUI's own Python
environment. Do not replace ComfyUI's PyTorch installation just for this pack.

## Optional capabilities

The three nodes degrade independently:

- Auto Editor's LLM planning uses a local Ollama server when available. If it
  is offline, deterministic editorial rules remain available.
- Florence-style visual understanding needs compatible `transformers` and
  `huggingface_hub` packages and model files. Set `video_understanding=OFF` to
  edit without that analysis.
- Lyrics Overlay uses `openai-whisper` for audio-to-lyric alignment when it is
  installed. A deterministic fallback is used if Whisper is unavailable.
- Roboto, Montserrat, and Bebas Neue may be downloaded from their upstream
  projects on first use and cached under `fonts/`. System fonts are preferred.

Optional packages are listed in `requirements-optional.txt` for reference.
Install only the features you intend to use and use ComfyUI's embedded Python.

## Quick workflow patterns

### Editorial assembly

```text
VHS Load Video 1.frames + video_info ─┐
VHS Load Video 2.frames + video_info ─┼─> Auto Editor
optional sources 3–6                 ─┘

Auto Editor.images_output + audio_output + video_info_output
    -> VHS Video Combine
```

`edit_report` records the chosen structure, effects, duration, and source use.
The remaining text/number outputs can drive a downstream music-generation
workflow or be attached as production metadata.

### Audio mix

```text
dialogue AUDIO ─┐
music AUDIO    ─┴─> Audio Mixer -> VHS Video Combine.audio
```

The mixer resamples to the higher input rate, matches channels, pads the shorter
track according to `alignment`, and returns a report with peak, RMS, and
headroom measurements.

### Lyrics

```text
song AUDIO + lyrics STRING + video IMAGE batch
    -> Lyrics Overlay
    -> VHS Video Combine
```

Connect `video_info` or provide `fps_override` when exact timing matters.

## Safety and operational notes

- Inputs are treated as immutable; returned tensors are newly constructed or
  passed through explicitly.
- Entire frame batches can consume substantial RAM/VRAM. Test short clips and
  the intended resolution before long production runs.
- Auto Editor performs editing, not generative identity or anatomy repair.
- Lyrics Overlay draws pixels over frames; preserve an unmodified source path
  elsewhere in the workflow if you need a clean master.
- Ollama and font downloads are network capabilities. They are optional and
  documented rather than silently required.
- Do not commit model weights, generated media, private lyrics, prompts, or
  machine-specific paths to this repository.

The complete interfaces are documented in
[docs/NODE_REFERENCE.md](docs/NODE_REFERENCE.md). Architecture, fallback
behavior, and troubleshooting are documented under `docs/`.

## Development

Run the static and lightweight behavioral tests:

```powershell
python -m unittest discover -s tests -v
```

The audio behavior tests run when PyTorch is available and skip on a minimal
documentation-only Python environment. GitHub Actions always runs compilation,
metadata, registration, documentation, and compatibility-contract checks.
