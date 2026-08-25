# Troubleshooting

## Nodes are red or missing

Restart the ComfyUI backend after installation, then hard-refresh the browser.
Verify all four types appear in `/object_info`: `DJ_AutoEditor`,
`DJ_AutoDirector`, `DJ_AudioMixer`, and `DJ_LyricsOverlay`.

## Ollama says offline

The pack expects the local Ollama API. Start Ollama and confirm a model is
installed, or keep the offline entry and use deterministic planning.

## Video understanding cannot load

Use `video_understanding=OFF` to isolate editing from the optional model stack.
Then verify `transformers`, `huggingface_hub`, and the expected Florence model
using ComfyUI's embedded Python rather than the system Python.

## Lyrics are early or late

Connect the matching `VHS_VIDEOINFO`. If the source metadata is unreliable,
set `fps_override` to the exact saved-video FPS and use `timing_offset_ms` only
for the remaining constant offset.

## A font cannot be downloaded

Select a Windows system font such as Arial, or place a properly licensed font
under `fonts/<family>.ttf`. Cached binaries remain local and are ignored by Git.

## Memory exhaustion

Reduce source duration or resolution. Auto Editor receives complete IMAGE
batches; six long high-resolution sources can require substantial RAM and
VRAM even when the final edit is short.
