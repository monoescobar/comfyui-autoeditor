# Architecture and side effects

`__init__.py` registers only the three implementation classes and four stable
ComfyUI type names. Helper modules are not registered as nodes.

- `auto_editor.py` owns source normalization, editorial planning, output-frame
  assembly, reporting, and music-direction outputs.
- `presets.py`, `transitions.py`, and `color_grading.py` provide deterministic
  plan and image operations.
- `vision_analysis.py` encapsulates optional Florence-style analysis and model
  cleanup.
- `ollama_bridge.py` contains localhost HTTP requests and structured fallback
  handling for optional LLM planning.
- `audio_mixer.py` is an independent tensor audio processor.
- `lyrics_overlay.py`, `lyrics_sync.py`, and `text_renderer.py` implement lyric
  alignment and raster rendering.

Network behavior is limited to optional local Ollama requests, optional model
downloads through the selected vision stack, and first-use downloads for the
three listed open-font families. The edit and mixer paths do not write source
media. Downloaded fonts are caches and are excluded from Git.

Large IMAGE batches are the principal resource risk. Processing should be
validated at short duration before production resolution and duration are
combined.
