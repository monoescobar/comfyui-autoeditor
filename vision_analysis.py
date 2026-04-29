"""
Vision Director for DJ_AutoEditor.

Uses Florence-2 to:
  1. Analyze video content — generate rich text descriptions of each video
  2. Detect AI distortions — find and flag frames with artifacts

Florence-2 model auto-downloads from HuggingFace on first use (~1.5GB).
Model is loaded, used, and fully unloaded to free VRAM.
"""

import os
import sys
import torch
import numpy as np
import traceback
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

FLORENCE_MODEL_NAME = "Florence-2-large"
FLORENCE_REPO_ID = "microsoft/Florence-2-large"

# Frame position labels for the LLM context (indexed by count)
KEYFRAME_LABELS_5 = ["Opening", "Early-mid", "Midpoint", "Late-mid", "Closing"]
KEYFRAME_LABELS_3 = ["Opening", "Midpoint", "Closing"]
KEYFRAME_LABELS_2 = ["Opening", "Closing"]

# ─── Vision Quality Presets ───────────────────────────────────────────────────
# Each preset controls: keyframes per video, caption task, beam search, max tokens

VISION_PRESETS = {
    "detailed": {
        "keyframe_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "keyframe_labels": KEYFRAME_LABELS_5,
        "caption_task": "<DETAILED_CAPTION>",
        "num_beams": 3,
        "max_tokens": 256,
        "description": "5 keyframes, detailed captions, beam search (slowest, best quality)",
    },
    "balanced": {
        "keyframe_positions": [0.0, 0.5, 1.0],
        "keyframe_labels": KEYFRAME_LABELS_3,
        "caption_task": "<DETAILED_CAPTION>",
        "num_beams": 1,
        "max_tokens": 128,
        "description": "3 keyframes, detailed captions, greedy decoding",
    },
    "fast": {
        "keyframe_positions": [0.0, 0.5, 1.0],
        "keyframe_labels": KEYFRAME_LABELS_3,
        "caption_task": "<CAPTION>",
        "num_beams": 1,
        "max_tokens": 128,
        "description": "3 keyframes, simple captions, greedy decoding (recommended)",
    },
    "turbo": {
        "keyframe_positions": [0.0, 1.0],
        "keyframe_labels": KEYFRAME_LABELS_2,
        "caption_task": "<CAPTION>",
        "num_beams": 1,
        "max_tokens": 64,
        "description": "2 keyframes only, simple captions, minimal tokens (fastest)",
    },
}

def get_vision_quality_names():
    """Return list of vision quality preset names for the ComfyUI dropdown."""
    return list(VISION_PRESETS.keys())


# ─── Florence-2 Model Management ─────────────────────────────────────────────

def _get_model_directory():
    """Get the ComfyUI LLM model directory (same as comfyui-florence2)."""
    try:
        import folder_paths
        model_dir = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    except ImportError:
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "LLM")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _ensure_model_downloaded(model_dir):
    """Download Florence-2-large if not already present."""
    model_path = os.path.join(model_dir, FLORENCE_MODEL_NAME)
    if os.path.exists(model_path) and any(
        f.endswith(('.safetensors', '.bin'))
        for f in os.listdir(model_path)
    ):
        print(f"[VisionDirector] ✅ Florence-2 found at {model_path}")
        return model_path

    print(f"[VisionDirector] 📥 Downloading {FLORENCE_REPO_ID} (~1.5GB)...")
    print(f"[VisionDirector]    Destination: {model_path}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=FLORENCE_REPO_ID,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"[VisionDirector] ✅ Download complete!")
        return model_path
    except Exception as e:
        print(f"[VisionDirector] ❌ Download failed: {e}")
        traceback.print_exc()
        return None


def _load_florence2(model_path):
    """Load Florence-2 model and processor.

    Strategy:
      1. Try loading via the comfyui-florence2 node (its DownloadAndLoadFlorence2Model)
      2. Fallback: direct HuggingFace AutoModel loading

    Returns (model, processor, dtype, device, offload_device) or (None, ...) on failure.
    """
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")

    dtype = torch.float16
    print(f"[VisionDirector] 🔄 Loading Florence-2 from {model_path}...")
    print(f"[VisionDirector]    Device: {device}, dtype: {dtype}")

    # ── Strategy 1: Use comfyui-florence2's node directly ─────────────────
    try:
        # Find the comfyui-florence2 package in ComfyUI's custom_nodes
        florence2_dir = None
        try:
            import folder_paths
            custom_nodes = os.path.join(os.path.dirname(folder_paths.models_dir), "custom_nodes")
            candidate = os.path.join(custom_nodes, "comfyui-florence2")
            if os.path.exists(os.path.join(candidate, "nodes.py")):
                florence2_dir = candidate
        except ImportError:
            pass

        if florence2_dir:
            # Import the nodes module from comfyui-florence2
            # It's already loaded by ComfyUI, so we can find it in sys.modules
            fl2_nodes = None

            # Check if already imported by ComfyUI
            for mod_name, mod in sys.modules.items():
                if hasattr(mod, 'Florence2ForConditionalGeneration') and 'florence' in mod_name.lower():
                    fl2_nodes = mod
                    break
                if hasattr(mod, 'DownloadAndLoadFlorence2Model') and 'florence' in mod_name.lower():
                    fl2_nodes = mod
                    break

            if fl2_nodes and hasattr(fl2_nodes, 'load_model'):
                print(f"[VisionDirector]    Using comfyui-florence2 load_model function...")
                # Try eager first, then sdpa (sdpa may fail on newer transformers)
                for attn_impl in ["eager", "sdpa"]:
                    try:
                        model, processor = fl2_nodes.load_model(
                            model_path, attn_impl, dtype, offload_device
                        )
                        print(f"[VisionDirector] ✅ Florence-2 loaded via comfyui-florence2 (attn={attn_impl})")
                        return model, processor, dtype, device, offload_device
                    except (AttributeError, RuntimeError) as attn_err:
                        print(f"[VisionDirector] ⚠️ comfyui-florence2 attn='{attn_impl}' failed: {attn_err}")
                        continue

            print(f"[VisionDirector]    comfyui-florence2 module not in sys.modules, trying fallback...")
    except Exception as e:
        print(f"[VisionDirector] ⚠️ comfyui-florence2 strategy failed: {e}")

    # ── Strategy 2: Direct HuggingFace loading (most reliable) ────────────
    try:
        print(f"[VisionDirector] 🔄 Loading via HuggingFace transformers...")
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Try eager attention first (compatible with all transformers versions)
        # Florence-2's custom code lacks _supports_sdpa for transformers >=4.57
        for attn_impl in ["eager", "sdpa"]:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                ).eval()

                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )

                print(f"[VisionDirector] ✅ Florence-2 loaded via HuggingFace transformers (attn={attn_impl})")
                return model, processor, dtype, device, offload_device
            except (AttributeError, RuntimeError) as attn_err:
                print(f"[VisionDirector] ⚠️ attn_implementation='{attn_impl}' failed: {attn_err}")
                continue

        raise RuntimeError("All attention implementations failed")

    except Exception as e:
        print(f"[VisionDirector] ⚠️ HuggingFace AutoModel failed: {e}")
        traceback.print_exc()

    # ── Strategy 3: Manual loading with comfyui-florence2's files ──────────
    try:
        florence2_dir = None
        try:
            import folder_paths
            custom_nodes = os.path.join(os.path.dirname(folder_paths.models_dir), "custom_nodes")
            florence2_dir = os.path.join(custom_nodes, "comfyui-florence2")
        except ImportError:
            pass

        if florence2_dir and os.path.exists(florence2_dir):
            print(f"[VisionDirector] 🔄 Manual loading from comfyui-florence2 files...")
            from comfy.utils import load_torch_file

            # Add florence2_dir to path temporarily for relative imports
            added_to_path = False
            if florence2_dir not in sys.path:
                sys.path.insert(0, florence2_dir)
                added_to_path = True

            try:
                # Import the Florence2 modules
                from configuration_florence2 import Florence2Config
                from modeling_florence2 import Florence2ForConditionalGeneration
                from processing_florence2 import Florence2Processor
                from transformers import CLIPImageProcessor, BartTokenizerFast

                config = Florence2Config.from_pretrained(model_path)
                config._attn_implementation = "eager"

                model = Florence2ForConditionalGeneration.from_pretrained(
                    model_path, config=config, dtype=dtype
                ).eval().to(offload_device)

                image_processor = CLIPImageProcessor(
                    do_resize=True, size={"height": 768, "width": 768}, resample=3,
                    do_center_crop=False, do_rescale=True, rescale_factor=1/255.0,
                    do_normalize=True, image_mean=[0.485, 0.456, 0.406],
                    image_std=[0.229, 0.224, 0.225],
                )
                image_processor.image_seq_length = 577

                tokenizer = BartTokenizerFast.from_pretrained(model_path)
                processor = Florence2Processor(
                    image_processor=image_processor, tokenizer=tokenizer
                )

                print(f"[VisionDirector] ✅ Florence-2 loaded via manual import")
                return model, processor, dtype, device, offload_device

            finally:
                if added_to_path:
                    sys.path.remove(florence2_dir)

    except Exception as e:
        print(f"[VisionDirector] ❌ Manual loading failed: {e}")
        traceback.print_exc()

    print(f"[VisionDirector] ❌ ALL loading strategies failed. Florence-2 is unavailable.")
    return None, None, None, None, None


def _unload_florence2(model, offload_device):
    """Fully unload Florence-2 from VRAM."""
    if model is not None:
        try:
            model.to(offload_device)
            del model
        except Exception:
            pass
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except ImportError:
            pass
    except Exception:
        pass
    print(f"[VisionDirector] 🗑️ Florence-2 unloaded, VRAM freed")


# ─── Captioning ───────────────────────────────────────────────────────────────

def _caption_frame(model, processor, frame_tensor, dtype, device,
                   task="<DETAILED_CAPTION>", num_beams=3, max_tokens=256):
    """Generate a text caption for a single frame tensor [H,W,C] in [0,1]."""
    from PIL import Image

    # Convert tensor to PIL
    np_arr = (frame_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(np_arr)

    # Run Florence-2
    model.to(device)
    processed = processor(text=task, images=pil_img, return_tensors="pt", do_rescale=False)

    # CRITICAL: input_ids must stay Long, only pixel_values gets cast to model dtype
    input_ids = processed["input_ids"].to(device)  # Long tensor — DO NOT cast to float
    pixel_values = processed["pixel_values"].to(dtype).to(device)  # Float tensor → model dtype

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=num_beams,
            use_cache=False,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Clean up special tokens
    result = result.replace("</s>", "").replace("<s>", "")
    for token in ["<DETAILED_CAPTION>", "<CAPTION>", "<MORE_DETAILED_CAPTION>"]:
        result = result.replace(token, "")
    return result.strip()


# ─── AI Distortion Detection ─────────────────────────────────────────────────

def detect_distortions(frames, fps, sensitivity=2.5):
    """
    Detect AI-generated distortion frames using frame-to-frame consistency.

    Computes MSE between consecutive frames. Frames where the difference
    spikes dramatically compared to neighbors are flagged as distorted.

    Args:
        frames: [N, H, W, C] tensor
        fps: frames per second
        sensitivity: lower = more aggressive flagging (default 2.5)

    Returns:
        bad_frame_indices: list of int — indices of distorted frames
        quality_scores: list of float — per-frame quality score (lower = worse)
    """
    n = frames.shape[0]
    if n < 5:
        return [], [1.0] * n

    # Compute frame-to-frame MSE (downsampled for speed)
    scale = 4
    h, w = frames.shape[1] // scale, frames.shape[2] // scale
    small = torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    diffs = []
    for i in range(1, n):
        mse = ((small[i] - small[i-1]) ** 2).mean().item()
        diffs.append(mse)

    if not diffs:
        return [], [1.0] * n

    # Compute rolling median of frame differences (window = ~0.5s)
    window = max(3, int(fps * 0.5))
    diffs_arr = np.array(diffs)

    padded = np.pad(diffs_arr, (window//2, window//2), mode='edge')
    rolling_median = np.array([
        np.median(padded[i:i+window]) for i in range(len(diffs_arr))
    ])

    # Also compute channel-wise mean shifts (for color glitches)
    channel_shifts = []
    for i in range(1, n):
        shift = (small[i].mean(dim=(0, 1)) - small[i-1].mean(dim=(0, 1))).abs().max().item()
        channel_shifts.append(shift)
    channel_arr = np.array(channel_shifts)
    channel_median = np.median(channel_arr) if len(channel_arr) > 0 else 0

    # Flag frames where difference >> local median
    bad_indices = set()
    quality_scores = [1.0] * n

    for i in range(len(diffs_arr)):
        local_med = max(rolling_median[i], 0.0001)
        spike_ratio = diffs_arr[i] / local_med
        color_spike = channel_arr[i] / max(channel_median, 0.0001) if channel_median > 0.0001 else 1.0

        frame_quality = 1.0 / max(spike_ratio, 1.0)
        if color_spike > sensitivity:
            frame_quality *= 0.5

        quality_scores[i + 1] = frame_quality

        if spike_ratio > sensitivity or color_spike > sensitivity * 1.5:
            bad_indices.add(i + 1)

    # Also flag isolated frames
    for i in range(1, n - 1):
        if i in bad_indices:
            continue
        if i < len(diffs_arr) and i - 1 >= 0:
            diff_prev = diffs_arr[i - 1] if i - 1 < len(diffs_arr) else 0
            diff_next = diffs_arr[i] if i < len(diffs_arr) else 0
            local_med = max(rolling_median[min(i, len(rolling_median)-1)], 0.0001)
            if diff_prev > local_med * sensitivity and diff_next > local_med * sensitivity:
                bad_indices.add(i)
                quality_scores[i] = min(quality_scores[i], 0.3)

    return sorted(bad_indices), quality_scores


def remove_distorted_frames(frames, bad_indices, mode="freeze"):
    """
    Remove or replace distorted frames.

    Args:
        frames: [N, H, W, C] tensor
        bad_indices: list of frame indices to fix
        mode: "skip" = remove frames entirely, "freeze" = replace with previous clean frame

    Returns:
        cleaned_frames: [N', H, W, C] tensor
    """
    if not bad_indices:
        return frames

    n = frames.shape[0]
    bad_set = set(bad_indices)

    if mode == "skip":
        clean_indices = [i for i in range(n) if i not in bad_set]
        if not clean_indices:
            return frames
        return frames[clean_indices]

    elif mode == "freeze":
        result = frames.clone()
        last_clean = 0
        for i in range(n):
            if i in bad_set:
                result[i] = frames[last_clean]
            else:
                last_clean = i
        return result

    return frames


# ─── Main Analysis Pipeline ──────────────────────────────────────────────────

def analyze_videos(all_images, fps, quality="fast"):
    """
    Full vision analysis pipeline:
    1. Load Florence-2
    2. Sample keyframes from each video
    3. Caption each keyframe
    4. Unload Florence-2
    5. Return structured descriptions

    Args:
        all_images: dict {video_idx: [N, H, W, C] tensor}
        fps: frames per second
        quality: preset name — "detailed", "balanced", "fast" (default), "turbo"

    Returns:
        descriptions: dict {video_idx: [(label, caption), ...]}
        or None if Florence-2 is not available
    """
    preset = VISION_PRESETS.get(quality, VISION_PRESETS["fast"])
    positions = preset["keyframe_positions"]
    labels = preset["keyframe_labels"]
    caption_task = preset["caption_task"]
    num_beams = preset["num_beams"]
    max_tokens = preset["max_tokens"]
    n_keyframes = len(positions)
    total_captions = n_keyframes * len(all_images)

    print(f"[VisionDirector] 🚀 Starting video content analysis...")
    print(f"[VisionDirector]    Quality: {quality} — {preset['description']}")
    print(f"[VisionDirector]    Videos: {len(all_images)}, Keyframes/video: {n_keyframes}, Total captions: {total_captions}")
    print(f"[VisionDirector]    FPS: {fps}")

    # Check and download model
    model_dir = _get_model_directory()
    print(f"[VisionDirector]    Model dir: {model_dir}")

    model_path = _ensure_model_downloaded(model_dir)
    if model_path is None:
        print("[VisionDirector] ❌ Florence-2 model unavailable — cannot analyze videos")
        return None

    # Load model
    model, processor, dtype, device, offload_device = _load_florence2(model_path)
    if model is None:
        print("[VisionDirector] ❌ Florence-2 failed to load — cannot analyze videos")
        return None

    descriptions = {}
    caption_count = 0

    try:
        for vid_idx in sorted(all_images.keys()):
            frames = all_images[vid_idx]
            n_frames = frames.shape[0]
            vid_descriptions = []

            # Sample keyframes at preset positions
            for pos_idx, pos in enumerate(positions):
                frame_idx = min(int(pos * (n_frames - 1)), n_frames - 1)
                label = labels[pos_idx]

                caption_count += 1
                print(f"[VisionDirector] 🔍 [{caption_count}/{total_captions}] Video {vid_idx}, {label} (frame {frame_idx}/{n_frames})...")
                try:
                    caption = _caption_frame(
                        model, processor, frames[frame_idx], dtype, device,
                        task=caption_task, num_beams=num_beams, max_tokens=max_tokens
                    )
                    vid_descriptions.append((label, caption))
                    print(f"[VisionDirector]    → {caption[:120]}...")
                except Exception as e:
                    print(f"[VisionDirector]    ❌ Caption failed: {e}")
                    traceback.print_exc()
                    vid_descriptions.append((label, f"(caption failed: {e})"))

            descriptions[vid_idx] = vid_descriptions
            print(f"[VisionDirector] ✅ Video {vid_idx}: {len(vid_descriptions)} keyframes analyzed")

    except Exception as e:
        print(f"[VisionDirector] ❌ Analysis pipeline error: {e}")
        traceback.print_exc()
    finally:
        # Always unload
        _unload_florence2(model, offload_device)

    if not descriptions:
        print("[VisionDirector] ❌ No descriptions generated")
        return None

    total = sum(len(v) for v in descriptions.values())
    print(f"[VisionDirector] 🎬 Analysis complete: {total} descriptions from {len(descriptions)} videos")
    return descriptions


def format_descriptions_for_llm(descriptions, all_images, fps):
    """
    Format video descriptions into a rich text context for the LLM.

    Returns a string with per-video content analysis + advertising context.
    """
    if not descriptions:
        return ""

    lines = []
    lines.append("DETAILED VIDEO CONTENT ANALYSIS (by Florence-2 Vision AI):")
    lines.append("=" * 60)

    for vid_idx in sorted(descriptions.keys()):
        n_frames = all_images[vid_idx].shape[0] if vid_idx in all_images else 0
        duration = n_frames / fps if fps > 0 else 0
        lines.append(f"\nVideo {vid_idx} ({n_frames} frames, {duration:.1f}s):")

        for label, caption in descriptions[vid_idx]:
            lines.append(f"  {label}: \"{caption}\"")

    lines.append("\n" + "=" * 60)
    lines.append("""
ADVERTISING EDITING INTELLIGENCE:
You are editing a PRODUCT ADVERTISEMENT. Use the visual analysis above to make editing decisions that SELL:

1. HERO SHOTS (product clearly visible, good lighting) → Use these as anchor points.
   Place them at the start, after transitions, and at the end.

2. ACTION SHOTS (spray, pour, hands interacting) → Use these for energy peaks.
   These create DESIRE. Cut to them with punchy transitions (flash, zoom punch).

3. DETAIL SHOTS (close-ups, textures, labels) → Use these as breathing moments.
   Let the viewer appreciate craftsmanship. Slightly longer cuts (1.5-2.5s).

4. ATMOSPHERE SHOTS (lighting effects, bokeh, reflections) → Use these for mood.
   These transitions between hero and action shots.

5. STORY ARC: Open strong (hero) → build intrigue (details) → energy peak (action) →
   satisfying close (hero from new angle).

6. NEVER put two similar shots back-to-back. Contrast is key.
   Follow a wide shot with a close-up. Follow static with motion.
""")

    return "\n".join(lines)
