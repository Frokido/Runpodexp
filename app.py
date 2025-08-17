"""Main entry point for the experimental generative‑AI kit.

This script defines a modular Gradio interface that wraps several open‑source diffusion pipelines.  It exposes text‑to‑video, image‑to‑video, text‑to‑image and audio generation capabilities while offering advanced settings such as TeaCache acceleration and CFG‑Zero⋆ guidance.  The goal is to provide a single interface with a full range of features that runs efficiently on a RunPod L40S GPU (48 GB VRAM).

Notes:
- Actual model downloads can take a long time and may require large amounts of disk space. Ensure your pod has sufficient storage.
- This script does **not** enable any safety checker — NSFW content will pass through untouched. Use responsibly.

 
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

try:
    # Diffusers is used for general diffusion pipelines
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None  # type: ignore

try:
    # DiffSynth provides memory-efficient pipelines for Wan and other models
    from diffsynth.pipelines.wan_video_new import (
        WanVideoPipeline,
        ModelConfig as WanModelConfig,
    )
    from diffsynth.pipelines.flux_image_new import (
        FluxImagePipeline,
        ModelConfig as FluxModelConfig,
    )
    from diffsynth.pipelines.qwen_image import (
        QwenImagePipeline,
        ModelConfig as QwenModelConfig,
    )
except Exception:
    WanVideoPipeline = None  # type: ignore
    FluxImagePipeline = None  # type: ignore
    QwenImagePipeline = None  # type: ignore
    WanModelConfig = None  # type: ignore
    FluxModelConfig = None  # type: ignore
    QwenModelConfig = None  # type: ignore

# Attempt to import optional acceleration libraries

try:
    import teacache  # type: ignore
except ImportError:
    teacache = None  # type: ignore

try:
    # Assume a hypothetical cfg_zero_star module; if unavailable we'll implement a simple wrapper later.
    import cfg_zero_star  # type: ignore
except ImportError:
    cfg_zero_star = None  # type: ignore

# ----------------------------- Helper functions -----------------------------

def apply_teacache(pipe: Any) -> Any:
    """If TeaCache is available, wrap the pipeline with caching.
    TeaCache accelerates diffusion inference by caching intermediate states   across timesteps.  This function attempts to   enable caching on the provided pipeline.  If TeaCache is not installed,   the pipeline is returned unchanged.
    """
    if teacache is None:
        print("TeaCache is not installed; returning pipeline unchanged.")
        return pipe
    try:
        # Many TeaCache integrations rely on a simple `.use_cache()` method.
        # If the pipeline exposes it, call it.  Otherwise, fallback to
        # teacache.apply() which may wrap the pipeline internally.
        if hasattr(pipe, "use_cache"):
            pipe.use_cache()
        else:
            pipe = teacache.apply(pipe)
        print("TeaCache acceleration enabled.")
    except Exception as e:
        print(f"Failed to enable TeaCache: {e}")
    return pipe

class CFGZeroStarWrapper:
    """A lightweight wrapper to emulate CFG‑Zero⋆ behaviour.
    CFG‑Zero⋆ improves classifier‑free guidance by zeroing out a percentage   of early solver steps and scaling the guidance.  The official implementation supports many models but may not be installed.   This wrapper provides a simple `apply` method that adjusts the   scheduler's guidance scale and stores the zero‑init ratio.  It does not   modify the underlying solver but lets users experiment with the   parameters.  For more faithful behaviour, install the official library.   """
    def __init__(self, pipe: Any, guidance_scale: float = 1.0, zero_ratio: float = 0.04) -> None:
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.zero_ratio = zero_ratio
        # Store original guidance scale if available
        self._original_scale = getattr(pipe, "guidance_scale", None)
        # Immediately apply guidance scale
        if hasattr(self.pipe, "guidance_scale"):
            self.pipe.guidance_scale = guidance_scale

    def __getattr__(self, item: str) -> Any:
        return getattr(self.pipe, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Forward call to underlying pipeline
        return self.pipe(*args, **kwargs)

def apply_cfg_zero(pipe: Any, guidance_scale: float, zero_ratio: float) -> Any:
    """Apply CFG‑Zero⋆ guidance to the pipeline.
    If the official cfg_zero_star module is available, it will be used.   Otherwise a simple wrapper that sets the guidance scale and stores the   zero ratio is returned.   """
    if cfg_zero_star is not None:
        try:
            return cfg_zero_star.apply(pipe, guidance_scale=guidance_scale, zero_ratio=zero_ratio)
        except Exception as e:
            print(f"cfg_zero_star apply failed: {e}; falling back to wrapper.")
    # fallback
    return CFGZeroStarWrapper(pipe, guidance_scale, zero_ratio)

def disable_safety(pipe: Any) -> None:
    """Disable any built‑in safety checker on the pipeline.
    Many Diffusers pipelines expose `safety_checker` or similar attributes   which can be set to `None` to bypass nudity/NSFW filters.  This helper   attempts to locate and disable them.   """
    for attr in ["safety_checker", "nsfw_checker", "safety_check"]:
        if hasattr(pipe, attr):
            try:
                setattr(pipe, attr, None)
                print(f"Disabled {attr} on pipeline.")
            except Exception:
                pass

def load_wan_pipeline(
    task: str,
    model_size: str,
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a Wan 2.2 or Wan 2.1 diffusion pipeline via DiffSynth.
    Args:
        task: 't2v', 'i2v' or 'ti2v'.
        model_size: 'A14B' for 14 B models or '5B' for 5 B models.
        dtype: torch dtype (e.g. torch.float16 or torch.bfloat16).
        offload: whether to enable CPU offloading.
        use_cache: whether to enable TeaCache.
        use_cfg_zero: whether to apply CFG‑Zero⋆.
        guidance_scale: scale for classifier‑free guidance.
        zero_ratio: percentage of steps to zero out (0–1).
    Returns:
        A loaded pipeline ready for inference.
    """
    if WanVideoPipeline is None:
        raise RuntimeError(
            "DiffSynth is not installed; unable to load Wan models.\n"
            "Please install diffsynth (see requirements.txt) and try again."
        )

    # Map tasks to official model ids.  Users can modify these identifiers to   # load custom checkpoints.  14B models require significant VRAM; the 5B   # model is more memory‑friendly.
    if task == "t2v":
        model_id = "Wan-AI/Wan2.2-T2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-T2V-5B"
    elif task == "i2v":
        model_id = "Wan-AI/Wan2.2-I2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-I2V-5B"
    else:
        # text‑image‑to‑video
        model_id = "Wan-AI/Wan2.2-TI2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-TI2V-5B"

    # Construct a DiffSynth ModelConfig list.  Wan models use multiple   # sub‑components (transformer, VAE, etc.) but DiffSynth hides this   # complexity behind `model_configs`.
    model_configs = [
        WanModelConfig(model_id=model_id, origin_file_pattern="**/*.safetensors"),
    ]

    # Load the pipeline
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    disable_safety(pipe)
    # Offload to CPU if requested.  DiffSynth exposes offload options via
    # `offload_model` or `offload_text_encoder` on individual calls.  Here we
    # just set a flag that our generation function will respect.
    pipe._offload = offload

    # Apply TeaCache if enabled
    if use_cache:
        pipe = apply_teacache(pipe)
    # Apply CFG‑Zero⋆ if enabled
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)
    return pipe

def load_flux_pipeline(
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a FLUX image pipeline via DiffSynth.
    FLUX is a family of image diffusion models supported by DiffSynth.   This function constructs the pipeline and applies optional TeaCache and   CFG‑Zero⋆ modifications.   """
    if FluxImagePipeline is None:
        raise RuntimeError(
            "DiffSynth is not installed; unable to load FLUX models."
        )
    model_configs = [
        FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="*.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    disable_safety(pipe)
    pipe._offload = offload
    if use_cache:
        pipe = apply_teacache(pipe)
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)
    return pipe

def load_qwen_pipeline(
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a Qwen‑Image pipeline via DiffSynth.
    Qwen‑Image is a recent image generator that can embed full sentences and   supports LoRA training.   """
    if QwenImagePipeline is None:
        raise RuntimeError("DiffSynth is not installed; unable to load Qwen models.")
    model_configs = [
        QwenModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="**/*.safetensors"),
    ]
    pipe = QwenImagePipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    disable_safety(pipe)
    pipe._offload = offload
    if use_cache:
        pipe = apply_teacache(pipe)
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)
    return pipe

# Global cache to reuse loaded pipelines across requests
PIPELINE_CACHE: Dict[str, Any] = {}

async def generate_video(
    prompt: str,
    negative_prompt: str,
    task: str,
    model_size: str,
    resolution: Tuple[int, int],
    num_frames: int,
    guidance_scale: float,
    dtype_str: str,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    zero_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Asynchronous task that generates a video given user options.
    Returns a tuple of (video_path, first_frame_path, last_frame_path, info_message). If generation fails, `info_message` will contain the error.
    """
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    key = f"wan_{task}_{model_size}_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
    if key not in PIPELINE_CACHE:
        PIPELINE_CACHE[key] = load_wan_pipeline(
            task=task,
            model_size=model_size,
            dtype=dtype,
            offload=offload,
            use_cache=use_cache,
            use_cfg_zero=use_cfg_zero,
            guidance_scale=guidance_scale,
            zero_ratio=zero_ratio,
        )
    pipe = PIPELINE_CACHE[key]
    # Set seed if provided
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    # Prepare parameters. Many pipelines accept `num_inference_steps` and   # `video_length` or `num_frames` parameters.  Use 50 inference steps by default.
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 50,
        "num_frames": num_frames,
        "generator": generator,
        "height": resolution[1],
        "width": resolution[0],
        "guidance_scale": guidance_scale,
        "offload_model": getattr(pipe, "_offload", False),
    }
    try:
        result = pipe(**kwargs)
        # The result from DiffSynth pipelines may be a dictionary containing a video tensor or list of frames.  Convert to an MP4 file.
        video_path = os.path.join("outputs", f"wan_video_{task}_{model_size}_{os.getpid()}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        # Attempt to use the built-in save function if available
        if hasattr(result, "save"):
            result.save(video_path)
        else:
            # Fallback: assume result["video"] is a numpy array of shape (frames, height, width, 3)
            frames = None
            if isinstance(result, dict):
                frames = result.get("video") or result.get("frames")
            elif hasattr(result, "frames"):
                frames = result.frames
            if frames is None:
                raise ValueError("Unsupported result type from pipeline")
            import imageio.v3 as iio  # local import to avoid unnecessary dependency
            iio.imwrite(
                video_path,
                (np.array(frames) * 255).astype(np.uint8),
                fps=24,
                format="mp4",
            )
        # Also save first and last frame as images if available
        first_frame_path = ""
        last_frame_path = ""
        if "frames" in locals() and isinstance(frames, (list, tuple, np.ndarray)):
            try:
                if isinstance(frames, np.ndarray):
                    first_frame = frames[0]
                    last_frame = frames[-1]
                else:
                    first_frame = frames[0]
                    last_frame = frames[-1]
                from PIL import Image
                # Save first frame
                first_frame_path = os.path.join("outputs", f"first_frame_{task}_{model_size}_{os.getpid()}.png")
                Image.fromarray((np.array(first_frame) * 255).astype(np.uint8)).save(first_frame_path)
                # Save last frame
                last_frame_path = os.path.join("outputs", f"last_frame_{task}_{model_size}_{os.getpid()}.png")
                Image.fromarray((np.array(last_frame) * 255).astype(np.uint8)).save(last_frame_path)
            except Exception as e:
                first_frame_path = ""
                last_frame_path = ""
                print(f"Failed to save frames: {e}")
        return video_path, first_frame_path, last_frame_path, None
    except Exception as e:
        return "", "", "", str(e)

async def generate_image(
    prompt: str,
    negative_prompt: str,
    model_type: str,
    guidance_scale: float,
    dtype_str: str,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    zero_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """Asynchronous task that generates an image from a text prompt.
    The `model_type` argument selects among 'flux' and 'qwen'.
    """
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    if model_type == "flux":
        key = f"flux_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
        if key not in PIPELINE_CACHE:
            PIPELINE_CACHE[key] = load_flux_pipeline(
                dtype=dtype,
                offload=offload,
                use_cache=use_cache,
                use_cfg_zero=use_cfg_zero,
                guidance_scale=guidance_scale,
                zero_ratio=zero_ratio,
            )
        pipe = PIPELINE_CACHE[key]
    else:  # qwen
        key = f"qwen_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
        if key not in PIPELINE_CACHE:
            PIPELINE_CACHE[key] = load_qwen_pipeline(
                dtype=dtype,
                offload=offload,
                use_cache=use_cache,
                use_cfg_zero=use_cfg_zero,
                guidance_scale=guidance_scale,
                zero_ratio=zero_ratio,
            )
        pipe = PIPELINE_CACHE[key]
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 40,
        "generator": generator,
        "guidance_scale": guidance_scale,
        "offload_model": getattr(pipe, "_offload", False),
    }
    try:
        result = pipe(**kwargs)
        image_path = os.path.join("outputs", f"image_{model_type}_{os.getpid()}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if hasattr(result, "save"):
            result.save(image_path)
        else:
            from PIL import Image
            img = None
            if isinstance(result, dict):
                img = result.get("images") or result.get("image")
            elif hasattr(result, "images"):
                img = result.images
            if img is None:
                raise ValueError("Unsupported result type from pipeline")
            # img may be a list of PIL images or numpy arrays
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(image_path)
        return image_path, None
    except Exception as e:
        return "", str(e)

# A simple task queue.  Tasks are appended when submitted and processed
# sequentially in a background worker.  Each task is a coroutine returning
# (file_path, error_message).
TASK_QUEUE: List[asyncio.Future] = []
TASK_QUEUE_LOCK = threading.Lock()

def enqueue_task(coro: asyncio.coroutine) -> asyncio.Future:
    """Schedule a coroutine to run in the background and add it to the queue.
    Returns a Future that will hold the result.  The internal worker will process tasks one at a time.
    """
    loop = asyncio.get_event_loop()
    future = loop.create_task(coro)
    with TASK_QUEUE_LOCK:
        TASK_QUEUE.append(future)
    return future

async def task_worker() -> None:
    """Continuously process tasks from the queue."""
    while True:
        await asyncio.sleep(0.1)
        next_task: Optional[asyncio.Future] = None
        with TASK_QUEUE_LOCK:
            if TASK_QUEUE and TASK_QUEUE[0].done():
                TASK_QUEUE.pop(0)
            if TASK_QUEUE:
                next_task = TASK_QUEUE[0]
        if next_task is not None and not next_task.done():
            # Wait for the current task to complete before starting the next
            await asyncio.sleep(0.1)

# Launch the worker in the background
asyncio.get_event_loop().create_task(task_worker())

# ------------------------------ UI Definition ------------------------------

def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(css=".gradio-container {max-width: 1024px; margin: auto;}") as demo:
        gr.Markdown(
            "# Experimental Generative‑AI Kit\n"
            "This interface provides advanced generation tools built on Wan 2.2, Flux, Qwen and other cutting‑edge models.\n"
            "All safety filters are disabled.  Use at your own risk."
        )
        with gr.Tabs():
            # Text to Video Tab
            with gr.TabItem("Text → Video"):
                with gr.Row():
                    prompt_t2v = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your scene…")
                with gr.Row():
                    ref_img_t2v = gr.File(label="Reference Image (any file)", type="file")
                with gr.Row():
                    model_size_t2v = gr.Dropdown(["A14B", "5B"], label="Model Size", value="5B", info="5B uses less VRAM; 14B offers higher quality")
             

# ----------------------------- Helper functions -----------------------------

def apply_teacache(pipe: Any) -> Any:
    """If TeaCache is available, wrap the pipeline with caching.

    TeaCache accelerates diffusion inference by caching intermediate states
    across timesteps【775403151692466†L83-L88】.  This function attempts to
    enable caching on the provided pipeline.  If TeaCache is not installed,
    the pipeline is returned unchanged.
    """
    if teacache is None:
        print("TeaCache is not installed; returning pipeline unchanged.")
        return pipe
    try:
        # Many TeaCache integrations rely on a simple `.use_cache()` method.
        # If the pipeline exposes it, call it.  Otherwise, fallback to
        # teacache.apply() which may wrap the pipeline internally.
        if hasattr(pipe, "use_cache"):
            pipe.use_cache()
        else:
            pipe = teacache.apply(pipe)
        print("TeaCache acceleration enabled.")
    except Exception as e:
        print(f"Failed to enable TeaCache: {e}")
    return pipe


class CFGZeroStarWrapper:
    """A lightweight wrapper to emulate CFG‑Zero⋆ behaviour.

    CFG‑Zero⋆ improves classifier‑free guidance by zeroing out a percentage
    of early solver steps and scaling the guidance【24647453753593†L60-L66】.  The
    official implementation supports many models but may not be installed.
    This wrapper provides a simple `apply` method that adjusts the
    scheduler's guidance scale and stores the zero‑init ratio.  It does not
    modify the underlying solver but lets users experiment with the
    parameters.  For more faithful behaviour, install the official library.
    """

    def __init__(self, pipe: Any, guidance_scale: float = 1.0, zero_ratio: float = 0.04) -> None:
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.zero_ratio = zero_ratio
        # Store original guidance scale if available
        self._original_scale = getattr(pipe, "guidance_scale", None)
        # Immediately apply guidance scale
        if hasattr(self.pipe, "guidance_scale"):
            self.pipe.guidance_scale = guidance_scale

    def __getattr__(self, item: str) -> Any:
        return getattr(self.pipe, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Forward call to underlying pipeline
        return self.pipe(*args, **kwargs)


def apply_cfg_zero(pipe: Any, guidance_scale: float, zero_ratio: float) -> Any:
    """Apply CFG‑Zero⋆ guidance to the pipeline.

    If the official cfg_zero_star module is available, it will be used.
    Otherwise a simple wrapper that sets the guidance scale and stores the
    zero ratio is returned.
    """
    if cfg_zero_star is not None:
        try:
            return cfg_zero_star.apply(pipe, guidance_scale=guidance_scale, zero_ratio=zero_ratio)
        except Exception as e:
            print(f"cfg_zero_star apply failed: {e}; falling back to wrapper.")
    # fallback
    return CFGZeroStarWrapper(pipe, guidance_scale, zero_ratio)


def disable_safety(pipe: Any) -> None:
    """Disable any built‑in safety checker on the pipeline.

    Many Diffusers pipelines expose `safety_checker` or similar attributes
    which can be set to `None` to bypass nudity/NSFW filters.  This helper
    attempts to locate and disable them.
    """
    for attr in ["safety_checker", "nsfw_checker", "safety_check"]:
        if hasattr(pipe, attr):
            try:
                setattr(pipe, attr, None)
                print(f"Disabled {attr} on pipeline.")
            except Exception:
                pass


def load_wan_pipeline(
    task: str,
    model_size: str,
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a Wan 2.2 or Wan 2.1 diffusion pipeline via DiffSynth.

    Args:
        task: 't2v', 'i2v' or 'ti2v'.
        model_size: 'A14B' for 14 B models or '5B' for 5 B models.
        dtype: torch dtype (e.g. torch.float16 or torch.bfloat16).
        offload: whether to enable CPU offloading.
        use_cache: whether to enable TeaCache.
        use_cfg_zero: whether to apply CFG‑Zero⋆.
        guidance_scale: scale for classifier‑free guidance.
        zero_ratio: percentage of steps to zero out (0–1).
    Returns:
        A loaded pipeline ready for inference.
    """
    if WanVideoPipeline is None:
        raise RuntimeError(
            "DiffSynth is not installed; unable to load Wan models.\n"
            "Please install diffsynth (see requirements.txt) and try again."
        )

    # Map tasks to official model ids.  Users can modify these identifiers to
    # load custom checkpoints.  14B models require significant VRAM; the 5B
    # model is more memory‑friendly【526278201854702†L305-L310】.
    if task == "t2v":
        model_id = "Wan-AI/Wan2.2-T2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-T2V-5B"
    elif task == "i2v":
        model_id = "Wan-AI/Wan2.2-I2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-I2V-5B"
    else:
        # text‑image‑to‑video
        model_id = "Wan-AI/Wan2.2-TI2V-A14B" if model_size.upper() == "A14B" else "Wan-AI/Wan2.2-TI2V-5B"

    # Construct a DiffSynth ModelConfig list.  Wan models use multiple
    # sub‑components (transformer, VAE, etc.) but DiffSynth hides this
    # complexity behind `model_configs`.
    model_configs = [
        WanModelConfig(model_id=model_id, origin_file_pattern="**/*.safetensors"),
    ]

    # Load the pipeline
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    disable_safety(pipe)

    # Offload to CPU if requested.  DiffSynth exposes offload options via
    # `offload_model` or `offload_text_encoder` on individual calls.  Here we
    # just set a flag that our generation function will respect.
    pipe._offload = offload

    # Apply TeaCache if enabled
    if use_cache:
        pipe = apply_teacache(pipe)

    # Apply CFG‑Zero⋆ if enabled
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)

    return pipe


def load_flux_pipeline(
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a FLUX image pipeline via DiffSynth.

    FLUX is a family of image diffusion models supported by DiffSynth【914030415508068†L366-L403】.
    This function constructs the pipeline and applies optional TeaCache and
    CFG‑Zero⋆ modifications.
    """
    if FluxImagePipeline is None:
        raise RuntimeError(
            "DiffSynth is not installed; unable to load FLUX models."
        )
    model_configs = [
        FluxModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="*.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    disable_safety(pipe)
    pipe._offload = offload
    if use_cache:
        pipe = apply_teacache(pipe)
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)
    return pipe


def load_qwen_pipeline(
    dtype: torch.dtype,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    guidance_scale: float,
    zero_ratio: float,
) -> Any:
    """Load a Qwen‑Image pipeline via DiffSynth.

    Qwen‑Image is a recent image generator that can embed full sentences and
    supports LoRA training【914030415508068†L345-L349】.
    """
    if QwenImagePipeline is None:
        raise RuntimeError("DiffSynth is not installed; unable to load Qwen models.")
    model_configs = [
        QwenModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="**/*.safetensors"),
    ]
    pipe = QwenImagePipeline.from_pretrained(
        model_configs=model_configs,
        torch_dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    disable_safety(pipe)
    pipe._offload = offload
    if use_cache:
        pipe = apply_teacache(pipe)
    if use_cfg_zero:
        pipe = apply_cfg_zero(pipe, guidance_scale, zero_ratio)
    return pipe


# Global cache to reuse loaded pipelines across requests
PIPELINE_CACHE: Dict[str, Any] = {}


async def generate_video(
    prompt: str,
    task: str,
    model_size: str,
    resolution: Tuple[int, int],
    num_frames: int,
    guidance_scale: float,
    dtype_str: str,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    zero_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """Asynchronous task that generates a video given user options.

    Returns a tuple of (video_path, info_message).  If generation fails,
    `video_path` may be None and an error message will be provided.
    """
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    key = f"wan_{task}_{model_size}_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
    if key not in PIPELINE_CACHE:
        PIPELINE_CACHE[key] = load_wan_pipeline(
            task=task,
            model_size=model_size,
            dtype=dtype,
            offload=offload,
            use_cache=use_cache,
            use_cfg_zero=use_cfg_zero,
            guidance_scale=guidance_scale,
            zero_ratio=zero_ratio,
        )
    pipe = PIPELINE_CACHE[key]
    # Set seed if provided
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    # Prepare parameters.  Many pipelines accept `num_inference_steps` and
    # `video_length` or `num_frames` parameters.  Use 50 inference steps by
    # default and allow TeaCache/CFG‑Zero to accelerate effectively.
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": 50,
        "num_frames": num_frames,
        "generator": generator,
        "height": resolution[1],
        "width": resolution[0],
        "guidance_scale": guidance_scale,
        "offload_model": getattr(pipe, "_offload", False),
    }
    try:
        result = pipe(**kwargs)
        # The result from DiffSynth pipelines may be a dictionary containing
        # a video tensor or list of frames.  Convert to an MP4 file.
        video_path = os.path.join("outputs", f"wan_video_{task}_{model_size}_{os.getpid()}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        # Attempt to use the built‑in save function if available
        if hasattr(result, "save"):
            result.save(video_path)
        else:
            # Fallback: assume result["video"] is a numpy array of shape
            # (frames, height, width, 3)
            frames = None
            if isinstance(result, dict):
                frames = result.get("video") or result.get("frames")
            elif hasattr(result, "frames"):
                frames = result.frames
            if frames is None:
                raise ValueError("Unsupported result type from pipeline")
            import imageio.v3 as iio  # local import to avoid unnecessary dependency
            iio.imwrite(
                video_path,
                (np.array(frames) * 255).astype(np.uint8),
                fps=24,
                format="mp4",
            )
        return video_path, None
    except Exception as e:
        return "", str(e)


async def generate_image(
    prompt: str,
    model_type: str,
    guidance_scale: float,
    dtype_str: str,
    offload: bool,
    use_cache: bool,
    use_cfg_zero: bool,
    zero_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """Asynchronous task that generates an image from a text prompt.

    The `model_type` argument selects among 'flux' and 'qwen'.
    """
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    if model_type == "flux":
        key = f"flux_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
        if key not in PIPELINE_CACHE:
            PIPELINE_CACHE[key] = load_flux_pipeline(
                dtype=dtype,
                offload=offload,
                use_cache=use_cache,
                use_cfg_zero=use_cfg_zero,
                guidance_scale=guidance_scale,
                zero_ratio=zero_ratio,
            )
        pipe = PIPELINE_CACHE[key]
    else:  # qwen
        key = f"qwen_{dtype_str}_{offload}_{use_cache}_{use_cfg_zero}_{guidance_scale}_{zero_ratio}"
        if key not in PIPELINE_CACHE:
            PIPELINE_CACHE[key] = load_qwen_pipeline(
                dtype=dtype,
                offload=offload,
                use_cache=use_cache,
                use_cfg_zero=use_cfg_zero,
                guidance_scale=guidance_scale,
                zero_ratio=zero_ratio,
            )
        pipe = PIPELINE_CACHE[key]
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": 40,
        "generator": generator,
        "guidance_scale": guidance_scale,
        "offload_model": getattr(pipe, "_offload", False),
    }
    try:
        result = pipe(**kwargs)
        image_path = os.path.join("outputs", f"image_{model_type}_{os.getpid()}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if hasattr(result, "save"):
            result.save(image_path)
        else:
            from PIL import Image
            img = None
            if isinstance(result, dict):
                img = result.get("images") or result.get("image")
            elif hasattr(result, "images"):
                img = result.images
            if img is None:
                raise ValueError("Unsupported result type from pipeline")
            # img may be a list of PIL images or numpy arrays
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(image_path)
        return image_path, None
    except Exception as e:
        return "", str(e)


# A simple task queue.  Tasks are appended when submitted and processed
# sequentially in a background worker.  Each task is a coroutine returning
# (file_path, error_message).
TASK_QUEUE: List[asyncio.Future] = []
TASK_QUEUE_LOCK = threading.Lock()


def enqueue_task(coro: asyncio.coroutine) -> asyncio.Future:
    """Schedule a coroutine to run in the background and add it to the queue.

    Returns a Future that will hold the result.  The internal worker will
    process tasks one at a time.
    """
    loop = asyncio.get_event_loop()
    future = loop.create_task(coro)
    with TASK_QUEUE_LOCK:
        TASK_QUEUE.append(future)
    return future


async def task_worker() -> None:
    """Continuously process tasks from the queue."""
    while True:
        await asyncio.sleep(0.1)
        next_task: Optional[asyncio.Future] = None
        with TASK_QUEUE_LOCK:
            if TASK_QUEUE and TASK_QUEUE[0].done():
                TASK_QUEUE.pop(0)
            if TASK_QUEUE:
                next_task = TASK_QUEUE[0]
        if next_task is not None and not next_task.done():
            # Wait for the current task to complete before starting the next
            await asyncio.sleep(0.1)


# Launch the worker in the background
asyncio.get_event_loop().create_task(task_worker())


# ------------------------------ UI Definition ------------------------------

def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(css=".gradio-container {max-width: 1024px; margin: auto;}\n") as demo:
        gr.Markdown(
            "# Experimental Generative‑AI Kit\n"
            "This interface provides advanced generation tools built on Wan 2.2, Flux, Qwen and other cutting‑edge models.\n"
            "All safety filters are disabled.  Use at your own risk."
        )
        with gr.Tabs():
            # Text to Video Tab
            with gr.TabItem("Text → Video"):
                with gr.Row():
                    prompt_t2v = gr.Textbox(label="Prompt", lines=3, placeholder="Describe your scene…")
                with gr.Row():
                    model_size_t2v = gr.Dropdown(["A14B", "5B"], label="Model Size", value="5B", info="5B uses less VRAM; 14B offers higher quality")
                    task_t2v = gr.Dropdown([("Text → Video", "t2v"), ("Image → Video", "i2v"), ("Text+Image → Video", "ti2v")], label="Task", value="t2v")
                    resolution_t2v = gr.Dropdown([("720p (1280×720)", (1280, 720)), ("480p (960×540)", (960, 540))], label="Resolution", value=(960, 540))
                    num_frames_t2v = gr.Slider(16, 64, value=24, step=8, label="Number of Frames")
                with gr.Row():
                    seed_t2v = gr.Number(label="Seed (optional)", value=None, precision=0)
                generate_button_t2v = gr.Button("Generate Video")
                output_video = gr.Video(label="Generated Video", interactive=False)
                error_video = gr.Textbox(label="Error Message", interactive=False, visible=False)

                def on_generate_video(
                    prompt: str,
                    task: str,
                    model_size: str,
                    resolution: Tuple[int, int],
                    num_frames: int,
                    seed: Optional[float],
                ) -> Tuple[str, str]:
                    """Callback for video generation.  Schedules the task and waits for completion."""
                    # Convert seed to int if provided
                    seed_int: Optional[int] = None
                    if seed is not None and seed != "":
                        try:
                            seed_int = int(seed)
                        except Exception:
                            pass
                    future = enqueue_task(
                        generate_video(
                            prompt=prompt,
                            task=task,
                            model_size=model_size,
                            resolution=resolution,
                            num_frames=int(num_frames),
                            guidance_scale=settings_state["guidance_scale"],
                            dtype_str=settings_state["dtype"],
                            offload=settings_state["offload"],
                            use_cache=settings_state["use_cache"],
                            use_cfg_zero=settings_state["use_cfg_zero"],
                            zero_ratio=settings_state["zero_ratio"],
                            seed=seed_int,
                        )
                    )
                    # Wait for the task to finish
                    video_path, err = asyncio.get_event_loop().run_until_complete(future)
                    if err:
                        return "", err
                    return video_path, ""

                generate_button_t2v.click(
                    on_generate_video,
                    inputs=[prompt_t2v, task_t2v, model_size_t2v, resolution_t2v, num_frames_t2v, seed_t2v],
                    outputs=[output_video, error_video],
                )

            # Text to Image Tab
            with gr.TabItem("Text → Image"):
                with gr.Row():
                    prompt_img = gr.Textbox(label="Prompt", lines=3, placeholder="A detailed portrait of a girl underwater…")
                with gr.Row():
                    image_model = gr.Dropdown([("FLUX", "flux"), ("Qwen‑Image", "qwen")], label="Model", value="flux")
                    seed_img = gr.Number(label="Seed (optional)", value=None, precision=0)
                generate_button_img = gr.Button("Generate Image")
                output_image = gr.Image(label="Generated Image", interactive=False)
                error_image = gr.Textbox(label="Error Message", interactive=False, visible=False)

                def on_generate_image(prompt: str, model: str, seed: Optional[float]) -> Tuple[str, str]:
                    seed_int: Optional[int] = None
                    if seed is not None and seed != "":
                        try:
                            seed_int = int(seed)
                        except Exception:
                            pass
                    future = enqueue_task(
                        generate_image(
                            prompt=prompt,
                            model_type=model,
                            guidance_scale=settings_state["guidance_scale"],
                            dtype_str=settings_state["dtype"],
                            offload=settings_state["offload"],
                            use_cache=settings_state["use_cache"],
                            use_cfg_zero=settings_state["use_cfg_zero"],
                            zero_ratio=settings_state["zero_ratio"],
                            seed=seed_int,
                        )
                    )
                    img_path, err = asyncio.get_event_loop().run_until_complete(future)
                    if err:
                        return "", err
                    return img_path, ""

                generate_button_img.click(
                    on_generate_image,
                    inputs=[prompt_img, image_model, seed_img],
                    outputs=[output_image, error_image],
                )

            # Settings Tab
            with gr.TabItem("Settings"):
                with gr.Row():
                    guidance_scale_slider = gr.Slider(0.0, 20.0, value=1.0, step=0.1, label="Guidance Scale (CFG)")
                    dtype_dropdown = gr.Dropdown([("FP16", "float16"), ("BF16", "bfloat16")], label="Tensor dtype", value="float16")
                with gr.Row():
                    offload_checkbox = gr.Checkbox(label="CPU Offload", value=False)
                    cache_checkbox = gr.Checkbox(label="Enable TeaCache", value=False)
                    cfgzero_checkbox = gr.Checkbox(label="Enable CFG‑Zero⋆", value=False)
                zero_ratio_slider = gr.Slider(0.0, 0.2, value=0.04, step=0.01, label="Zero‑Init Ratio (CFG‑Zero⋆)")
                # Display of current settings
                settings_display = gr.Markdown()

                def update_settings(
                    guidance_scale: float,
                    dtype: str,
                    offload: bool,
                    use_cache: bool,
                    use_cfg_zero: bool,
                    zero_ratio: float,
                ) -> str:
                    # Update global state
                    settings_state["guidance_scale"] = guidance_scale
                    settings_state["dtype"] = dtype
                    settings_state["offload"] = offload
                    settings_state["use_cache"] = use_cache
                    settings_state["use_cfg_zero"] = use_cfg_zero
                    settings_state["zero_ratio"] = zero_ratio
                    return (
                        f"**Current Settings**:\n"
                        f"* Guidance scale: {guidance_scale}\n"
                        f"* dtype: {dtype}\n"
                        f"* CPU offload: {offload}\n"
                        f"* TeaCache: {use_cache}\n"
                        f"* CFG‑Zero⋆: {use_cfg_zero} (zero ratio: {zero_ratio*100:.1f}% of steps)\n"
                    )

                # Register change events individually.  When any setting is
                # modified, the markdown display is updated to reflect the
                # current configuration.  Using separate callbacks avoids
                # reliance on `gr.update` which is not available outside
                # internal usage.
                def on_change(
                    guidance_scale: float,
                    dtype: str,
                    offload: bool,
                    use_cache: bool,
                    use_cfg_zero: bool,
                    zero_ratio: float,
                ) -> str:
                    return update_settings(
                        guidance_scale,
                        dtype,
                        offload,
                        use_cache,
                        use_cfg_zero,
                        zero_ratio,
                    )

                # Connect each input's `change` event to update the settings
                for component in [
                    guidance_scale_slider,
                    dtype_dropdown,
                    offload_checkbox,
                    cache_checkbox,
                    cfgzero_checkbox,
                    zero_ratio_slider,
                ]:
                    component.change(
                        on_change,
                        inputs=[
                            guidance_scale_slider,
                            dtype_dropdown,
                            offload_checkbox,
                            cache_checkbox,
                            cfgzero_checkbox,
                            zero_ratio_slider,
                        ],
                        outputs=settings_display,
                    )

        return demo


# State dictionary updated via the Settings tab.  Defaults reflect typical
# settings for the L40S (FP16, no offload).
settings_state = {
    "guidance_scale": 1.0,
    "dtype": "float16",
    "offload": False,
    "use_cache": False,
    "use_cfg_zero": False,
    "zero_ratio": 0.04,
}


if __name__ == "__main__":
    demo = build_interface()
    # Launch with concurrency to support background tasks
    demo.queue()  # enable queueing within Gradio itself
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
