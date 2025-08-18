#!/usr/bin/env python3
"""
RunPod AI Kit - Fixed Production Version
Addresses: AsyncIO issues, thread safety, resource management, error handling
Maintains: NSFW capabilities for testing/experimentation
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc

import torch
import gradio as gr
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CACHE_SIZE = 3  # Maximum number of cached pipelines
TASK_TIMEOUT = 300  # 5 minutes timeout for tasks

# Thread-safe pipeline cache with proper cleanup
class PipelineCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Clean up CUDA memory from evicted pipeline
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used pipeline with proper cleanup"""
        if not self._cache:
            return
            
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        pipeline = self._cache.pop(oldest_key)
        self._access_times.pop(oldest_key)
        
        # Cleanup CUDA memory
        if hasattr(pipeline, 'to'):
            pipeline.to('cpu')
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Evicted pipeline: {oldest_key}")
    
    def clear(self) -> None:
        with self._lock:
            for pipeline in self._cache.values():
                if hasattr(pipeline, 'to'):
                    pipeline.to('cpu')
            self._cache.clear()
            self._access_times.clear()
            torch.cuda.empty_cache()
            gc.collect()

# Global instances
PIPELINE_CACHE = PipelineCache()
TASK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AI-Task")

# Resolution configurations with proper validation
RESOLUTION_PRESETS = {
    "Square (1024x1024)": (1024, 1024),
    "Portrait (768x1024)": (768, 1024),
    "Landscape (1024x768)": (1024, 768),
    "Widescreen (1280x720)": (1280, 720),
    "Ultra-wide (1536x640)": (1536, 640)
}

VIDEO_RESOLUTION_PRESETS = {
    "480p (854x480)": (854, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080)
}

@contextmanager
def cuda_memory_cleanup():
    """Context manager for CUDA memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def validate_resolution(width: int, height: int, max_pixels: int = 2073600) -> Tuple[int, int]:
    """Validate and adjust resolution to prevent memory issues"""
    # Ensure dimensions are multiples of 8 for stable diffusion
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)
    
    # Check total pixel count
    total_pixels = width * height
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        width = int((width * scale) // 8) * 8
        height = int((height * scale) // 8) * 8
        logger.warning(f"Resolution scaled down to {width}x{height} to prevent memory issues")
    
    return width, height

def load_pipeline_safe(pipeline_type: str, model_name: str, **kwargs) -> Any:
    """Thread-safe pipeline loading with proper error handling"""
    cache_key = f"{pipeline_type}_{model_name}"
    
    # Check cache first
    pipeline = PIPELINE_CACHE.get(cache_key)
    if pipeline is not None:
        logger.info(f"Using cached pipeline: {cache_key}")
        return pipeline
    
    logger.info(f"Loading new pipeline: {cache_key}")
    
    try:
        with cuda_memory_cleanup():
            if pipeline_type == "flux":
                from diffusers import FluxPipeline
                pipeline = FluxPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="balanced",
                    max_memory={0: "20GB", "cpu": "30GB"}
                )
                # Keep safety checker disabled for testing
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                
            elif pipeline_type == "wuerstchen":
                from diffusers import WuerstchenCombinedPipeline
                pipeline = WuerstchenCombinedPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                ).to(DEVICE)
                # Disable safety features
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                
            elif pipeline_type == "qwen":
                from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)
                pipeline = {"model": model, "tokenizer": tokenizer, "processor": processor}
                
            elif pipeline_type == "video":
                from diffusers import CogVideoXPipeline
                pipeline = CogVideoXPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16
                ).to(DEVICE)
                # Disable safety for uncensored generation
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                    
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")
            
            # Cache the loaded pipeline
            PIPELINE_CACHE.set(cache_key, pipeline)
            logger.info(f"Successfully loaded and cached: {cache_key}")
            return pipeline
            
    except Exception as e:
        logger.error(f"Failed to load pipeline {cache_key}: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Pipeline loading failed: {str(e)}")

def generate_image_safe(
    pipeline_type: str,
    model_name: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    num_images: int = 1,
    seed: Optional[int] = None
) -> List[Image.Image]:
    """Safe image generation with proper error handling and validation"""
    
    # Validate inputs
    width, height = validate_resolution(width, height)
    num_images = max(1, min(4, num_images))  # Limit to prevent memory issues
    num_inference_steps = max(1, min(100, num_inference_steps))
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    try:
        pipeline = load_pipeline_safe(pipeline_type, model_name)
        
        # Set up generation parameters
        generator = torch.Generator(device=DEVICE)
        if seed is not None:
            generator.manual_seed(seed)
        
        with cuda_memory_cleanup():
            if pipeline_type == "flux":
                images = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    max_sequence_length=512
                ).images
                
            elif pipeline_type == "wuerstchen":
                images = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator
                ).images
                
            else:
                raise ValueError(f"Image generation not supported for: {pipeline_type}")
        
        logger.info(f"Successfully generated {len(images)} images")
        return images
        
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Image generation failed: {str(e)}")

def generate_video_safe(
    model_name: str,
    prompt: str,
    width: int = 720,
    height: int = 480,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None
) -> str:
    """Safe video generation with proper validation"""
    
    # Validate video resolution
    if (width, height) not in VIDEO_RESOLUTION_PRESETS.values():
        logger.warning(f"Invalid video resolution {width}x{height}, using 720x480")
        width, height = 854, 480
    
    # Validate frame count (CogVideoX specific)
    valid_frames = [49, 81]  # CogVideoX supported frame counts
    if num_frames not in valid_frames:
        num_frames = 49
        logger.warning(f"Invalid frame count, using {num_frames}")
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    try:
        pipeline = load_pipeline_safe("video", model_name)
        
        # Set up generation
        generator = torch.Generator(device=DEVICE)
        if seed is not None:
            generator.manual_seed(seed)
        
        with cuda_memory_cleanup():
            video_frames = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).frames[0]
        
        # Save video
        output_path = f"/tmp/generated_video_{int(time.time())}.mp4"
        
        # Convert frames to video using imageio
        import imageio
        with imageio.get_writer(output_path, fps=8) as writer:
            for frame in video_frames:
                writer.append_data(np.array(frame))
        
        logger.info(f"Video saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Video generation failed: {str(e)}")

def analyze_image_safe(model_name: str, image: Image.Image, prompt: str) -> str:
    """Safe image analysis with proper error handling"""
    
    if image is None:
        raise ValueError("No image provided")
    
    if not prompt or not prompt.strip():
        prompt = "Describe this image in detail."
    
    try:
        pipeline_components = load_pipeline_safe("qwen", model_name)
        model = pipeline_components["model"]
        tokenizer = pipeline_components["tokenizer"]
        processor = pipeline_components["processor"]
        
        # Prepare the conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = processor.process(
            text=[text], images=[image], videos=None, padding=True, return_tensors="pt"
        )
        
        # Move to device
        image_inputs = image_inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **image_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(image_inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info("Image analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Image analysis failed: {str(e)}")

# Gradio interface functions
def gradio_generate_image(
    pipeline_type: str,
    model_name: str,
    prompt: str,
    resolution: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_images: int,
    seed: Optional[int]
) -> Tuple[List[Image.Image], str]:
    """Gradio wrapper for image generation"""
    
    if not prompt.strip():
        return [], "Error: Prompt cannot be empty"
    
    try:
        # Parse resolution
        if resolution in RESOLUTION_PRESETS:
            width, height = RESOLUTION_PRESETS[resolution]
        else:
            width, height = 1024, 1024  # Default
        
        # Submit task to executor to prevent blocking
        future = TASK_EXECUTOR.submit(
            generate_image_safe,
            pipeline_type, model_name, prompt, width, height,
            num_inference_steps, guidance_scale, num_images, seed
        )
        
        # Wait for completion with timeout
        images = future.result(timeout=TASK_TIMEOUT)
        
        return images, f"Successfully generated {len(images)} images"
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        logger.error(error_msg)
        return [], error_msg

def gradio_generate_video(
    model_name: str,
    prompt: str,
    resolution: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int]
) -> Tuple[Optional[str], str]:
    """Gradio wrapper for video generation"""
    
    if not prompt.strip():
        return None, "Error: Prompt cannot be empty"
    
    try:
        # Parse resolution
        if resolution in VIDEO_RESOLUTION_PRESETS:
            width, height = VIDEO_RESOLUTION_PRESETS[resolution]
        else:
            width, height = 854, 480  # Default
        
        # Submit task to executor
        future = TASK_EXECUTOR.submit(
            generate_video_safe,
            model_name, prompt, width, height,
            num_frames, num_inference_steps, guidance_scale, seed
        )
        
        # Wait for completion with timeout
        video_path = future.result(timeout=TASK_TIMEOUT * 2)  # Videos take longer
        
        return video_path, f"Video generated successfully"
        
    except Exception as e:
        error_msg = f"Video generation failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def gradio_analyze_image(
    model_name: str,
    image: Image.Image,
    prompt: str
) -> Tuple[str, str]:
    """Gradio wrapper for image analysis"""
    
    if image is None:
        return "", "Error: Please upload an image"
    
    try:
        # Submit task to executor
        future = TASK_EXECUTOR.submit(analyze_image_safe, model_name, image, prompt)
        
        # Wait for completion
        analysis = future.result(timeout=TASK_TIMEOUT)
        
        return analysis, "Analysis completed successfully"
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def create_gradio_interface() -> gr.Blocks:
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="RunPod AI Kit - Fixed Version",
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}"
    ) as interface:
        
        gr.Markdown("# 🚀 RunPod AI Kit - Production Ready")
        gr.Markdown("*Fixed version with proper async handling, thread safety, and error management*")
        
        with gr.Tabs():
            # Image Generation Tab
            with gr.Tab("🎨 Image Generation"):
                with gr.Row():
                    with gr.Column():
                        img_pipeline = gr.Dropdown(
                            choices=["flux", "wuerstchen"],
                            value="flux",
                            label="Pipeline Type"
                        )
                        img_model = gr.Textbox(
                            value="black-forest-labs/FLUX.1-dev",
                            label="Model Name"
                        )
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful landscape...",
                            lines=3
                        )
                        img_resolution = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value="Square (1024x1024)",
                            label="Resolution"
                        )
                        
                        with gr.Row():
                            img_steps = gr.Slider(1, 100, value=20, label="Steps")
                            img_guidance = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
                        
                        with gr.Row():
                            img_count = gr.Slider(1, 4, value=1, step=1, label="Images")
                            img_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        img_generate_btn = gr.Button("🎨 Generate Images", variant="primary")
                    
                    with gr.Column():
                        img_gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                        img_status = gr.Textbox(label="Status", interactive=False)
                
                img_generate_btn.click(
                    fn=gradio_generate_image,
                    inputs=[
                        img_pipeline, img_model, img_prompt, img_resolution,
                        img_steps, img_guidance, img_count, img_seed
                    ],
                    outputs=[img_gallery, img_status]
                )
            
            # Video Generation Tab
            with gr.Tab("🎬 Video Generation"):
                with gr.Row():
                    with gr.Column():
                        vid_model = gr.Textbox(
                            value="THUDM/CogVideoX-5b",
                            label="Model Name"
                        )
                        vid_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A person walking in the forest...",
                            lines=3
                        )
                        vid_resolution = gr.Dropdown(
                            choices=list(VIDEO_RESOLUTION_PRESETS.keys()),
                            value="480p (854x480)",
                            label="Resolution"
                        )
                        
                        with gr.Row():
                            vid_frames = gr.Dropdown(
                                choices=[49, 81],
                                value=49,
                                label="Frames"
                            )
                            vid_steps = gr.Slider(10, 100, value=50, label="Steps")
                        
                        with gr.Row():
                            vid_guidance = gr.Slider(1, 20, value=6.0, label="Guidance Scale")
                            vid_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        vid_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="Generated Video")
                        vid_status = gr.Textbox(label="Status", interactive=False)
                
                vid_generate_btn.click(
                    fn=gradio_generate_video,
                    inputs=[
                        vid_model, vid_prompt, vid_resolution,
                        vid_frames, vid_steps, vid_guidance, vid_seed
                    ],
                    outputs=[vid_output, vid_status]
                )
            
            # Image Analysis Tab
            with gr.Tab("🔍 Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        ana_model = gr.Textbox(
                            value="Qwen/Qwen2-VL-7B-Instruct",
                            label="Model Name"
                        )
                        ana_image = gr.Image(
                            type="pil",
                            label="Upload Image"
                        )
                        ana_prompt = gr.Textbox(
                            value="Describe this image in detail.",
                            label="Analysis Prompt",
                            lines=2
                        )
                        ana_analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                    
                    with gr.Column():
                        ana_output = gr.Textbox(
                            label="Analysis Result",
                            lines=15,
                            interactive=False
                        )
                        ana_status = gr.Textbox(label="Status", interactive=False)
                
                ana_analyze_btn.click(
                    fn=gradio_analyze_image,
                    inputs=[ana_model, ana_image, ana_prompt],
                    outputs=[ana_output, ana_status]
                )
            
            # System Info Tab
            with gr.Tab("📊 System Info"):
                with gr.Column():
                    gr.Markdown("### System Information")
                    
                    system_info = f"""
                    - **Device**: {DEVICE}
                    - **CUDA Available**: {torch.cuda.is_available()}
                    - **Cache Size**: {PIPELINE_CACHE._max_size}
                    - **Task Timeout**: {TASK_TIMEOUT}s
                    - **Safety Features**: Disabled (Testing Mode)
                    """
                    
                    gr.Markdown(system_info)
                    
                    if torch.cuda.is_available():
                        gpu_info = f"""
                        ### GPU Information
                        - **GPU Count**: {torch.cuda.device_count()}
                        - **Current Device**: {torch.cuda.current_device()}
                        - **Device Name**: {torch.cuda.get_device_name(0)}
                        """
                        gr.Markdown(gpu_info)
                    
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                    cache_status = gr.Textbox(label="Cache Status", interactive=False)
                    
                    def clear_pipeline_cache():
                        try:
                            PIPELINE_CACHE.clear()
                            return "Cache cleared successfully"
                        except Exception as e:
                            return f"Error clearing cache: {str(e)}"
                    
                    clear_cache_btn.click(
                        fn=clear_pipeline_cache,
                        outputs=[cache_status]
                    )
        
        gr.Markdown("---")
        gr.Markdown("*RunPod AI Kit v2.0 - Fixed Production Version*")
    
    return interface

def main():
    """Main application entry point"""
    logger.info("Starting RunPod AI Kit - Fixed Version")
    
    try:
        # Create and launch interface
        interface = create_gradio_interface()
        
        # Launch with proper configuration
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            enable_queue=True,
            max_threads=10,
            auth=None,
            inbrowser=False
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup resources
        PIPELINE_CACHE.clear()
        TASK_EXECUTOR.shutdown(wait=True)
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()
