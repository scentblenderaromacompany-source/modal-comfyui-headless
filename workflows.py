"""
Advanced Multi-Node Workflow Engine
====================================

Production-grade workflows that chain multiple models, LoRAs, and post-processing
steps to generate high-quality content.

Workflows:
  1. txt2img-flux          — Single image generation with FLUX
  2. txt2video-hyperframes — Keyframe generation + RIFE interpolation
  3. img2img-refine        — Image refinement with img2img + upscale
  4. style-transfer        — Apply style reference via IP-Adapter
  5. music-visualizer      — Audio-reactive video generation
  6. batch-generate        — Batch process multiple prompts
"""

from dataclasses import dataclass
import json
import time
import uuid


@dataclass
class GenerationRequest:
    """A single generation request."""
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, washed out, ugly, deformed"
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    seed: int = -1
    loras: list = None  # [(name, strength), ...]
    workflow: str = "txt2img-flux"  # Which workflow to use

    def __post_init__(self):
        if self.seed == -1:
            self.seed = int(time.time()) % 2**32
        if self.loras is None:
            self.loras = []


# ============================================================================
# WORKFLOW BUILDERS
# ============================================================================

def build_txt2img_flux(req: GenerationRequest) -> dict:
    """
    Single image generation workflow using FLUX.2 dev.
    
    Pipeline:
      UNETLoader → VAELoader → CLIPLoader → CLIPTextEncode → EmptyLatent → KSampler → VAEDecode → SaveImage
    """
    workflow = {}
    nid = _NodeID()

    # Model loaders
    n_unet = nid()
    n_vae = nid()
    n_clip = nid()

    workflow[n_unet] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}
    }
    workflow[n_vae] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "flux2-vae.safetensors"}
    }
    workflow[n_clip] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}
    }

    # Optional LoRAs
    model_out = n_unet
    clip_out = n_clip
    if req.loras:
        for lora_name, lora_strength in req.loras:
            n_lora = nid()
            workflow[n_lora] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": [model_out, 0],
                    "clip": [clip_out, 0],
                }
            }
            model_out = n_lora
            clip_out = n_lora

    # Text encoding
    n_text = nid()
    workflow[n_text] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": req.prompt, "clip": [clip_out, 0]}
    }

    # Latent
    n_latent = nid()
    workflow[n_latent] = {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": req.width, "height": req.height, "batch_size": 1}
    }

    # Sampler
    n_sampler = nid()
    workflow[n_sampler] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": req.seed,
            "steps": req.steps,
            "cfg": req.cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": req.denoise if hasattr(req, 'denoise') else 1.0,
            "model": [model_out, 0],
            "positive": [n_text, 0],
            "negative": [n_text, 0],
            "latent_image": [n_latent, 0],
        }
    }

    # Decode & save
    n_decode = nid()
    workflow[n_decode] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}
    }
    n_save = nid()
    workflow[n_save] = {
        "class_type": "SaveImage",
        "inputs": {"images": [n_decode, 0], "filename_prefix": "flux"},
    }

    return workflow


def build_txt2video_hyperframes(req: GenerationRequest, num_keyframes: int = 4,
                                 frames_per_keyframe: int = 8, output_fps: int = 24) -> dict:
    """
    Text-to-video with hyperframes interpolation.
    
    Pipeline:
      1. Generate N keyframes with FLUX (different seeds)
      2. RIFE interpolate between consecutive keyframes
      3. VHS combine into MP4
    """
    workflow = {}
    nid = _NodeID()

    # Shared model loaders
    n_unet = nid()
    n_vae = nid()
    n_clip = nid()
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

    # Generate keyframes
    keyframe_saves = []
    for i in range(num_keyframes):
        n_text = nid()
        n_latent = nid()
        n_sampler = nid()
        n_decode = nid()
        n_save = nid()

        frame_prompt = f"{req.prompt}, scene {i+1} of {num_keyframes}"
        workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": frame_prompt, "clip": [n_clip, 0]}}
        workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": req.width, "height": req.height, "batch_size": 1}}
        workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": req.seed + i * 1000, "steps": req.steps, "cfg": req.cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
        workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
        workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"keyframe_{i:03d}"}}
        keyframe_saves.append(n_save)

    # RIFE interpolation between keyframes
    interpolated = []
    for i in range(len(keyframe_saves) - 1):
        n_rife = nid()
        workflow[n_rife] = {
            "class_type": "RIFE_VFI",
            "inputs": {
                "ckpt_name": "flownet.pkl",
                "frames": [keyframe_saves[i], 0],
                "optional_interpolation_states": None,
                "multiplier": frames_per_keyframe,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
            }
        }
        interpolated.append(n_rife)

    # Video output
    if interpolated:
        n_video = nid()
        workflow[n_video] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [interpolated[0], 0],
                "frame_rate": output_fps,
                "loop_count": 0,
                "filename_prefix": "hyperframe",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_output": True,
                "videopreview": {"hidden": True},
            }
        }

    return workflow


def build_img2img_refine(req: GenerationRequest, strength: float = 0.5) -> dict:
    """
    Image refinement: img2img pass with lower denoise for detail enhancement.
    
    Pipeline:
      LoadImage → VAEDecode → KSampler (img2img) → VAEDecode → SaveImage
    """
    workflow = {}
    nid = _NodeID()

    n_load = nid()
    n_vae = nid()
    n_clip = nid()
    n_unet = nid()

    workflow[n_load] = {"class_type": "LoadImage", "inputs": {"image": ""}}  # Filled at runtime
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

    # Encode image to latent
    n_encode = nid()
    workflow[n_encode] = {"class_type": "VAEEncode", "inputs": {"pixels": [n_load, 0], "vae": [n_vae, 0]}}

    # Text
    n_text = nid()
    workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": req.prompt, "clip": [n_clip, 0]}}

    # Sampler with lower denoise for refinement
    n_sampler = nid()
    workflow[n_sampler] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": req.seed,
            "steps": req.steps,
            "cfg": req.cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": strength,
            "model": [n_unet, 0],
            "positive": [n_text, 0],
            "negative": [n_text, 0],
            "latent_image": [n_encode, 0],
        }
    }

    n_decode = nid()
    workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}

    n_save = nid()
    workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": "refined"}}

    return workflow


def build_batch_workflow(prompts: list[str], width: int = 512, height: int = 512,
                         steps: int = 15, base_seed: int = 42) -> dict:
    """
    Batch generation: multiple images in one workflow run.
    
    Pipeline:
      [UNETLoader → VAELoader → CLIPLoader] × 1
      [CLIPTextEncode → EmptyLatent → KSampler → VAEDecode → SaveImage] × N
    """
    workflow = {}
    nid = _NodeID()

    # Shared loaders
    n_unet = nid()
    n_vae = nid()
    n_clip = nid()
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

    # Each prompt gets its own generation chain
    for i, prompt in enumerate(prompts):
        n_text = nid()
        n_latent = nid()
        n_sampler = nid()
        n_decode = nid()
        n_save = nid()

        workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n_clip, 0]}}
        workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": base_seed + i, "steps": steps, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
        workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
        workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"batch_{i:03d}"}}

    return workflow


class _NodeID:
    def __init__(self):
        self._counter = 0
    def __call__(self):
        self._counter += 1
        return str(self._counter)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def create_generation(channel: str, prompt: str, **kwargs) -> dict:
    """
    Create a generation workflow from a channel config and prompt.
    
    Args:
        channel: Channel name (youtube-trap, tiktok, etc.)
        prompt: Text prompt
        **kwargs: Override any channel settings
    
    Returns:
        ComfyUI workflow JSON dict
    """
    from channels import get_channel
    cfg = get_channel(channel)

    # Apply overrides
    width = kwargs.get("width", cfg.width)
    height = kwargs.get("height", cfg.height)
    steps = kwargs.get("steps", cfg.steps)
    cfg_val = kwargs.get("cfg", cfg.cfg)
    seed = kwargs.get("seed", -1)

    req = GenerationRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg_val,
        seed=seed,
        loras=cfg.loras,
    )

    # Select workflow based on channel output format
    if cfg.output_format == "mp4":
        return build_txt2video_hyperframes(req, cfg.num_keyframes, cfg.frames_per_keyframe, cfg.fps)
    else:
        return build_txt2img_flux(req)
