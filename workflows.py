"""
Production Music Video Visualizer Workflows
============================================

Advanced workflows that chain multiple models, LoRAs, and post-processing
steps to generate high-quality music video content for YouTube rap/trap channels.

Workflow Architecture:
  Input (prompt/audio) → Keyframe Generation → Style Transfer → Frame Interpolation → Upscale → Encode

Each workflow is a ComfyUI graph with 10-50+ nodes.
"""

import json
import time


class NodeID:
    """Generates sequential node IDs for workflow JSON."""
    def __init__(self, start=1):
        self._counter = start - 1
    def __call__(self):
        self._counter += 1
        return str(self._counter)
    @property
    def current(self):
        return str(self._counter)


# ============================================================================
# WORKFLOW 1: Music Video Keyframe Generator
# ============================================================================

def build_music_video_keyframes(
    prompts: list[str],
    width: int = 1920,
    height: int = 1080,
    steps: int = 25,
    cfg: float = 7.5,
    base_seed: int = 42,
    loras: list = None,
    style_reference: str = None,
    controlnet_image: str = None,
    controlnet_type: str = "openpose",  # openpose, depth, canny
    controlnet_strength: float = 0.8,
) -> dict:
    """
    Generate a sequence of keyframe images for a music video.
    
    Pipeline per keyframe:
      1. Load transformer (FLUX.2 dev) + text encoder + VAE
      2. Apply LoRAs for style
      3. Encode prompt → CLIPTextEncode
      4. Optional: ControlNet preprocessing (pose/depth from reference)
      5. Optional: IP-Adapter style transfer from reference image
      6. KSampler with schedule (high denoise for first frame, lower for continuity)
      7. VAEDecode → SaveImage
    
    Args:
        prompts: List of prompts (one per keyframe). Each describes a scene.
        width/height: Output resolution (1920x1080 for YouTube)
        steps: Sampling steps per keyframe
        cfg: CFG scale
        base_seed: Starting seed (incremented per keyframe)
        loras: List of (filename, strength) tuples
        style_reference: Path to style reference image for IP-Adapter
        controlnet_image: Path to control image for ControlNet
        controlnet_type: Type of control (openpose, depth, canny)
        controlnet_strength: ControlNet conditioning strength
    
    Returns:
        ComfyUI workflow JSON dict
    """
    workflow = {}
    nid = NodeID()

    # ── Shared Model Loaders ──
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

    # ── LoRA Chain ──
    model_out = n_unet
    clip_out = n_clip
    if loras:
        for lora_name, lora_strength in loras:
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

    # ── IP-Adapter Style Transfer (optional) ──
    if style_reference:
        n_ipadapter = nid()
        n_clip_vision = nid()
        n_ip_encode = nid()
        n_apply_ip = nid()

        workflow[n_ipadapter] = {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": "ip-adapter-plus-face_sd15.bin"}
        }
        workflow[n_clip_vision] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "clip_vision_h.safetensors"}
        }
        workflow[n_ip_encode] = {
            "class_type": "IPAdapterEncoder",
            "inputs": {
                "clip_vision": [n_clip_vision, 0],
                "image": style_reference,
                "weight": 0.6,
            }
        }
        workflow[n_apply_ip] = {
            "class_type": "IPAdapterApply",
            "inputs": {
                "ipadapter": [n_ipadapter, 0],
                "clip_vision": [n_clip_vision, 0],
                "image": style_reference,
                "model": [model_out, 0],
                "weight": 0.6,
                "weight_type": "original",
                "start_at": 0.0,
                "end_at": 1.0,
                "insightface": None,
            }
        }
        # Update model output to go through IP-Adapter
        model_out = n_apply_ip

    # ── ControlNet (optional) ──
    control_out = None
    if controlnet_image:
        n_controlnet = nid()
        n_preprocessor = nid()
        n_apply_control = nid()

        control_model_map = {
            "openpose": "control_v11p_sd15_openpose.pth",
            "depth": "control_v11f1p_sd15_depth.pth",
            "canny": "control_v11p_sd15_canny.pth",
        }
        preprocessor_map = {
            "openpose": "OpenPosePreprocessor",
            "depth": "DepthPreprocessor",
            "canny": "CannyPreprocessor",
        }

        workflow[n_controlnet] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": control_model_map.get(controlnet_type, "control_v11p_sd15_openpose.pth")}
        }
        workflow[n_preprocessor] = {
            "class_type": preprocessor_map.get(controlnet_type, "OpenPosePreprocessor"),
            "inputs": {"image": controlnet_image, "resolution": 512}
        }
        workflow[n_apply_control] = {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": [clip_out, 0],
                "control_net": [n_controlnet, 0],
                "image": [n_preprocessor, 0],
                "strength": controlnet_strength,
            }
        }
        control_out = n_apply_control

    # ── Generate Keyframes ──
    keyframe_saves = []
    for i, prompt in enumerate(prompts):
        n_text_pos = nid()
        n_text_neg = nid()
        n_latent = nid()
        n_sampler = nid()
        n_decode = nid()
        n_save = nid()

        # Positive prompt with timestamp info
        ts_prompt = f"{prompt}, cinematic, dark aesthetic, high contrast, 8k detail, professional music video"

        workflow[n_text_pos] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ts_prompt, "clip": [clip_out, 0]}
        }
        workflow[n_text_neg] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, distorted, washed out, ugly, deformed, bad anatomy, watermark, text, logo",
                "clip": [clip_out, 0]
            }
        }

        # Latent with slight noise variation for continuity
        n_seed = base_seed + i * 777  # Large step for variety but deterministic
        workflow[n_latent] = {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        }

        # Sampler — use ControlNet conditioning if available
        positive_in = control_out if control_out else n_text_pos
        workflow[n_sampler] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": n_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": [model_out, 0],
                "positive": [positive_in, 0] if control_out else [n_text_pos, 0],
                "negative": [n_text_neg, 0],
                "latent_image": [n_latent, 0],
            }
        }

        workflow[n_decode] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}
        }

        workflow[n_save] = {
            "class_type": "SaveImage",
            "inputs": {"images": [n_decode, 0], "filename_prefix": f"keyframe_{i:03d}"}
        }
        keyframe_saves.append(n_save)

    return workflow


# ============================================================================
# WORKFLOW 2: Frame Interpolation Pipeline
# ============================================================================

def build_interpolation_pipeline(
    keyframe_node_ids: list,
    frames_per_transition: int = 12,
    output_fps: int = 24,
    output_prefix: str = "interpolated",
) -> dict:
    """
    Given a sequence of keyframe SaveImage node IDs, build frame interpolation
    between each consecutive pair. Returns workflow nodes to add.
    """
    workflow_additions = {}
    nid = NodeID(start=1000)
    interpolated = []

    for i in range(len(keyframe_node_ids) - 1):
        n_model = nid()
        n_rife = nid()
        workflow_additions[n_model] = {
            "class_type": "FrameInterpolationModelLoader",
            "inputs": {"model_name": "flownet.pkl"}
        }
        workflow_additions[n_rife] = {
            "class_type": "FrameInterpolate",
            "inputs": {
                "interp_model": [n_model, 0],
                "images": [keyframe_node_ids[i], 0],
                "multiplier": frames_per_transition,
            }
        }
        interpolated.append(n_rife)

    if interpolated:
        n_save = nid()
        workflow_additions[n_save] = {
            "class_type": "SaveAnimatedPNG",
            "inputs": {
                "filename_prefix": output_prefix,
                "fps": output_fps,
                "images": [interpolated[0], 0],
            }
        }

    return workflow_additions


# ============================================================================
# WORKFLOW 3: Full Music Video Pipeline
# ============================================================================

def build_full_music_video(
    prompts: list[str],
    width: int = 1920,
    height: int = 1080,
    steps: int = 20,
    cfg: float = 7.5,
    base_seed: int = 42,
    loras: list = None,
    frames_per_transition: int = 12,
    output_fps: int = 24,
    upscale: bool = True,
    upscale_factor: float = 1.5,
    output_prefix: str = "music_video",
) -> dict:
    """
    Complete music video generation pipeline.
    
    Pipeline:
      1. Generate N keyframes with FLUX (different seeds, LoRAs)
      2. RIFE interpolate between consecutive keyframes
      3. Optional: Upscale all frames
      4. VHS combine into MP4 video
    
    Args:
        prompts: Scene descriptions for each keyframe
        width/height: Output resolution
        steps: Sampling steps per keyframe
        cfg: CFG scale
        base_seed: Starting seed
        loras: List of (filename, strength) tuples
        frames_per_transition: Interpolated frames between each keyframe pair
        output_fps: Video frame rate
        upscale: Whether to upscale output
        upscale_factor: Upscale multiplier
        output_prefix: Output filename prefix
    
    Returns:
        Complete ComfyUI workflow JSON dict
    """
    # Step 1: Generate keyframes
    workflow = build_music_video_keyframes(
        prompts=prompts,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        base_seed=base_seed,
        loras=loras,
    )

    # Find keyframe SaveImage nodes
    keyframe_saves = [
        nid for nid, node in workflow.items()
        if node.get("class_type") == "SaveImage" and "keyframe" in node.get("inputs", {}).get("filename_prefix", "")
    ]

    if len(keyframe_saves) >= 2:
        # Step 2: Add interpolation pipeline
        interp_nodes = build_interpolation_pipeline(
            keyframe_saves,
            frames_per_transition,
            output_fps,
            output_prefix,
        )
        workflow.update(interp_nodes)

    return workflow


# ============================================================================
# WORKFLOW 4: Audio-Reactive Visualizer
# ============================================================================

def build_audio_reactive_visualizer(
    audio_path: str,
    prompt: str,
    width: int = 1920,
    height: int = 1080,
    duration_seconds: int = 30,
    fps: int = 24,
    bpm: int = 140,
    base_seed: int = 42,
) -> dict:
    """
    Audio-reactive music video visualizer.
    
    Generates frames that sync to audio beats. Uses BPM to time keyframe
    transitions, then interpolates between them.
    
    Pipeline:
      1. Analyze audio BPM (external, passed as parameter)
      2. Generate keyframes at beat intervals
      3. RIFE interpolate between beats
      4. VHS combine with audio overlay
    
    Args:
        audio_path: Path to audio file (MP3/WAV)
        prompt: Base visual prompt
        width/height: Output resolution
        duration_seconds: Video duration
        fps: Output frame rate
        bpm: Beats per minute (for sync timing)
        base_seed: Random seed
    
    Returns:
        ComfyUI workflow JSON dict
    """
    # Calculate keyframe timing based on BPM
    beat_interval = 60.0 / bpm  # seconds per beat
    total_beats = int(duration_seconds / beat_interval)
    keyframe_interval = max(1, total_beats // 8)  # ~8 keyframes total

    # Generate prompts for each keyframe (evolving scene)
    prompts = []
    for i in range(0, total_beats, keyframe_interval):
        beat_time = i * beat_interval
        # Evolve the prompt over time
        if i == 0:
            scene = f"{prompt}, opening scene, establishing shot, dark atmosphere"
        elif i < total_beats // 3:
            scene = f"{prompt}, building energy, neon lights emerging, urban landscape"
        elif i < 2 * total_beats // 3:
            scene = f"{prompt}, peak energy, intense colors, dynamic movement, cinematic"
        else:
            scene = f"{prompt}, closing scene, fade to dark, atmospheric"
        prompts.append(scene)

    # Build the full pipeline
    return build_full_music_video(
        prompts=prompts,
        width=width,
        height=height,
        steps=15,  # Faster for video
        cfg=7.0,
        base_seed=base_seed,
        frames_per_transition=int(beat_interval * fps),
        output_fps=fps,
        output_prefix="audio_reactive",
    )


# ============================================================================
# WORKFLOW 5: Style Transfer Chain
# ============================================================================

def build_style_transfer_chain(
    input_image_b64: str,
    style_reference_b64: str,
    prompt: str,
    width: int = 1920,
    height: int = 1080,
    steps: int = 20,
    cfg: float = 7.0,
    ipadapter_strength: float = 0.6,
    controlnet_strength: float = 0.4,
    base_seed: int = 42,
) -> dict:
    """
    Apply style transfer from a reference image to a generated image.
    
    Pipeline:
      1. Generate base image with FLUX
      2. IP-Adapter applies style from reference
      3. ControlNet preserves structure
      4. Blend original + styled output
    
    Useful for: applying album art style to generated frames,
    maintaining visual consistency across a music video.
    """
    workflow = {}
    nid = NodeID()

    # Model loaders
    n_unet = nid()
    n_vae = nid()
    n_clip = nid()
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

    # Text encoding
    n_text = nid()
    workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n_clip, 0]}}

    # Latent
    n_latent = nid()
    workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}

    # Base generation
    n_sampler = nid()
    workflow[n_sampler] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": base_seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0],
            "latent_image": [n_latent, 0],
        }
    }

    n_decode = nid()
    workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}

    # IP-Adapter style transfer
    n_ip_model = nid()
    n_clip_vision = nid()
    n_apply_ip = nid()

    workflow[n_ip_model] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "ip-adapter-plus-face_sd15.bin"}}
    workflow[n_clip_vision] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}}
    workflow[n_apply_ip] = {
        "class_type": "IPAdapterApply",
        "inputs": {
            "ipadapter": [n_ip_model, 0],
            "clip_vision": [n_clip_vision, 0],
            "image": style_reference_b64,
            "model": [n_unet, 0],
            "weight": ipadapter_strength,
            "weight_type": "original",
            "start_at": 0.0,
            "end_at": 1.0,
        }
    }

    # Re-sampler with IP-Adapter applied
    n_latent2 = nid()
    workflow[n_latent2] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}

    n_sampler2 = nid()
    workflow[n_sampler2] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": base_seed + 1, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 0.5,  # Lower denoise for style transfer
            "model": [n_apply_ip, 0], "positive": [n_text, 0], "negative": [n_text, 0],
            "latent_image": [n_latent2, 0],
        }
    }

    n_decode2 = nid()
    workflow[n_decode2] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler2, 0], "vae": [n_vae, 0]}}

    # Save styled output
    n_save = nid()
    workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode2, 0], "filename_prefix": "styled"}}

    return workflow


# ============================================================================
# WORKFLOW 6: Batch Album Art Generator
# ============================================================================

def build_batch_album_art(
    prompts: list[str],
    style_reference: str = None,
    width: int = 3000,
    height: int = 3000,
    steps: int = 30,
    cfg: float = 7.5,
    base_seed: int = 42,
) -> dict:
    """
    Generate multiple album cover art variations in one workflow run.
    
    Pipeline:
      For each prompt:
        1. Generate base image with FLUX
        2. Optional: Apply style transfer from reference
        3. Save as high-res PNG
    
    Args:
        prompts: List of album art concepts
        style_reference: Optional style reference image
        width/height: Output resolution (3000x3000 for print quality)
        steps: Sampling steps
        cfg: CFG scale
        base_seed: Starting seed
    
    Returns:
        ComfyUI workflow JSON dict
    """
    workflow = {}
    nid = NodeID()

    # Shared loaders
    n_unet = nid()
    n_vae = nid()
    n_clip = nid()
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

    for i, prompt in enumerate(prompts):
        n_text = nid()
        n_latent = nid()
        n_sampler = nid()
        n_decode = nid()
        n_save = nid()

        album_prompt = f"{prompt}, album cover art, square format, high detail, professional, dark aesthetic, cinematic lighting"

        workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": album_prompt, "clip": [n_clip, 0]}}
        workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        workflow[n_sampler] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": base_seed + i, "steps": steps, "cfg": cfg,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0],
                "latent_image": [n_latent, 0],
            }
        }
        workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
        workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"album_art_{i:03d}"}}

    return workflow
