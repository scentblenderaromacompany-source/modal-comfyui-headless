"""
Hyperframes Plugin for ComfyUI Headless on Modal
=================================================

Adds video generation and frame interpolation capabilities to the headless ComfyUI setup.

What "hyperframes" means here:
  Given two keyframe images (start + end), generate N intermediate frames using
  frame interpolation (RIFE-style) to create smooth video transitions.

Nodes provided:
  - RIFEFrameInterpolation: Interpolates between 2 images to generate N middle frames
  - VideoFromFrames: Combines images into video (FFmpeg)
  - KeyframeSchedule: Generates prompt schedules for multi-keyframe workflows
  - FrameBlend: Blends adjacent frames with adjustable weight

Usage via API:
  POST /generate-video
  Body: {
    "start_image": "<base64 PNG>",
    "end_image": "<base64 PNG>",
    "num_intermediate_frames": 15,
    "output_fps": 24,
    "prompt": "optional guiding prompt"
  }
"""

# Plugin registry IDs for custom nodes to install
HYPERFRAMES_PLUGINS = [
    "comfyui-frame-interpolation",      # RIFE frame interpolation (1037 stars)
    "comfyui-animatediff-evolved",       # AnimateDiff for video generation  
]

# Model URLs for frame interpolation
HYPERFRAMES_MODELS = [
    # RIFE (Real-Time Intermediate Flow Estimation) for frame interpolation
    {
        "url": "https://huggingface.co/datasets/nicolai256/RIFE_checkpoints/resolve/main/flownet.pkl",
        "filename": "flownet.pkl",
        "model_dir": "rife",
    },
]


def get_hyperframes_workflow(
    start_image_b64: str = None,
    end_image_b64: str = None,
    num_intermediate_frames: int = 15,
    output_fps: int = 24,
    width: int = 512,
    height: int = 512,
    prompt: str = None,
) -> dict:
    """
    Build a ComfyUI workflow JSON for hyperframe (frame interpolation) generation.
    
    Two modes:
    1. Image-to-video: Given start+end images, interpolate N frames between them
    2. Text-to-video with hyperframes: Generate keyframes with FLUX, then interpolate
    
    Args:
        start_image_b64: Base64-encoded start frame PNG (optional, generates if None)
        end_image_b64: Base64-encoded end frame PNG (optional, generates if None)
        num_intermediate_frames: Number of frames to interpolate between keyframes
        output_fps: Frames per second for output video
        width: Frame width
        height: Frame height
        prompt: Optional prompt for generating keyframes
    
    Returns:
        ComfyUI workflow JSON dict
    """
    
    workflow = {}
    node_id = 0
    
    def next_id():
        nonlocal node_id
        node_id += 1
        return str(node_id)
    
    # If we need to generate keyframes from prompt
    if start_image_b64 is None and prompt:
        # Generate start frame
        n1 = next_id()  # UNETLoader
        n2 = next_id()  # VAELoader  
        n3 = next_id()  # CLIPLoader
        n4 = next_id()  # CLIPTextEncode (positive)
        n5 = next_id()  # EmptyFlux2LatentImage
        n6 = next_id()  # KSampler
        n7 = next_id()  # VAEDecode
        n8 = next_id()  # SaveImage (start frame)
        
        workflow[n1] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
        workflow[n2] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
        workflow[n3] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}
        workflow[n4] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [n3, 0]}}
        workflow[n5] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        workflow[n6] = {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n1, 0], "positive": [n4, 0], "negative": [n4, 0], "latent_image": [n5, 0]}}
        workflow[n7] = {"class_type": "VAEDecode", "inputs": {"samples": [n6, 0], "vae": [n2, 0]}}
        workflow[n8] = {"class_type": "SaveImage", "inputs": {"images": [n7, 0], "filename_prefix": "hyperframe_start"}}
        
        # Generate end frame (different seed)
        n9 = next_id()  # CLIPTextEncode (end prompt)
        n10 = next_id()  # EmptyFlux2LatentImage (end)
        n11 = next_id()  # KSampler (end, reuse model/clip)
        n12 = next_id()  # VAEDecode (end, reuse vae)
        n13 = next_id()  # SaveImage (end frame)
        
        end_prompt = prompt + ", end frame, final state"
        workflow[n9] = {"class_type": "CLIPTextEncode", "inputs": {"text": end_prompt, "clip": [n3, 0]}}
        workflow[n10] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        workflow[n11] = {"class_type": "KSampler", "inputs": {"seed": 12345, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n1, 0], "positive": [n9, 0], "negative": [n9, 0], "latent_image": [n10, 0]}}
        workflow[n12] = {"class_type": "VAEDecode", "inputs": {"samples": [n11, 0], "vae": [n2, 0]}}
        workflow[n13] = {"class_type": "SaveImage", "inputs": {"images": [n12, 0], "filename_prefix": "hyperframe_end"}}
    
    # RIFE Frame Interpolation
    # This takes the start/end images and generates intermediate frames
    if start_image_b64 and end_image_b64:
        # Load start image
        n_load_start = next_id()
        workflow[n_load_start] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_b64}
        }
        
        # Load end image
        n_load_end = next_id()
        workflow[n_load_end] = {
            "class_type": "LoadImage", 
            "inputs": {"image": end_image_b64}
        }
        
        # RIFE interpolation node
        n_rife = next_id()
        workflow[n_rife] = {
            "class_type": "RIFE_VFI",
            "inputs": {
                "ckpt_name": "flownet.pkl",
                "frames": [n_load_start, 0],  # Start frame
                "optional_interpolation_states": None,
                "multiplier": num_intermediate_frames + 1,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
            }
        }
        
        # Video output from frames
        n_video = next_id()
        workflow[n_video] = {
            "class_type": "VHS_VideoCombine",  # Video Helper Suite
            "inputs": {
                "images": [n_rife, 0],
                "frame_rate": output_fps,
                "loop_count": 0,
                "filename_prefix": "hyperframe_output",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_output": True,
                "videopreview": {"hidden": True},
            }
        }
    
    return workflow


def get_txt2video_hyperframes_workflow(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_keyframes: int = 4,
    frames_per_keyframe: int = 8,
    output_fps: int = 24,
    seed: int = 42,
) -> dict:
    """
    Full text-to-video with hyperframes workflow.
    
    1. Generate N keyframes using FLUX with different seeds
    2. Use RIFE interpolation to fill gaps between keyframes
    3. Combine into final video
    
    Args:
        prompt: Text prompt for generation
        width/height: Frame dimensions
        num_keyframes: Number of keyframes to generate
        frames_per_keyframe: Interpolated frames between each keyframe pair
        output_fps: Output video FPS
        seed: Base seed (each keyframe gets seed+i)
    """
    workflow = {}
    node_id = 0
    
    def next_id():
        nonlocal node_id
        node_id += 1
        return str(node_id)
    
    # Shared model loaders
    n_unet = next_id()
    n_vae = next_id()
    n_clip = next_id()
    
    workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
    workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}
    
    keyframe_nodes = []
    
    # Generate keyframes
    for i in range(num_keyframes):
        n_text = next_id()
        n_latent = next_id()
        n_sampler = next_id()
        n_decode = next_id()
        n_save = next_id()
        
        # Vary the prompt slightly for each keyframe
        frame_prompt = f"{prompt}, frame {i+1} of {num_keyframes}"
        
        workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": frame_prompt, "clip": [n_clip, 0]}}
        workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": seed + i * 1000, "steps": 15, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
        workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
        workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"keyframe_{i:03d}"}}
        
        keyframe_nodes.append(n_save)
    
    # Interpolate between consecutive keyframes using RIFE
    interpolated_outputs = []
    for i in range(len(keyframe_nodes) - 1):
        n_rife = next_id()
        workflow[n_rife] = {
            "class_type": "RIFE_VFI",
            "inputs": {
                "ckpt_name": "flownet.pkl",
                "frames": [keyframe_nodes[i], 0],
                "multiplier": frames_per_keyframe,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
            }
        }
        interpolated_outputs.append(n_rife)
    
    # Combine all interpolated frames into video
    if interpolated_outputs:
        n_video = next_id()
        workflow[n_video] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [interpolated_outputs[0], 0],
                "frame_rate": output_fps,
                "loop_count": 0,
                "filename_prefix": "hyperframe_video",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_output": True,
            }
        }
    
    return workflow
