"""
Model definitions for Modal ComfyUI — Music Video Production

Models are downloaded at build time to /cache/models/ and symlinked into ComfyUI.
The music-video-models Modal volume already contains most of these — this file
defines additional models to download from HuggingFace or direct URLs.

Channel: youtube-trap (rap/trap music video visualizer)
  - Dark, cinematic, high-contrast aesthetic
  - Neon lights, urban environments, luxury cars, street scenes
  - Fast-paced cuts with smooth interpolation
  - 1920×1080 output, 24fps

Models needed:
  - FLUX.2 dev (already in volume) — main transformer
  - Mistral text encoder (already in volume) — matches FLUX.2
  - RIFE frame interpolation (already in volume) — smooth transitions
  - IP-Adapter (for style transfer from album art / reference images)
  - ControlNet OpenPose (for pose-guided generation)
  - ControlNet Depth (for depth-guided generation)
  - 4x Upscale model (for 1080p output)
"""

models = [
    # IP-Adapter for FLUX — style transfer from reference images
    # Allows us to feed album art or artist photos as style references
    {
        "repo_id": "h94/IP-Adapter",
        "filename": "ip-adapter-plus-face_sd15.bin",
        "model_dir": "ipadapter",
    },

    # ControlNet models for guided generation
    {
        "repo_id": "lllyasviel/ControlNet-v1-1",
        "filename": "control_v11p_sd15_openpose.pth",
        "model_dir": "controlnet",
    },
    {
        "repo_id": "lllyasviel/ControlNet-v1-1",
        "filename": "control_v11f1p_sd15_depth.pth",
        "model_dir": "controlnet",
    },

    # 4x Upscale model for high-res output
    {
        "repo_id": "Kim2091/ClearRealityV1",
        "filename": "4x-ClearRealityV1.pth",
        "model_dir": "upscale_models",
    },

    # AnimateDiff motion model (for video generation)
    # {
    #     "repo_id": "guoyww/animatediff",
    #     "filename": "mm_sd_v15_v2.ckpt",
    #     "model_dir": "animatediff_models",
    # },
]

models_ext = [
    # Trap/rap aesthetic LoRAs from Civitai
    # These define the visual style for our music video channel

    # Dark cinematic LoRA
    # {
    #     "url": "https://civitai.com/api/download/models/XXXXX",
    #     "filename": "dark-cinematic-trap.safetensors",
    #     "model_dir": "loras",
    # },

    # Neon/cyberpunk aesthetic LoRA
    # {
    #     "url": "https://civitai.com/api/download/models/XXXXX",
    #     "filename": "neon-cyberpunk-style.safetensors",
    #     "model_dir": "loras",
    # },

    # Music video visualizer LoRA
    # {
    #     "url": "https://civitai.com/api/download/models/XXXXX",
    #     "filename": "music-visualizer-style.safetensors",
    #     "model_dir": "loras",
    # },
]
