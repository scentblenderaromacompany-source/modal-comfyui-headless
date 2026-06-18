"""
Model definitions for Modal ComfyUI.

Copy this file to models.py and edit.

Two sections:
  - `models`: HuggingFace downloads (repo_id + filename)
  - `models_ext`: Direct URL downloads (e.g. Civitai)

`model_dir` controls where the model lands:
  - Relative paths resolve under /cache/models/ (e.g. "checkpoints" -> /cache/models/checkpoints)
  - Standard ComfyUI folders: checkpoints, diffusion_models, vae, loras,
    text_encoders, clip_vision, controlnet, upscale_models, embeddings
"""

models = [
    # HuggingFace downloads via huggingface_hub.
    # Uncomment and edit to add models:
    #
    # {
    #     "repo_id": "black-forest-labs/FLUX.1-dev",
    #     "filename": "flux1-dev.safetensors",
    #     "model_dir": "diffusion_models",
    # },
    # {
    #     "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    #     "filename": "sd_xl_base_1.0.safetensors",
    #     "model_dir": "checkpoints",
    # },
    # {
    #     "repo_id": "stabilityai/sdxl-vae",
    #     "filename": "sdxl_vae.safetensors",
    #     "model_dir": "vae",
    # },
]

models_ext = [
    # Direct URL downloads via aria2c.
    # Use for Civitai, Google Drive, or any direct link:
    #
    # {
    #     "url": "https://civitai.com/api/download/models/12345",
    #     "filename": "my_lora.safetensors",
    #     "model_dir": "loras",
    # },
]
