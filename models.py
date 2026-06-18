"""
Model definitions for Modal ComfyUI.

Two sections:
  - `models`: HuggingFace downloads (repo_id + filename)
  - `models_ext`: Direct URL downloads (e.g. Civitai)

`model_dir` controls where the model lands (relative to /cache/models/).
Standard ComfyUI folders: checkpoints, diffusion_models, vae, loras,
  text_encoders, clip_vision, controlnet, upscale_models, embeddings
"""

models = [
    # Stable Diffusion 1.5 (small, good for testing)
    {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "filename": "v1-5-pruned-emaonly.safetensors",
        "model_dir": "checkpoints",
    },
]

models_ext = [
    # Direct URL downloads via aria2c.
    # {
    #     "url": "https://civitai.com/api/download/models/12345",
    #     "filename": "my_lora.safetensors",
    #     "model_dir": "loras",
    # },
]
