"""
Custom node registry IDs for Modal ComfyUI — Music Video Production Pack

These are installed at build time via comfy-cli.
Find node IDs at: https://registry.comfy.org/

Categories:
  - Frame interpolation (RIFE, DAIN)
  - Video generation (AnimateDiff, VHS)
  - IP-Adapter (style transfer)
  - ControlNet (pose, depth, canny)
  - Upscaling (4x, 8x)
  - Audio processing (audio-reactive)
"""

comfy_plugins: list[str] = [
    # Frame Interpolation — generates intermediate frames for smooth video
    "comfyui-frame-interpolation",           # RIFE_VFI node (1037 stars)

    # Video Helper Suite — video output, frame loading, batch processing
    "comfyui-video-helper-suite",            # VHS_VideoCombine, VHS_LoadVideo

    # IP-Adapter — style transfer from reference images
    "comfyui-ipadapter-plus",                # IPAdapterApply, IPAdapterEncoder

    # ControlNet preprocessors — pose, depth, canny edge detection
    "comfyui-controlnet-preprocessors",      # OpenPose, Depth, Canny, etc.

    # AnimateDiff — video generation with motion models
    # "comfyui-animatediff-evolved",          # AnimateDiff loader (large, uncomment if needed)

    # Upscaling — high-quality image upscaling
    "comfyui-upscale-model-loader",          # UpscaleModelLoader
    "comfyui-impact-pack",                   # Various utility nodes including detailer

    # Additional utilities
    "comfyui-essentials",                    # ImageResize, ImageCrop, etc.
    "comfyui-kjnodes",                       # GetImageSize, ImageConcat, etc.
    "comfyui-advanced-latent-control",       # Advanced latent manipulation
]
