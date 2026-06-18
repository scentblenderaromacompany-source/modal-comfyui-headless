"""
Production Channel Configurations for Content Generation
=========================================================

Each channel defines a complete generation pipeline:
  - Resolution, aspect ratio, FPS
  - Model selection (transformer, VAE, text encoder)
  - LoRA presets with strengths
  - Generation parameters (steps, CFG, sampler)
  - Post-processing (upscale, interpolate, encode)
  - Output format and quality settings
  - Prompt templates and negative prompts
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChannelConfig:
    """Configuration for a content generation channel."""
    name: str
    description: str
    width: int
    height: int
    fps: int = 24
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    transformer: str = "flux2_dev_fp8mixed.safetensors"
    text_encoder: str = "mistral_3_small_flux2_bf16.safetensors"
    vae: str = "flux2-vae.safetensors"
    loras: list = field(default_factory=list)  # [(name, strength), ...]
    lora_strength: float = 0.8
    num_keyframes: int = 4
    frames_per_keyframe: int = 8
    upscale: bool = False
    upscale_factor: float = 1.5
    denoise: float = 1.0
    output_format: str = "png"  # png, mp4, webm
    crf: int = 18
    seed: int = -1
    negative_prompt: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""


# ============================================================================
# YOUTUBE RAP/TRAP MUSIC VIDEO CHANNEL
# ============================================================================

CHANNELS = {
    "youtube-trap": ChannelConfig(
        name="youtube-trap",
        description="YouTube rap/trap music video visualizer — cinematic, dark, high contrast",
        width=1920,
        height=1080,
        fps=24,
        steps=25,
        cfg=7.5,
        sampler="euler",
        scheduler="normal",
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[
            # Add LoRAs as they become available in the volume
            # ("dever_arcane_flux2_klein_9b.safetensors", 0.6),
            # ("dever_clothes_line_flux2_klein_9b.safetensors", 0.4),
        ],
        lora_strength=0.75,
        num_keyframes=6,
        frames_per_keyframe=12,
        upscale=False,
        denoise=1.0,
        output_format="mp4",
        crf=18,
        negative_prompt="blurry, low quality, distorted, washed out, ugly, deformed, bad anatomy, watermark, text, logo, bright, cheerful, happy, soft, pastel",
        prompt_prefix="cinematic, dark aesthetic, high contrast, professional music video, 8k detail, ",
        prompt_suffix=", dramatic lighting, volumetric fog, ray tracing, film grain, anamorphic lens",
    ),

    "youtube-lofi": ChannelConfig(
        name="youtube-lofi",
        description="Lo-fi hip hop visualizer — warm, ambient, anime-inspired",
        width=1920,
        height=1080,
        fps=24,
        steps=20,
        cfg=7.0,
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],
        lora_strength=0.6,
        num_keyframes=4,
        frames_per_keyframe=16,
        upscale=False,
        output_format="mp4",
        crf=20,
        negative_prompt="blurry, low quality, distorted, harsh, dark, violent, text, watermark",
        prompt_prefix="lo-fi aesthetic, warm tones, cozy, ambient, anime style, ",
        prompt_suffix=", soft lighting, film grain, nostalgic, dreamy, bokeh",
    ),

    "tiktok": ChannelConfig(
        name="tiktok",
        description="TikTok/Reels vertical video — fast generation, trending aesthetics",
        width=1080,
        height=1920,
        fps=30,
        steps=15,
        cfg=7.0,
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],
        lora_strength=0.7,
        num_keyframes=3,
        frames_per_keyframe=10,
        upscale=False,
        output_format="mp4",
        crf=22,
        negative_prompt="blurry, low quality, distorted, text, watermark, boring, static",
        prompt_prefix="trending, viral aesthetic, eye-catching, dynamic, ",
        prompt_suffix=", vibrant colors, smooth motion, cinematic transition",
    ),

    "instagram": ChannelConfig(
        name="instagram",
        description="Instagram post — square format, high quality",
        width=1080,
        height=1080,
        fps=24,
        steps=25,
        cfg=7.0,
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],
        lora_strength=0.8,
        num_keyframes=4,
        frames_per_keyframe=8,
        upscale=False,
        output_format="mp4",
        crf=20,
        negative_prompt="blurry, low quality, distorted, text, watermark",
        prompt_prefix="instagram-worthy, aesthetic, curated, professional, ",
        prompt_suffix=", clean composition, balanced colors, high detail",
    ),

    "album-art": ChannelConfig(
        name="album-art",
        description="Album cover art — square, high resolution, detailed",
        width=3000,
        height=3000,
        fps=1,
        steps=30,
        cfg=7.5,
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],
        lora_strength=0.9,
        num_keyframes=1,
        frames_per_keyframe=1,
        upscale=False,
        output_format="png",
        crf=18,
        negative_prompt="blurry, low quality, distorted, text, watermark, low resolution, pixelated",
        prompt_prefix="album cover art, professional, high detail, print quality, ",
        prompt_suffix=", centered composition, bold typography space, dramatic lighting, 8k",
    ),

    "twitter": ChannelConfig(
        name="twitter",
        description="Twitter/X media — compressed, fast",
        width=1280,
        height=720,
        fps=24,
        steps=15,
        cfg=7.0,
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],
        lora_strength=0.7,
        num_keyframes=3,
        frames_per_keyframe=6,
        upscale=False,
        output_format="mp4",
        crf=24,
        negative_prompt="blurry, low quality, distorted, text, watermark",
        prompt_prefix="social media, eye-catching, bold, ",
        prompt_suffix=", high contrast, compressed, web-optimized",
    ),
}


def get_channel(name: str) -> ChannelConfig:
    """Get channel config by name."""
    if name not in CHANNELS:
        available = ", ".join(CHANNELS.keys())
        raise ValueError(f"Unknown channel '{name}'. Available: {available}")
    return CHANNELS[name]


def list_channels() -> dict:
    """List all available channels with descriptions."""
    return {name: cfg.description for name, cfg in CHANNELS.items()}


def build_prompt(channel: str, user_prompt: str) -> str:
    """Build a full prompt from channel template + user prompt."""
    cfg = get_channel(channel)
    return f"{cfg.prompt_prefix}{user_prompt}{cfg.prompt_suffix}"
