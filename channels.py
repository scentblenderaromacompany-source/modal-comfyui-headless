"""
Production Workflow Engine for Content Generation
==================================================

Channel-based workflow system for automated video/image generation.

Channels:
  youtube-trap     — YouTube rap/trap music videos (16:9, 1080p/4K, high quality)
  youtube-lofi     — Lo-fi hip hop visualizers (16:9, ambient)
  tiktok           — TikTok/Reels vertical (9:16, 1080×1920, fast generation)
  instagram        — Instagram posts/stories (1:1 or 4:5)
  twitter          — Twitter/X header/media (16:9, compressed)
  album-art        — Album cover art (1:1, 3000×3000)

Each channel defines:
  - Resolution, aspect ratio, FPS
  - Model selection (transformer, VAE, text encoder)
  - LoRA presets
  - Generation parameters (steps, CFG, sampler)
  - Post-processing (upscale, interpolate, encode)
  - Output format and quality settings
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
    loras: list = field(default_factory=list)
    lora_strength: float = 0.8
    num_keyframes: int = 4
    frames_per_keyframe: int = 8
    upscale: bool = False
    upscale_factor: float = 1.5
    denoise: float = 1.0
    output_format: str = "png"  # png, mp4, webm
    crf: int = 18  # video quality (lower = better)
    seed: int = -1  # -1 = random


# ============================================================================
# CHANNEL DEFINITIONS
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
        transformer="flux2_dev_fp8mixed.safetensors",
        text_encoder="mistral_3_small_flux2_bf16.safetensors",
        vae="flux2-vae.safetensors",
        loras=[],  # Add trap-specific LoRAs here
        lora_strength=0.75,
        num_keyframes=6,
        frames_per_keyframe=12,
        upscale=False,
        output_format="mp4",
        crf=18,
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
