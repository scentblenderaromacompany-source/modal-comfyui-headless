# Content Generation Automation Skill

Production-grade headless ComfyUI system on Modal Labs for automated content generation.
Supports image and video generation with channel-specific presets for social media platforms.

## Architecture

```
Local machine                    Modal Cloud
┌──────────────────┐            ┌─────────────────────────────┐
│ generate.py      │──POST /gen──▶  ComfyUI headless (A10 GPU) │
│ trap_producer.py │◀──prompt_id──  │                             │
│                  │──GET /result─▶ │  music-video-models volume  │
│                  │◀──{images}───  │    (150+ GB models)        │
│                  │               │  Custom nodes: RIFE, VHS,   │
│                  │               │    IP-Adapter, ControlNet   │
└──────────────────┘            └─────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `comfyui_headless.py` | Main Modal app — deploy with `modal deploy comfyui_headless.py` |
| `generate.py` | Local CLI client — image + video generation |
| `trap_music_video_producer.py` | Production trap music video producer with 5 style presets |
| `workflows.py` | 8 workflow builders (txt2img, txt2video, keyframes, etc.) |
| `channels.py` | 6 channel presets (youtube-trap, youtube-lofi, tiktok, etc.) |
| `hyperframes.py` | Hyperframes video workflow builder |
| `models.py` | Model definitions (HF + direct URLs) |
| `plugins.py` | Custom node registry IDs |

## Deployment

```bash
cd /home/bobby/modal-comfyui-headless
modal deploy comfyui_headless.py
# App URL: https://robertmcasper--comfyui-headless-serve.modal.run
```

## Quick Start

### Generate an image
```bash
python generate.py "a beautiful sunset" --width 512 --height 512 --steps 15
```

### Generate video keyframes
```bash
python generate.py "trap music video" --video --width 512 --height 512 --keyframes 3
```

### Generate with channel preset
```bash
python generate.py "dark trap visual" --channel youtube-trap
python generate.py "tiktok trend" --channel tiktok
python generate.py "album cover" --channel album-art
```

### Trap music video producer
```bash
python trap_music_video_producer.py --style dark-trap --channel youtube --duration 180
python trap_music_video_producer.py --style neon-trap --channel tiktok --duration 60
python trap_music_video_producer.py --list-styles
python trap_music_video_producer.py --list-channels
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Submit image generation. Body: `{prompt, width, height, steps, cfg, seed, loras?}` |
| POST | `/generate-video` | Submit video. Body: `{prompt, width, height, num_keyframes, frames_per_keyframe, output_fps, steps, cfg, seed, loras?, channel?}` |
| GET | `/result/{id}` | Poll for results. Returns `{status: running\|done, images: [{filename, size_bytes, base64}]}` |
| GET | `/models` | List available models by category |
| GET | `/system_stats` | Health check (GPU, RAM, ComfyUI version) |
| GET | `/view?filename=X&subfolder=&type=output` | Download output file |
| * | All other paths | Transparent proxy to ComfyUI API |

## Channel Presets

| Channel | Resolution | FPS | Steps | CFG | Format |
|---------|-----------|-----|-------|-----|--------|
| `youtube-trap` | 1920×1080 | 24 | 25 | 7.5 | mp4 |
| `youtube-lofi` | 1920×1080 | 24 | 20 | 7.0 | mp4 |
| `tiktok` | 1080×1920 | 30 | 15 | 7.0 | mp4 |
| `instagram` | 1080×1080 | 24 | 25 | 7.0 | mp4 |
| `album-art` | 3000×3000 | 1 | 30 | 7.5 | png |
| `twitter` | 1280×720 | 24 | 15 | 7.0 | mp4 |

Each channel defines: resolution, FPS, steps, CFG, sampler, negative prompt, prompt prefix/suffix.

## Trap Style Presets

| Style | Description | Keyframes | Steps |
|-------|-------------|-----------|-------|
| `dark-trap` | Dark, cinematic — neon lights, urban decay, luxury | 6 | 25 |
| `neon-trap` | Vibrant neon cyberpunk — synthwave colors | 8 | 20 |
| `luxury-trap` | High-end — mansions, cars, jewelry, money | 4 | 30 |
| `horror-trap` | Dark horror — haunted, eerie, intense | 5 | 25 |
| `drill-trap` | UK drill — dark streets, moody, intense | 6 | 25 |

## Models (from `music-video-models` Modal volume)

### Active models (working)
| Category | Model | Size | Notes |
|----------|-------|------|-------|
| `diffusion_models` | `flux2_dev_fp8mixed.safetensors` | 33 GB | Single-file FLUX.2 dev transformer |
| `text_encoders` | `mistral_3_small_flux2_bf16.safetensors` | 33 GB | Matching text encoder for FLUX.2 dev |
| `vae` | `flux2-vae.safetensors` | 321 MB | FLUX VAE |
| `loras/` | 18 files | various | Flattened from arcane/, cinematic/, devil_may_cry/, turbo/, models/ |

### Key model files in volume structure
```
music-video-models/
├── flux2-klein-9B/           # Klein 9B model (split format, not directly loadable)
│   ├── transformer/          # diffusion_pytorch_model-00001/00002.safetensors + index
│   ├── text_encoder/         # model-00001/00004.safetensors + index
│   └── vae/
├── models/
│   ├── clip/                 # 4 text encoder files (single-file)
│   ├── unet/                 # 4 diffusion model files (single-file)
│   ├── vae/                  # 4 VAE files
│   └── loras/                # 12 LoRA files (top-level)
├── loras/
│   ├── arcane/               # 4 LoRA files
│   ├── cinematic/            # 1 LoRA + checkpoints
│   ├── devil_may_cry/        # 4 LoRA files
│   └── turbo/                # 1 LoRA (1.3 GB)
├── ltx-video/                # LTX video models (3 versions)
├── ltx-2.3/                  # LTX 2.3 with upscalers
└── models--*/                # HF snapshot format (split shards)
```

## How It Works

### Container startup sequence:
1. **Symlink models** from Modal volume into `/root/comfy/ComfyUI/models/`
2. **Flatten LoRAs** — symlink LoRA files from subdirectories to top-level `loras/`
3. **Copy RIFE model** from volume if available (optional, for frame interpolation)
4. **Start ComfyUI** headless on port 8188 via `comfy launch --background`
5. **Start poll worker** — background thread that polls ComfyUI history for completed generations

### Request flow:
1. Client POSTs to `/generate` or `/generate-video` with prompt + settings
2. Server builds FLUX workflow JSON (with optional LoRAs, channel presets)
3. Server submits to ComfyUI's `/prompt` endpoint
4. Server stores `prompt_id` in `results_store` with status `running`
5. Background poll worker checks ComfyUI `/history/{prompt_id}` every 2s
6. When complete, worker stores image metadata in `results_store`
7. Client polls `/result/{id}` and receives base64-encoded images/videos when done

### Image workflow JSON:
```
1: UNETLoader ──→ flux2_dev_fp8mixed.safetensors
2: VAELoader ───→ flux2-vae.safetensors
3: CLIPLoader ──→ mistral_3_small_flux2_bf16.safetensors (type=flux2)
4: CLIPTextEncode ← (3) clip, prompt text
5: EmptyFlux2LatentImage (width × height)
6: KSampler ← (1) model, (4) conditioning, (5) latent
7: VAEDecode ← (6) samples, (2) vae
8: SaveImage ← (7) images
```

### Video workflow JSON:
```
1: UNETLoader, 2: VAELoader, 3: CLIPLoader (shared)
For each keyframe i:
  4+i: CLIPTextEncode ← prompt with scene description
  5+i: EmptyFlux2LatentImage
  6+i: KSampler ← model, conditioning, latent
  7+i: VAEDecode ← samples, vae
  8+i: SaveImage ← images (keyframe_NNN)
Last: VHS_VideoCombine ← all keyframe images → MP4
```

## Custom Nodes

| Node | Purpose | Installed |
|------|---------|-----------|
| `ComfyUI-Frame-Interpolation` | RIFE frame interpolation | ✅ |
| `ComfyUI-VideoHelperSuite` | VHS_VideoCombine MP4 output | ✅ |
| `ComfyUI-IPAdapter-Plus` | Style transfer from reference images | ✅ |
| `ComfyUI-ControlNet-Preprocessors` | Pose, depth, canny edge detection | ✅ |
| `ComfyUI-Upscale-Model-Loader` | 4x upscaling | ✅ |
| `ComfyUI-Impact-Pack` | Utility nodes | ✅ |
| `ComfyUI-Essentials` | Image resize, crop, etc. | ✅ |
| `ComfyUI-KJNodes` | GetImageSize, ImageConcat, etc. | ✅ |

## Performance

Typical generation times on A10 GPU:
- 512×512, 10 steps: ~55s
- 512×512, 15 steps: ~100s
- 1024×1024, 20 steps: ~3-5 min
- 1920×1080, 25 steps: ~8-12 min per keyframe

Video (3 keyframes, 512×512, 10 steps each): ~3-5 min total

## Key Learnings / Pitfalls

1. **CLIPLoader type=flux2** — Must use `type="flux2"` not default. DualCLIPLoader doesn't work with FLUX.2 dev.
2. **CLIPTextEncode not CLIPTextEncodeFlux** — The Flux-specific node expects `clip_l` and `t5xxl` inputs that don't match our text encoder. Use basic CLIPTextEncode.
3. **LoRA symlink ordering** — Symlink subdirectory LoRAs to top-level `loras/` so LoraLoader can find them. Don't symlink `models/loras` over `loras/`.
4. **RIFE model** — `flownet.pkl` cannot be downloaded from public URLs reliably. Must be uploaded to Modal volume at `rife/flownet.pkl`.
5. **VHS_VideoCombine** — Saves MP4 but doesn't report in ComfyUI history API. Client-side FFmpeg encoding is used as fallback.
6. **1080p generation** — Very slow with FLUX.2 dev (8-12 min per keyframe). Use 512×512 for previews, 1080p for final output.
7. **Immediate-return async pattern** — `/generate` returns `prompt_id` immediately, `/result/{id}` polls. Don't block the request handler.
8. **Background poll worker** — Thread that polls ComfyUI `/history/{prompt_id}` every 2s and updates `results_store`.
9. **EmptyFlux2LatentImage** — Use this (not EmptyLatentImage) for FLUX models.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ComfyUI not ready" | Container cold-starting. Wait 60s and retry |
| "ComfyUI rejected prompt" | Check workflow JSON — model names must match exactly |
| LoRA not found | Use just filename, not path. Check `/models` endpoint |
| Generation times out | Increase timeout, or reduce resolution/steps |
| Model dimension mismatch | Ensure text encoder matches transformer (hidden_size, intermediate_size) |
| View image 404 | Image not generated yet, or filename wrong |
| 503 Service Unavailable | Container scaled down. Make a request to wake it up |
| 500 Internal Server Error | Check `modal app logs comfyui-headless` for traceback |

## Environment

- **Modal app name:** `comfyui-headless`
- **Modal volume:** `music-video-models` (150+ GB)
- **GPU:** NVIDIA A10 (24 GB VRAM)
- **ComfyUI version:** 0.25.0
- **Python:** 3.11
- **GitHub:** `scentblenderaromacompany-source/modal-comfyui-headless`
