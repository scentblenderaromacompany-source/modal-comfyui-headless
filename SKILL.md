# Content Generation Automation Skill

This skill documents the headless ComfyUI system on Modal Labs for automated content generation.

## Architecture

```
Local machine                    Modal Cloud
┌──────────────┐                ┌─────────────────────────┐
│ generate.py  │──POST /generate─▶  ComfyUI headless       │
│              │◀──prompt_id──────  (A10 GPU)              │
│              │                │                         │
│              │──GET /result/─▶  │  music-volume         │
│              │◀──{images}──────  │    (150+ GB models)   │
│              │                │                         │
│              │──GET /view─────▶│  output/               │
│              │◀──PNG──────────│    flux_00001_.png      │
└──────────────┘                └─────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `comfyui_headless.py` | Main Modal app — deploy with `modal deploy comfyui_headless.py` |
| `generate.py` | Local client — submit generations, poll, save to disk |
| `models.py` | Model definitions (currently empty, uses volume directly) |
| `plugins.py` | Custom node registry IDs (currently empty) |

## Deployment

```bash
cd /home/bobby/modal-comfyui-headless
modal deploy comfyui_headless.py
# App URL: https://robertmcasper--comfyui-headless-serve.modal.run
```

## Usage

### Generate an image
```bash
cd /home/bobby/modal-comfyui-headless
python generate.py "a beautiful sunset over mountains" --width 512 --height 512 --steps 15
```

### CLI options
```
python generate.py "prompt" [options]
  --width N       Image width (default: 1024)
  --height N      Image height (default: 1024)
  --steps N       Sampling steps (default: 20)
  --cfg N         CFG scale (default: 7.0)
  --seed N        Random seed (default: random)
  --output DIR    Output directory (default: ./output)
  --url URL       Modal app URL
  --list-models   List available models and exit
```

### API endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Submit generation. Body: `{prompt, width, height, steps, cfg, seed}`. Returns `{prompt_id, status}` |
| GET | `/result/{id}` | Poll for results. Returns `{status: running\|done, images: [{filename, base64}]}` |
| GET | `/models` | List available models by category |
| GET | `/system_stats` | Health check (GPU, RAM, ComfyUI version) |
| GET | `/view?filename=X&subfolder=&type=output` | Download output image |
| * | All other paths | Transparent proxy to ComfyUI API |

### API workflow
```bash
# 1. Submit
curl -X POST "$MODAL_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "sunset", "width": 512, "height": 512, "steps": 15, "seed": 42}'
# Returns: {"status": "accepted", "prompt_id": "abc-123", "poll_url": "/result/abc-123"}

# 2. Poll
curl "$MODAL_URL/result/abc-123"
# Returns: {"status": "running"} or {"status": "done", "images": [...]}

# 3. Download (or use generate.py which does all steps)
curl "$MODAL_URL/view?filename=flux_00001_.png&subfolder=&type=output" -o output.png
```

## Models (from `music-video-models` Modal volume)

### Active models (symlained and working)
| Category | Model | Size | Notes |
|----------|-------|------|-------|
| `diffusion_models` | `flux2_dev_fp8mixed.safetensors` | 33 GB | Single-file FLUX.2 dev transformer |
| `text_encoders` | `mistral_3_small_flux2_bf16.safetensors` | 33 GB | Matching text encoder for FLUX.2 dev |
| `text_encoders` | `klein_qwen_3_8b_fp8mixed.safetensors` | 8 GB | Klein 9B text encoder (not used) |
| `text_encoders` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6 GB | T5 text encoder (not used) |
| `vae` | `flux2-vae.safetensors` | 321 MB | FLUX VAE |
| `vae` | `klein_flux2-vae.safetensors` | 321 MB | Klein VAE |
| `loras/` | 18 files | various | Flattened from arcane/, cinematic/, devil_may_cry/, turbo/, models/ |

### Models in volume but NOT symlinked (need config update)
- `qwen_image_fp8_hq.safetensors` (21 GB) — Qwen image model
- `smoothMixWan2214BI2V_i2vV20High.safetensors` (13 GB) — Wan video I2V
- `ltx-video/`, `ltx-2.3/` — LTX video models
- `models/unet/` — additional diffusion models
- `models/vae/` — additional VAEs
- Klein 9B split transformer and text encoder (incompatible with UNETLoader)

### Key model files in volume structure
```
music-video-models/
├── flux2-klein-9B/           # Klein 9B model (split format)
│   ├── transformer/          # diffusion_pytorch_model-00001/00002.safetensors + index
│   ├── text_encoder/         # model-00001/00004.safetensors + index
│   └── vae/                  # diffusion_pytorch_model.safetensors
├── models/
│   ├── clip/                 # 4 text encoder files (single-file)
│   ├── unet/                 # 4 diffusion model files (single-file)
│   ├── vae/                  # 4 VAE files
│   └── loras/                # 12 LoRA files (top-level)
├── loras/
│   ├── arcane/               # 4 LoRA files (9 items including .cache)
│   ├── cinematic/            # 1 LoRA + checkpoints
│   ├── devil_may_cry/        # 4 LoRA files
│   └── turbo/                # 1 LoRA (1.3 GB)
├── ltx-video/                # LTX video models (3 versions)
├── ltx-2.3/                  # LTX 2.3 with upscalers
└── models--*/                # HF snapshot format (split shards)
```

## How It Works

### Container startup sequence:
1. **Symlink models** from Modal volume `music-volume-models/` into `/root/comfy/ComfyUI/models/`
2. **Flatten LoRAs** — symlink LoRA files from subdirectories (`loras/arcane/`, etc.) to top-level `loras/`
3. **Start ComfyUI** headless on port 8188 via `comfy launch --background`
4. **Start poll worker** — background thread that polls ComfyUI history for completed generations

### Request flow:
1. Client POSTs to `/generate` with prompt + settings
2. Server builds FLUX workflow JSON, submits to ComfyUI's `/prompt` endpoint
3. Server stores `prompt_id` in `results_store` with status `running`
4. Background poll worker checks ComfyUI `/history/{prompt_id}` every 2s
5. When complete, worker stores image metadata in `results_store`
6. Client polls `/result/{id}` and receives base64-encoded images when done

### Workflow JSON structure:
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

## Adding New Models

### From HuggingFace
Edit `models.py`:
```python
models = [
    {"repo_id": "org/model-name", "filename": "model.safetensors", "model_dir": "diffusion_models"},
]
```
Then `modal deploy comfyui_headless.py`.

### From Cloudflare R2
The R2 credentials are stored in Modal Secret `r2-credentials`. To use R2 as primary source instead of Modal Volume, update the `_r2_secret()` and `_download_from_r2()` functions in `comfyui_headless.py`.

### To a new Modal volume
Edit `VOLUME_TO_COMFY` in `comfyui_headless.py`:
```python
VOLUME_TO_COMFY = [
    ("volume-subdir", "comfyui-model-subdir"),
    ...
]
```

## Generating with LoRAs

The `generate.py` client supports LoRA via the API. Submit with `"lora": "filename.safetensors"` (just the filename, no path). Available LoRAs in top-level `loras/`:
- `dever_arcane_flux2_klein_9b.safetensors` (79 MB)
- `dever_clothes_line_flux2_klein_9b.safetensors` (158 MB)
- `dever_cyanide_and_happiness_flux2_klein_9b.safetensors` (79 MB)
- `dever_devil_may_cry_flux2_klein_9b.safetensors` (158 MB)
- `Flux_2-Turbo-LoRA_comfyui.safetensors` (2.6 GB)
- And 13 more

Note: LoRA loading adds significant time to generation. The `/result/` endpoint handles polling with a background worker.

## Performance

Typical generation times on A10 GPU:
- 512×512, 15 steps: ~55s
- 1024×1024, 20 steps: ~3-5 min

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ComfyUI not ready" | Container is cold-starting. Wait 60s and retry |
| "ComfyUI rejected prompt" | Check workflow JSON — model names must match exactly |
| LoRA not found | Use just filename, not path. Check `/models` endpoint |
| Generation times out | Increase `max_wait` in poll loop, or use async polling |
| Model dimension mismatch | Ensure text encoder matches transformer (hidden_size, intermediate_size) |
| View image 404 | Image may not be generated yet, or filename is wrong |

## Environment

- **Modal app name:** `comfyui-headless`
- **Modal volume:** `music-video-models` (150+ GB)
- **GPU:** NVIDIA A10 (24 GB VRAM)
- **ComfyUI version:** 0.25.0
- **Python:** 3.11
- **Key packages:** comfy-cli, starlette, httpx
