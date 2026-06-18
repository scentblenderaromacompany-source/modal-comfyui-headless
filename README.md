# Headless ComfyUI on Modal Labs

Run ComfyUI as a serverless GPU worker. Submit workflow JSON, poll for results,
and retrieve generated images — no GUI, no local GPU required.

## How It Works

```
Your code / curl
      |
      v
[ Modal HTTP proxy ]  <-- modal.web_server(:8000)
      |
      v
[ ComfyUI server ]   <-- `comfy launch --background`
      |
      v
[ GPU (A10G/A100/L4/T4) ]
```

The Modal app:
1. **Builds once** — installs ComfyUI, downloads models to a Modal Volume
2. **Starts fast** — GPU memory snapshots resume in ~10s vs ~60s cold start
3. **Exposes the full ComfyUI API** — transparent HTTP proxy on port 8000
4. **Scales to zero** — shuts down after 5min idle, no cost when not in use

## Quick Start

### 1. Install dependencies

```bash
pip install modal httpx
modal setup   # authenticate with Modal Labs
```

### 2. Configure models

```bash
cp models.example.py models.py
# Edit models.py to add your checkpoints, LoRAs, etc.
```

### 3. Configure custom nodes (optional)

```bash
cp plugins.example.py plugins.py
# Edit plugins.py to add ComfyUI registry node IDs
```

### 4. Deploy

```bash
modal deploy comfyui_headless.py
```

This outputs your app URL, e.g. `https://yourname--comfyui-headless-serve.modal.run`

### 5. Generate

```bash
export MODAL_URL=https://yourname--comfyui-headless-serve.modal.run

# Health check
python client.py --action health

# Run a workflow
python client.py --action run --workflow my_workflow_api.json --output ./outputs

# Or use the API directly with curl:
curl -X POST "$MODAL_URL/prompt" \
  -H "Content-Type: application/json" \
  -d @my_workflow_api.json
```

## API Reference

The full ComfyUI HTTP API is available at your Modal URL:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/prompt` | Submit a workflow JSON |
| `GET` | `/history/{id}` | Check status / get outputs |
| `GET` | `/view?filename=...` | Download output image |
| `GET` | `/system_stats` | Health check |
| `GET` | `/object_info` | List all node types |
| `GET` | `/queue` | View queue status |
| `POST` | `/interrupt` | Cancel current prompt |
| `POST` | `/free` | Free memory |

### Submitting a Prompt

```bash
curl -X POST "$MODAL_URL/prompt" \
  -H "Content-Type: application/json" \
  -d '{"prompt": {"3": {"class_type": "KSampler", ...}}}'
# Returns: {"prompt_id": "abc-123", ...}
```

### Checking Results

```bash
curl "$MODAL_URL/history/abc-123"
# When done, contains outputs with image metadata
```

### Downloading Images

```bash
curl "$MODAL_URL/view?filename=modal_comfyui_00001_.png&type=output" -o output.png
```

## GPU Options

Edit the `@app.cls` decorator in `comfyui_headless.py`:

```python
@app.cls(
    gpu="A10G",    # Options: T4, L4, A10G, A100, H100
    ...
)
```

| GPU | VRAM | Best For | ~Cost/hr |
|-----|------|----------|----------|
| T4  | 16GB | SD 1.5, light SDXL | $0.40 |
| L4  | 24GB | SDXL, FLUX dev | $0.70 |
| A10G| 24GB | SDXL, FLUX, video | $1.00 |
| A100| 40GB| Large models, batch | $1.50 |

## Model Management

Models are downloaded into a Modal Volume at build time and symlinked into
ComfyUI at container startup. This means:

- **First deploy**: downloads models (slow, one-time)
- **Subsequent deploys**: models are already cached (fast)
- **Container restart**: symlinks are recreated instantly

To add models, edit `models.py` and re-deploy:

```python
models = [
    {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "filename": "flux1-dev.safetensors",
        "model_dir": "diffusion_models",
    },
]
```

## Architecture Details

### Image Build Pipeline

```
debian-slim 3.11
  + git, git-lfs, libgl, ffmpeg, aria2
  + comfy-cli, huggingface_hub, httpx
  + comfy --skip-prompt install --nvidia   # installs ComfyUI
  + download models to /cache/models/       # Modal Volume
  + install custom nodes via comfy-cli
```

### Container Lifecycle

```
Cold start:    image pull → model symlink → comfy launch → ready (~60s)
Warm start:    snapshot restore → comfy already running → ready (~10s)
Idle:          5min timeout → container stops → $0 cost
```

### Memory Snapshots

`enable_memory_snapshot=True` + `enable_gpu_snapshot=True` means Modal
snapshots the GPU memory after first start. On restore, ComfyUI is already
loaded in VRAM — no re-initialization needed.

## Files

| File | Purpose |
|------|---------|
| `comfyui_headless.py` | Main Modal app — deploy this |
| `models.example.py` | Copy to `models.py` and add your models |
| `plugins.example.py` | Copy to `plugins.py` and add custom node IDs |
| `client.py` | Python client for the deployed app |

## Troubleshooting

**Container won't start**: Check `modal app logs comfyui-headless`
**Model not found**: Verify `model_dir` matches ComfyUI's expected folder name
**Out of memory**: Switch to a larger GPU or reduce resolution/batch size
**Slow first request**: Normal — cold start takes ~60s. Subsequent requests ~10s.
