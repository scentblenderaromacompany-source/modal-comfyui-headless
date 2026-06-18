"""
Headless ComfyUI on Modal Labs
================================
Runs ComfyUI as a serverless GPU worker with models from Modal Volume.

Endpoints:
  POST /generate     - Submit generation, returns {prompt_id, status}
  GET  /result/{id}  - Poll for results, returns {status, files, ...}
  GET  /models       - List available models
  GET  /system_stats - Health check
  *                  - All ComfyUI API endpoints proxied transparently
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_INTERNAL_PORT = 8188
COMFYUI_DIR = Path("/root/comfy/ComfyUI")
MVM_VOL_PATH = Path("/mvm")

VOLUME_TO_COMFY: list[tuple[str, str]] = [
    ("flux2-klein-9B/transformer", "diffusion_models/flux2-klein-9B/transformer"),
    ("flux2-klein-9B/text_encoder", "text_encoders/flux2-klein-9B"),
    ("flux2-klein-9B/vae", "vae/flux2-klein-9B"),
    ("loras/arcane", "loras/arcane"),
    ("loras/cinematic", "loras/cinematic"),
    ("loras/devil_may_cry", "loras/devil_may_cry"),
    ("loras/turbo", "loras/turbo"),
    ("ltx-video", "diffusion_models/ltx-video"),
    ("ltx-2.3", "diffusion_models/ltx-2.3"),
    ("models/clip", "text_encoders"),
    ("models/clip_vision", "clip_vision"),
    ("models/loras", "loras/models"),
    ("models/unet", "diffusion_models"),
    ("models/vae", "vae"),
    ("models--black-forest-labs--FLUX.2-klein-9B/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216",
     "diffusion_models/flux2-klein-9B-hf"),
    ("models--Lightricks--LTX-Video/snapshots/8984fa25007f376c1a299016d0957a37a2f797bb",
     "diffusion_models/ltx-video-hf"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_port(host: str, port: int, timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Port {port} not ready after {timeout}s")


def _comfy_request(comfy_base: str, method: str, path: str, data: bytes = None, timeout: int = 600) -> tuple[int, bytes, dict]:
    """Make HTTP request to local ComfyUI."""
    import urllib.request
    import urllib.error
    url = f"{comfy_base}{path}"
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


# ---------------------------------------------------------------------------
# Image build
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "libsm6",
                 "libxrender1", "libxext6", "ffmpeg")
    .pip_install("comfy-cli>=1.3.0", "httpx>=0.27.0", "starlette>=0.38.0")
    .run_commands(
        "comfy --skip-prompt install --nvidia",
        "git lfs install",
        # Install frame interpolation custom node
        "cd /root/comfy/ComfyUI/custom_nodes && "
        "git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git 2>/dev/null || true",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && "
        "pip install -r requirements.txt --quiet 2>/dev/null || true",
        # Install Video Helper Suite
        "cd /root/comfy/ComfyUI/custom_nodes && "
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 2>/dev/null || true",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && "
        "pip install -r requirements.txt --quiet 2>/dev/null || true",
        # NOTE: RIFE model (flownet.pkl) is downloaded at container startup
        # in serve() because the custom node directory must exist first",
    )
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App(name="comfyui-headless", image=image)


@app.function(
    gpu="A10G",
    volumes={"/mvm": modal.Volume.from_name("music-video-models")},
    scaledown_window=300,
    timeout=3600,
    max_containers=1,
    memory=16384,
)
@modal.asgi_app()
def serve():
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import Response, JSONResponse
    from starlette.routing import Route

    comfy_port = COMFYUI_INTERNAL_PORT
    comfy_base = f"http://127.0.0.1:{comfy_port}"
    comfy_ready = threading.Event()
    startup_error: list[str] = []

    # ---- Symlink models ----
    comfy_models = COMFYUI_DIR / "models"
    mvm = MVM_VOL_PATH

    linked_count = 0
    for vol_subdir, comfy_subdir in VOLUME_TO_COMFY:
        src = mvm / vol_subdir
        dst = comfy_models / comfy_subdir
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                import shutil
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(src)
        linked_count += 1

    # ---- Symlink LoRAs from subdirectories to top-level loras/ ----
    loras_dir = comfy_models / "loras"
    if loras_dir.exists():
        for item in loras_dir.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    if f.is_file() and f.suffix == ".safetensors":
                        target = loras_dir / f.name
                        if not target.exists():
                            os.symlink(f, target)
        # Also link loras/models/ contents
        loras_models = loras_dir / "models"
        if loras_models.exists():
            for f in loras_models.iterdir():
                if f.is_file() and f.suffix == ".safetensors":
                    target = loras_dir / f.name
                    if not target.exists():
                        os.symlink(f, target)
    print(f"[Models] Linked {linked_count} dirs, LoRAs flattened")

    # ---- Download RIFE frame interpolation model (optional) ----
    # RIFE model URLs are unreliable. Frame interpolation is optional.
    # Video generation works without it — frames are generated directly.
    try:
        rife_dir = COMFYUI_DIR / "models" / "frame_interpolation"
        rife_dir.mkdir(parents=True, exist_ok=True)
        rife_model = rife_dir / "flownet.pkl"
        if not rife_model.exists():
            import urllib.request as _dl
            urls = [
                "https://huggingface.co/datasets/nicolai256/RIFE_checkpoints/resolve/main/flownet.pkl",
            ]
            for url in urls:
                try:
                    req = _dl.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    _dl.urlretrieve(req, str(rife_model))
                    if rife_model.exists() and rife_model.stat().st_size > 100000:
                        print(f"[RIFE] Model downloaded ({rife_model.stat().st_size/1024/1024:.1f} MB)")
                        break
                    else:
                        rife_model.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[RIFE] Failed: {e}")
            else:
                print("[RIFE] Model not available — video will use direct frame generation (no interpolation)")
        else:
            print(f"[RIFE] Model present ({rife_model.stat().st_size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"[RIFE] Error: {e}")

    # ---- Start ComfyUI ----
    def _start_comfyui():
        nonlocal startup_error
        try:
            proc = subprocess.Popen(
                ["comfy", "launch", "--background", "--",
                 "--listen", "127.0.0.1", "--port", str(comfy_port), "--disable-auto-launch"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "COMFY_CLI_BACKGROUND": "true"},
            )
            _wait_for_port("127.0.0.1", comfy_port, timeout=300)
            comfy_ready.set()
            print(f"[ComfyUI] Ready (pid {proc.pid})")
        except Exception as e:
            startup_error.append(str(e))
            print(f"[ComfyUI] ERROR: {e}")

    threading.Thread(target=_start_comfyui, daemon=True).start()

    # ---- In-flight generations store ----
    # Maps prompt_id -> {status, files, error, ...}
    results_store: dict[str, dict] = {}
    results_lock = threading.Lock()

    # ---- Background worker: polls ComfyUI for completed prompts ----
    def _poll_worker():
        """Poll ComfyUI history for completed generations."""
        while True:
            time.sleep(2)
            with results_lock:
                pending = {pid: info for pid, info in results_store.items() if info["status"] == "running"}
            if not pending:
                continue
            for prompt_id, info in pending.items():
                try:
                    _, body, _ = _comfy_request(comfy_base, "GET", f"/history/{prompt_id}", timeout=10)
                    history = json.loads(body)
                    if prompt_id in history:
                        data = history[prompt_id]
                        outputs = data.get("outputs", {})
                        images = []
                        for nid, nout in outputs.items():
                            for img in nout.get("images", []):
                                images.append({
                                    "filename": img["filename"],
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output"),
                                    "node_id": nid,
                                })
                        with results_lock:
                            results_store[prompt_id]["status"] = "done"
                            results_store[prompt_id]["images"] = images
                            results_store[prompt_id]["done_at"] = time.time()
                except Exception:
                    pass

    threading.Thread(target=_poll_worker, daemon=True).start()

    # ---- Fallback inline workflow builders ----

    class _NodeID:
        def __init__(self, start=1):
            self._counter = start - 1
        def __call__(self):
            self._counter += 1
            return str(self._counter)

    def _build_flux_workflow(prompt, width, height, steps, cfg, seed, loras):
        """Fallback inline FLUX image workflow."""
        workflow = {}
        nid = _NodeID()
        n_unet = nid(); n_vae = nid(); n_clip = nid()
        workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
        workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
        workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}
        model_out, clip_out = n_unet, n_clip
        if loras:
            for ln, ls in loras:
                n_lora = nid()
                workflow[n_lora] = {"class_type": "LoraLoader", "inputs": {"lora_name": ln, "strength_model": ls, "strength_clip": ls, "model": [model_out, 0], "clip": [clip_out, 0]}}
                model_out, clip_out = n_lora, n_lora
        n_text = nid(); workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": [clip_out, 0]}}
        n_latent = nid(); workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        n_sampler = nid(); workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [model_out, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
        n_decode = nid(); workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
        n_save = nid(); workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": "flux"}}
        return workflow

    def _build_video_workflow(prompt, width, height, num_keyframes, frames_per_keyframe, output_fps, steps, seed, loras):
        """Fallback inline hyperframes video workflow."""
        workflow = {}
        nid = _NodeID()
        n_unet = nid(); n_vae = nid(); n_clip = nid()
        workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
        workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
        workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}
        model_out, clip_out = n_unet, n_clip
        if loras:
            for ln, ls in loras:
                n_lora = nid()
                workflow[n_lora] = {"class_type": "LoraLoader", "inputs": {"lora_name": ln, "strength_model": ls, "strength_clip": ls, "model": [model_out, 0], "clip": [clip_out, 0]}}
                model_out, clip_out = n_lora, n_lora
        keyframe_saves = []
        for i in range(num_keyframes):
            n_text = nid(); n_latent = nid(); n_sampler = nid(); n_decode = nid(); n_save = nid()
            workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": f"{prompt}, frame {i+1} of {num_keyframes}", "clip": [clip_out, 0]}}
            workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
            workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": seed + i * 1000, "steps": steps, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
            workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
            workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"keyframe_{i:03d}"}}
            keyframe_saves.append(n_save)
        for i in range(len(keyframe_saves) - 1):
            n_rife = nid()
            workflow[n_rife] = {"class_type": "RIFE_VFI", "inputs": {"ckpt_name": "flownet.pkl", "frames": [keyframe_saves[i], 0], "multiplier": frames_per_keyframe, "fast_mode": True, "ensemble": True, "scale_factor": 1.0}}
        if keyframe_saves:
            n_video = nid()
            workflow[n_video] = {"class_type": "VHS_VideoCombine", "inputs": {"images": [keyframe_saves[0], 0], "frame_rate": output_fps, "loop_count": 0, "filename_prefix": "hyperframe", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 18, "save_output": True, "videopreview": {"hidden": True}}}
        return workflow

    async def generate(request):
        """Submit image generation. Returns {prompt_id, status: 'accepted'} immediately."""
        if not comfy_ready.is_set():
            return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "a beautiful landscape")
        width = body.get("width", 1024)
        height = body.get("height", 1024)
        steps = body.get("steps", 20)
        cfg = body.get("cfg", 7.0)
        seed = body.get("seed", 42)
        loras = body.get("loras", [])  # [(name, strength), ...]

        # Build workflow using the workflow engine
        try:
            from workflows import build_txt2img_flux, GenerationRequest
            req = GenerationRequest(
                prompt=prompt, width=width, height=height,
                steps=steps, cfg=cfg, seed=seed, loras=loras,
            )
            workflow = build_txt2img_flux(req)
        except ImportError:
            # Fallback to inline workflow if workflows.py not available
            workflow = _build_flux_workflow(prompt, width, height, steps, cfg, seed, loras)

        return await _submit_workflow(workflow, results_store, results_lock)

    async def generate_video(request):
        """Submit video generation with hyperframes. Returns {prompt_id, status: 'accepted'}."""
        if not comfy_ready.is_set():
            return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "a beautiful landscape")
        width = body.get("width", 512)
        height = body.get("height", 512)
        num_keyframes = body.get("num_keyframes", 4)
        frames_per_keyframe = body.get("frames_per_keyframe", 8)
        output_fps = body.get("output_fps", 24)
        steps = body.get("steps", 15)
        seed = body.get("seed", 42)
        loras = body.get("loras", [])
        channel = body.get("channel")  # Optional channel preset

        # Apply channel preset if specified
        if channel:
            try:
                from channels import get_channel, build_prompt
                ch = get_channel(channel)
                width = ch.width
                height = ch.height
                steps = ch.steps
                cfg = ch.cfg
                loras = ch.loras
                output_fps = ch.fps
                num_keyframes = ch.num_keyframes
                frames_per_keyframe = ch.frames_per_keyframe
                prompt = build_prompt(channel, prompt)
            except (ImportError, ValueError) as e:
                print(f"[generate-video] Channel '{channel}' error: {e}, using defaults")

        # Build video workflow
        try:
            from workflows import build_txt2video_hyperframes, GenerationRequest
            req = GenerationRequest(
                prompt=prompt, width=width, height=height,
                steps=steps, cfg=cfg, seed=seed, loras=loras,
            )
            workflow = build_txt2video_hyperframes(req, num_keyframes, frames_per_keyframe, output_fps)
        except ImportError:
            workflow = _build_video_workflow(prompt, width, height, num_keyframes, frames_per_keyframe, output_fps, steps, seed, loras)

        return await _submit_workflow(workflow, results_store, results_lock)

    async def _submit_workflow(workflow, results_store, results_lock):
        """Submit a workflow to ComfyUI and track it."""
        import json as _json
        import uuid as _uuid
        client_id = str(_uuid.uuid4())
        payload = _json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        code, resp_body, _ = _comfy_request(comfy_base, "POST", "/prompt", data=payload, timeout=30)

        if code != 200:
            return JSONResponse({"error": f"ComfyUI error {code}: {resp_body.decode()[:300]}"}, status_code=500)

        result = _json.loads(resp_body)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return JSONResponse({"error": f"No prompt_id: {result}"}, status_code=500)

        with results_lock:
            results_store[prompt_id] = {
                "status": "running",
                "images": [],
                "submitted_at": time.time(),
            }

        return JSONResponse({
            "status": "accepted",
            "prompt_id": prompt_id,
            "poll_url": f"/result/{prompt_id}",
        })

    async def get_result(request):
        """Poll for generation results."""
        prompt_id = request.path_params["id"]
        with results_lock:
            info = results_store.get(prompt_id)
        if not info:
            return JSONResponse({"error": "Unknown prompt_id"}, status_code=404)

        if info["status"] == "running":
            elapsed = round(time.time() - info["submitted_at"], 1)
            return JSONResponse({"status": "running", "elapsed_seconds": elapsed})

        # Done — download images from ComfyUI and return
        output_files = []
        for img_info in info.get("images", []):
            view_url = f"/view?filename={img_info['filename']}&subfolder={img_info.get('subfolder', '')}&type={img_info.get('type', 'output')}"
            try:
                _, img_data, _ = _comfy_request(comfy_base, "GET", view_url, timeout=60)
                output_files.append({
                    "filename": img_info["filename"],
                    "size_bytes": len(img_data),
                    "base64": base64.b64encode(img_data).decode(),
                })
            except Exception as e:
                output_files.append({"filename": img_info["filename"], "error": str(e)})

        return JSONResponse({
            "status": "done",
            "prompt_id": prompt_id,
            "images": output_files,
            "settings": info.get("settings", {}),
        })

    async def generate_video(request):
        """
        Generate video with hyperframes (frame interpolation).
        
        POST /generate-video
        Body: {
            "prompt": "a beautiful landscape transforming from day to night",
            "width": 512,
            "height": 512,
            "num_keyframes": 4,           // Number of keyframes to generate
            "frames_per_keyframe": 8,     // Interpolated frames between each pair
            "output_fps": 24,
            "steps": 15,                  // Steps per keyframe
            "seed": 42
        }
        
        Returns: {prompt_id, status: "accepted", poll_url: "/result/{id}"}
        """
        if not comfy_ready.is_set():
            return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "a beautiful landscape")
        width = body.get("width", 512)
        height = body.get("height", 512)
        num_keyframes = body.get("num_keyframes", 4)
        frames_per_keyframe = body.get("frames_per_keyframe", 8)
        output_fps = body.get("output_fps", 24)
        steps = body.get("steps", 15)
        seed = body.get("seed", 42)

        # Build hyperframes workflow
        # 1. Generate keyframes with FLUX
        # 2. Interpolate between them with RIFE
        # 3. Combine into video with VHS
        workflow = {}
        node_id = 0

        def nid():
            nonlocal node_id
            node_id += 1
            return str(node_id)

        # Shared model loaders
        n_unet = nid()
        n_vae = nid()
        n_clip = nid()
        workflow[n_unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}}
        workflow[n_vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
        workflow[n_clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2"}}

        keyframe_save_nodes = []
        vae_decode_nodes = []

        # Generate keyframes
        for i in range(num_keyframes):
            n_text = nid()
            n_latent = nid()
            n_sampler = nid()
            n_decode = nid()
            n_save = nid()

            frame_prompt = f"{prompt}, scene {i+1} of {num_keyframes}"
            workflow[n_text] = {"class_type": "CLIPTextEncode", "inputs": {"text": frame_prompt, "clip": [n_clip, 0]}}
            workflow[n_latent] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
            workflow[n_sampler] = {"class_type": "KSampler", "inputs": {"seed": seed + i * 1000, "steps": steps, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": [n_unet, 0], "positive": [n_text, 0], "negative": [n_text, 0], "latent_image": [n_latent, 0]}}
            workflow[n_decode] = {"class_type": "VAEDecode", "inputs": {"samples": [n_sampler, 0], "vae": [n_vae, 0]}}
            workflow[n_save] = {"class_type": "SaveImage", "inputs": {"images": [n_decode, 0], "filename_prefix": f"keyframe_{i:03d}"}}
            keyframe_save_nodes.append(n_save)
            vae_decode_nodes.append(n_decode)

        # Frame interpolation (optional, if RIFE model available)
        rife_available = (COMFYUI_DIR / "models" / "frame_interpolation" / "flownet.pkl").exists()
        interpolated = []
        if rife_available:
            for i in range(len(keyframe_save_nodes) - 1):
                n_model = nid()
                n_rife = nid()
                workflow[n_model] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "flownet.pkl"}}
                workflow[n_rife] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": [n_model, 0], "images": [keyframe_save_nodes[i], 0], "multiplier": frames_per_keyframe}}
                interpolated.append(n_rife)

        # Video output — use VHS_VideoCombine with VAEDecode output
        if vae_decode_nodes:
            n_video = nid()
            workflow[n_video] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": [vae_decode_nodes[0], 0],
                    "frame_rate": output_fps,
                    "loop_count": 0,
                    "filename_prefix": "music_video",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,
                    "save_output": True,
                    "videopreview": {"hidden": True},
                }
            }

        # Submit to ComfyUI
        client_id = str(uuid.uuid4())
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        code, resp_body, _ = _comfy_request(comfy_base, "POST", "/prompt", data=payload, timeout=30)

        if code != 200:
            return JSONResponse({"error": f"ComfyUI error {code}: {resp_body.decode()[:300]}"}, status_code=500)

        result = json.loads(resp_body)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return JSONResponse({"error": f"No prompt_id: {result}"}, status_code=500)

        with results_lock:
            results_store[prompt_id] = {
                "status": "running",
                "images": [],
                "submitted_at": time.time(),
                "prompt": prompt,
                "settings": {"width": width, "height": height, "num_keyframes": num_keyframes,
                             "frames_per_keyframe": frames_per_keyframe, "output_fps": output_fps},
            }

        return JSONResponse({
            "status": "accepted",
            "prompt_id": prompt_id,
            "poll_url": f"/result/{prompt_id}",
        })

    async def list_models(request):
        if not comfy_ready.is_set():
            return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)
        models = {}
        for model_dir in comfy_models.iterdir():
            if not model_dir.is_dir():
                continue
            files = []
            for f in model_dir.rglob("*"):
                if f.is_file() and f.suffix in (".safetensors", ".ckpt", ".pt", ".bin", ".gguf", ".pth"):
                    files.append({"name": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 1)})
            if files:
                models[model_dir.name] = sorted(files, key=lambda x: x["name"])
        return JSONResponse(models)

    async def proxy(request):
        if not comfy_ready.is_set():
            return JSONResponse({"error": "ComfyUI not ready"}, status_code=503)
        target_url = f"{comfy_base}{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept", "authorization")}
        async with httpx.AsyncClient(timeout=600) as client:
            try:
                resp = await client.request(method=request.method, url=target_url, content=body, headers=headers)
                return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
            except httpx.TimeoutException:
                return JSONResponse({"error": "Upstream timeout"}, status_code=504)
            except Exception as e:
                return JSONResponse({"error": f"Bad Gateway: {e}"}, status_code=502)

    starlette_app = Starlette(routes=[
        Route("/generate", generate, methods=["POST"]),
        Route("/generate-video", generate_video, methods=["POST"]),
        Route("/result/{id}", get_result),
        Route("/models", list_models),
        Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
    ])

    return starlette_app
