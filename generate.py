#!/usr/bin/env python3
"""
ComfyUI Headless Client — Production Content Generation

Usage:
    # Single image
    python generate.py "a beautiful sunset"
    
    # Use a channel preset
    python generate.py "trap music video" --channel youtube-trap
    python generate.py "lofi hip hop" --channel youtube-lofi
    python generate.py "trending tiktok" --channel tiktok
    
    # Video
    python generate.py "landscape timelapse" --video --channel youtube-trap
    
    # Batch
    --batch prompt1 prompt2 prompt3
  
    # List channels/models
    python generate.py --list-channels
    python generate.py --list-models
"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
import urllib.request
import urllib.error
from pathlib import Path

MODAL_URL = "https://robertmcasper--comfyui-headless-serve.modal.run"


def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{MODAL_URL}{path}", data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)


def api_get(path):
    req = urllib.request.Request(f"{MODAL_URL}{path}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)


def poll_result(poll_url, timeout=600):
    """Poll for generation completion."""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(5)
        result = api_get(poll_url)
        status = result.get("status")

        if status == "running":
            elapsed = result.get("elapsed_seconds", 0)
            print(f"\r  Running... {elapsed:.0f}s", end="", flush=True)
            continue

        if status == "done":
            return result

        if status == "error":
            print(f"\nGeneration failed: {result.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

        print(f"\nUnexpected status: {status}")
        sys.exit(1)

    print(f"\nTimeout after {timeout}s", file=sys.stderr)
    sys.exit(1)


def health_check():
    """Check Modal/ComfyUI health and print GPU state."""
    req = urllib.request.Request(f"{MODAL_URL}/system_stats")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            print(f"Not ready: HTTP {e.code}")
            return False
    except Exception as e:
        print(f"Not ready: {e}")
        return False
    if "error" in data:
        print(f"Not ready: {data['error']}")
        return False
    device = data.get("devices", [{}])[0]
    system = data.get("system", {})
    print("Ready")
    print(f"  ComfyUI: {system.get('comfyui_version', 'unknown')}")
    print(f"  GPU: {device.get('name', 'unknown')}")
    if device.get("vram_free"):
        print(f"  VRAM free: {device['vram_free'] / 1024**3:.1f} GB")
    return True


def save_images(result, output_dir):
    """Save base64 images from result to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for img in result.get("images", []):
        fn = img["filename"]
        if "base64" in img:
            data = base64.b64decode(img["base64"])
            path = out / fn
            path.write_bytes(data)
            saved.append(str(path))
            print(f"  {fn} ({len(data)/1024:.0f} KB) -> {path}")
        elif "error" in img:
            print(f"  {fn}: ERROR - {img['error']}")
    return saved


def save_manifest(result, saved_files, output_dir, request_payload=None):
    """Write a reproducibility manifest next to outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modal_url": MODAL_URL,
        "prompt_id": result.get("prompt_id"),
        "status": result.get("status"),
        "request": request_payload or {},
        "files": saved_files,
        "settings": result.get("settings", {}),
    }
    path = out / f"manifest_{result.get('prompt_id', int(time.time()))}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {path}")
    return str(path)


def generate_image(prompt, channel=None, output_dir="./output", width=None, height=None,
                   steps=None, cfg=None, seed=None, timeout=600):
    """Generate a single image."""
    payload = {"prompt": prompt}
    if width: payload["width"] = width
    if height: payload["height"] = height
    if steps: payload["steps"] = steps
    if cfg: payload["cfg"] = cfg
    if seed: payload["seed"] = seed
    if channel: payload["channel"] = channel

    print(f"Submitting: '{prompt[:70]}'")
    if channel:
        print(f"  Channel: {channel}")

    resp = api_post("/generate", payload)
    if "error" in resp:
        print(f"Error: {resp['error']}", file=sys.stderr)
        sys.exit(1)

    prompt_id = resp["prompt_id"]
    poll_url = resp["poll_url"]
    print(f"  prompt_id={prompt_id}")

    result = poll_result(poll_url, timeout=timeout)
    print(f"\r  Done! Saving to {output_dir}/")
    saved = save_images(result, output_dir)
    save_manifest(result, saved, output_dir, payload)
    return saved


def generate_video(prompt, channel=None, output_dir="./output", width=None, height=None,
                   steps=None, seed=None, num_keyframes=4, frames_per_keyframe=8, fps=24, timeout=1200):
    """Generate video with hyperframes."""
    # Try /generate-video endpoint first
    payload = {
        "prompt": prompt,
        "width": width or 512,
        "height": height or 512,
        "num_keyframes": num_keyframes,
        "frames_per_keyframe": frames_per_keyframe,
        "output_fps": fps,
        "steps": steps or 15,
        "seed": seed or -1,
        "channel": channel,
    }

    print(f"Submitting video: '{prompt[:70]}' ({num_keyframes} keyframes, {frames_per_keyframe} interp)")

    resp = api_post("/generate-video", payload)
    if "error" in resp:
        print(f"Error: {resp['error']}", file=sys.stderr)
        sys.exit(1)

    prompt_id = resp["prompt_id"]
    poll_url = resp["poll_url"]
    print(f"  prompt_id={prompt_id}")

    result = poll_result(poll_url, timeout=timeout)
    print(f"\r  Done! Saving to {output_dir}/")
    saved = save_images(result, output_dir)
    save_manifest(result, saved, output_dir, payload)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate images and videos with ComfyUI on Modal")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--output", "-o", default="./output")
    parser.add_argument("--url", default=None, help="Modal app URL")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--channel", default=None, help="Channel preset (youtube-trap, tiktok, etc.)")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--keyframes", type=int, default=4)
    parser.add_argument("--interp", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--list-channels", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--health", action="store_true", help="Check Modal/ComfyUI health and exit")
    parser.add_argument("--timeout", type=int, default=None, help="Polling timeout in seconds")

    args = parser.parse_args()

    global MODAL_URL
    if args.url:
        MODAL_URL = args.url.rstrip("/")

    if args.health:
        ok = health_check()
        sys.exit(0 if ok else 1)

    if args.list_channels:
        try:
            from channels import list_channels
            channels = list_channels()
            print("Available channels:")
            for name, desc in channels.items():
                print(f"  {name:20s} {desc}")
        except ImportError:
            print("channels.py not found")
        return

    if args.list_models:
        models = api_get("/models")
        for cat, files in sorted(models.items()):
            total = sum(f["size_mb"] for f in files)
            print(f"\n{cat}/ ({len(files)} files, {total/1024:.1f} GB)")
            for f in files:
                print(f"  {f['name']} ({f['size_mb']:.0f} MB)")
        return

    if not args.prompt:
        parser.error("Please provide a prompt or use --list-channels/--list-models")

    if args.video:
        generate_video(
            args.prompt, args.channel, args.output,
            args.width, args.height, args.steps, args.seed,
            args.keyframes, args.interp, args.fps, args.timeout or 1200,
        )
    else:
        generate_image(
            args.prompt, args.channel, args.output,
            args.width, args.height, args.steps, args.cfg, args.seed, args.timeout or 600,
        )


if __name__ == "__main__":
    main()
