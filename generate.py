#!/usr/bin/env python3
"""
ComfyUI Headless Client — Generate images and save to local disk.

Usage:
    python generate.py "a beautiful sunset over mountains"
    python generate.py "cyberpunk city" --width 1024 --height 1024 --steps 30
    python generate.py "test" --output ./my-images --seed 42
    python generate.py --list-models
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

MODAL_URL = "https://robertmcasper--comfyui-headless-serve.modal.run"


def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{MODAL_URL}{path}", data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)


def get(path):
    req = urllib.request.Request(f"{MODAL_URL}{path}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        sys.exit(1)


def generate(prompt, output_dir="./output", width=1024, height=1024, steps=20, cfg=7.0, seed=None):
    if seed is None:
        seed = int(time.time()) % 2**32

    print(f"Submitting: '{prompt[:60]}' ({width}x{height}, {steps} steps, seed={seed})")

    resp = post("/generate", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
    })

    if "error" in resp:
        print(f"Error: {resp['error']}", file=sys.stderr)
        sys.exit(1)

    prompt_id = resp["prompt_id"]
    print(f"Accepted. prompt_id={prompt_id}")

    # Poll
    while True:
        time.sleep(5)
        result = get(f"/result/{prompt_id}")
        status = result.get("status")

        if status == "running":
            elapsed = result.get("elapsed_seconds", 0)
            print(f"\r  Running... {elapsed:.0f}s", end="", flush=True)
            continue

        if status == "done":
            print(f"\r  Done! Saving to {output_dir}/")
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for img in result.get("images", []):
                fn = img["filename"]
                if "base64" in img:
                    data = base64.b64decode(img["base64"])
                    path = out / fn
                    path.write_bytes(data)
                    print(f"  {fn} ({len(data)/1024:.0f} KB) -> {path}")
                elif "error" in img:
                    print(f"  {fn}: ERROR - {img['error']}")
            return result

        print(f"Unexpected status: {status}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate images with ComfyUI on Modal")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--output", "-o", default="./output")
    parser.add_argument("--url", default=None, help="Modal app URL")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()
    if args.url:
        global MODAL_URL
        MODAL_URL = args.url.rstrip("/")

    if args.list_models:
        models = get("/models")
        for cat, files in sorted(models.items()):
            total = sum(f["size_mb"] for f in files)
            print(f"\n{cat}/ ({len(files)} files, {total/1024:.1f} GB)")
            for f in files:
                print(f"  {f['name']} ({f['size_mb']:.0f} MB)")
        return

    if not args.prompt:
        parser.error("Please provide a prompt or use --list-models")

    generate(args.prompt, args.output, args.width, args.height, args.steps, args.cfg, args.seed)


if __name__ == "__main__":
    main()
