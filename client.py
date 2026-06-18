"""
Client for headless ComfyUI on Modal.

Usage:
  1. Deploy:  modal deploy comfyui_headless.py
  2. Get URL from Modal dashboard, set MODAL_URL env var
  3. Run:     python client.py

The client supports two modes:
  - HTTP proxy mode: forwards requests directly to ComfyUI's API
  - Modal .remote() mode: calls worker methods via Modal's RPC
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    print("pip install httpx")
    raise


# ---------------------------------------------------------------------------
# HTTP proxy client (talks to the Modal web_server)
# ---------------------------------------------------------------------------

class ComfyUIHTTPClient:
    """Talks to the Modal-deployed ComfyUI via HTTP proxy."""

    def __init__(self, base_url: str, timeout: float = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = httpx.get(f"{self.base}/system_stats", timeout=10)
        return r.json()

    def object_info(self) -> dict:
        r = httpx.get(f"{self.base}/object_info", timeout=30)
        return r.json()

    def queue(self) -> dict:
        r = httpx.get(f"{self.base}/queue", timeout=10)
        return r.json()

    def submit(self, workflow: dict, client_id: str = "modal-client") -> str:
        """Submit a workflow, return prompt_id."""
        r = httpx.post(
            f"{self.base}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        data = r.json()
        return data["prompt_id"]

    def history(self, prompt_id: str) -> dict:
        r = httpx.get(f"{self.base}/history/{prompt_id}", timeout=30)
        return r.json()

    def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        """Download an output image."""
        r = httpx.get(
            f"{self.base}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=60,
        )
        return r.content

    def run_and_wait(
        self,
        workflow: dict,
        poll_interval: float = 2.0,
        on_progress=None,
    ) -> dict:
        """Submit a workflow, poll until done, return outputs with image bytes."""
        prompt_id = self.submit(workflow)
        print(f"[client] Submitted prompt {prompt_id}")

        while True:
            hist = self.history(prompt_id)
            if prompt_id in hist:
                print(f"[client] Done!")
                return hist[prompt_id].get("outputs", {})
            if on_progress:
                on_progress()
            time.sleep(poll_interval)

    def run_and_download(
        self,
        workflow: dict,
        output_dir: str = "./outputs",
        poll_interval: float = 2.0,
    ) -> list[str]:
        """Submit workflow, wait, download all output images to disk."""
        outputs = self.run_and_wait(workflow, poll_interval)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                data = self.view(
                    img["filename"],
                    img.get("subfolder", ""),
                    img.get("type", "output"),
                )
                dest = out_path / img["filename"]
                dest.write_bytes(data)
                saved.append(str(dest))
                print(f"[client] Saved {dest} ({len(data)} bytes)")

        return saved


# ---------------------------------------------------------------------------
# Modal RPC client (calls worker methods via modal.Cls)
# ---------------------------------------------------------------------------

class ComfyUIModalClient:
    """Calls the Modal worker class directly via .remote()."""

    def __init__(self, app_name: str = "comfyui-headless", class_name: str = "ComfyUIWorker"):
        import modal
        cls = modal.Cls.from_name(app_name, class_name)
        self._worker = cls()

    def health(self) -> dict:
        return self._worker.health.remote()

    def object_info(self) -> dict:
        return self._worker.object_info.remote()

    def list_models(self) -> dict:
        return self._worker.list_models.remote()

    def run_prompt(self, workflow: dict) -> dict:
        """Returns {images: [{filename, data (base64), node_id}], videos: [...]}"""
        return self._worker.run_prompt.remote(workflow)


# ---------------------------------------------------------------------------
# Example: minimal text-to-image workflow (SD 1.5)
# ---------------------------------------------------------------------------

EXAMPLE_WORKFLOW_SD15 = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a beautiful landscape, mountains, sunset, highly detailed",
            "clip": ["4", 1],
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted",
            "clip": ["4", 1],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "modal_comfyui"},
    },
}


def main():
    parser = argparse.ArgumentParser(description="Headless ComfyUI on Modal client")
    parser.add_argument("--url", default=os.environ.get("MODAL_URL", ""), help="Modal app URL")
    parser.add_argument("--mode", choices=["http", "rpc"], default="http", help="Client mode")
    parser.add_argument("--action", choices=["health", "run", "models", "object_info"], default="health")
    parser.add_argument("--workflow", type=str, help="Path to workflow_api.json")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    if args.mode == "http":
        if not args.url:
            print("Set MODAL_URL env var or pass --url")
            sys.exit(1)
        client = ComfyUIHTTPClient(args.url)
    else:
        client = ComfyUIModalClient()

    if args.action == "health":
        print(json.dumps(client.health(), indent=2))

    elif args.action == "models":
        print(json.dumps(client.list_models(), indent=2))

    elif args.action == "object_info":
        info = client.object_info()
        print(f"Available nodes: {len(info)}")
        for name in sorted(info.keys())[:20]:
            print(f"  {name}")
        print("  ... (truncated)")

    elif args.action == "run":
        if args.workflow:
            workflow = json.loads(Path(args.workflow).read_text())
        else:
            print("No --workflow provided, using example SD 1.5 workflow")
            print("NOTE: You need to have the checkpoint in your models.py!")
            workflow = EXAMPLE_WORKFLOW_SD15

        if args.mode == "http":
            saved = client.run_and_download(workflow, args.output)
            print(f"\nSaved {len(saved)} images to {args.output}/")
        else:
            result = client.run_prompt(workflow)
            out_path = Path(args.output)
            out_path.mkdir(parents=True, exist_ok=True)
            for img in result.get("images", []):
                data = base64.b64decode(img["data"])
                dest = out_path / img["filename"]
                dest.write_bytes(data)
                print(f"Saved {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
