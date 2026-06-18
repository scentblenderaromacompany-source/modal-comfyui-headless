#!/usr/bin/env python3
"""
trap_music_video_producer.py

Production-grade music video visualizer for YouTube rap/trap channels.
Generates high-quality video content by chaining multiple AI models and post-processing steps.

Usage:
    python trap_music_video_producer.py --audio track.mp3 --style dark-trap --duration 180
    python trap_music_video_producer.py --style dark-trap --keyframes 6 --manual
    python trap_music_video_producer.py --list-styles
    python trap_music_video_producer.py --list-channels
    
Environment:
    MODAL_URL - Modal app URL (default: https://robertmcasper--comfyui-headless-serve.modal.run)
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

MODAL_URL = os.environ.get("MODAL_URL", "https://robertmcasper--comfyui-headless-serve.modal.run").rstrip("/")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TRAP MUSIC VIDEO STYLE PRESETS
# ============================================================================

TRAP_STYLES = {
    "dark-trap": {
        "name": "Dark Trap",
        "description": "Dark, cinematic trap aesthetic — neon lights, urban decay, luxury",
        "resolution": (1920, 1080),
        "fps": 24,
        "steps": 25,
        "cfg": 7.5,
        "num_keyframes": 6,
        "frames_per_keyframe": 12,
        "prompt_prefix": "cinematic dark trap music video, ",
        "prompt_suffix": ", neon lights, urban night scene, luxury cars, volumetric fog, dramatic lighting, 8k detail, film grain",
        "negative_prompt": "blurry, low quality, bright, cheerful, soft, pastel, anime, cartoon, text, watermark, deformed",
        "loras": [],
        "sampler": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "lora_strength": 0.75,
        "post_process": ["upscale_2x", "sharpen", "color_grade"],
    },

    "neon-trap": {
        "name": "Neon Trap",
        "description": "Vibrant neon cyberpunk trap — synthwave colors, futuristic city",
        "resolution": (1920, 1080),
        "fps": 24,
        "steps": 20,
        "cfg": 7.0,
        "num_keyframes": 8,
        "frames_per_keyframe": 8,
        "prompt_prefix": "neon cyberpunk trap music video, ",
        "prompt_suffix": ", vibrant neon colors, futuristic cityscape, synthwave aesthetic, glowing lights, rain reflections, 8k",
        "negative_prompt": "blurry, low quality, dark, muted, grayscale, text, watermark",
        "loras": [],
        "sampler": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "lora_strength": 0.7,
        "post_process": ["upscale_2x", "color_grade_neon"],
    },

    "luxury-trap": {
        "name": "Luxury Trap",
        "description": "High-end luxury aesthetic — mansions, cars, jewelry, money",
        "resolution": (1920, 1080),
        "fps": 24,
        "steps": 30,
        "cfg": 8.0,
        "num_keyframes": 4,
        "frames_per_keyframe": 16,
        "prompt_prefix": "luxury trap music video, high-end lifestyle, ",
        "prompt_suffix": ", mansion, expensive cars, gold jewelry, champagne, city skyline at night, cinematic lighting, 8k",
        "negative_prompt": "blurry, low quality, cheap, poor, dirty, text, watermark",
        "loras": [],
        "sampler": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "lora_strength": 0.8,
        "post_process": ["upscale_2x", "sharpen"],
    },

    "horror-trap": {
        "name": "Horror Trap",
        "description": "Dark horror-inspired trap — haunted, eerie, intense",
        "resolution": (1920, 1080),
        "fps": 24,
        "steps": 25,
        "cfg": 7.5,
        "num_keyframes": 5,
        "frames_per_keyframe": 10,
        "prompt_prefix": "horror trap music video, dark and eerie, ",
        "prompt_suffix": ", haunted mansion, fog, dim lighting, candles, shadows, gothic architecture, cinematic, 8k",
        "negative_prompt": "blurry, low quality, bright, sunny, happy, cartoon, text, watermark",
        "loras": [],
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 1.0,
        "lora_strength": 0.75,
        "post_process": ["upscale_2x", "color_grade_dark"],
    },

    "drill-trap": {
        "name": "Drill Trap",
        "description": "UK drill-inspired aesthetic — dark streets, moody, intense",
        "resolution": (1920, 1080),
        "fps": 24,
        "steps": 25,
        "cfg": 7.5,
        "num_keyframes": 6,
        "frames_per_keyframe": 12,
        "prompt_prefix": "drill music video, dark urban aesthetic, ",
        "prompt_suffix": ", dark alleyways, street lights, moody atmosphere, fog, cinematic composition, 8k detail",
        "negative_prompt": "blurry, low quality, bright, cheerful, colorful, text, watermark",
        "loras": [],
        "sampler": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "lora_strength": 0.75,
        "post_process": ["upscale_2x", "color_grade_moody"],
    },
}

# Channel-specific output configs
CHANNEL_CONFIGS = {
    "youtube": {
        "description": "YouTube video — 16:9, high quality",
        "formats": ["mp4"],
        "codec": "libx264",
        "crf": 18,
        "preset": "slow",
        "audio_codec": "aac",
        "audio_bitrate": "320k",
    },
    "tiktok": {
        "description": "TikTok/Reels — 9:16 vertical, fast",
        "formats": ["mp4"],
        "codec": "libx264",
        "crf": 22,
        "preset": "medium",
        "audio_codec": "aac",
        "audio_bitrate": "128k",
    },
    "instagram": {
        "description": "Instagram — 1:1 or 4:5",
        "formats": ["mp4"],
        "codec": "libx264",
        "crf": 20,
        "preset": "medium",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    },
}


# ============================================================================
# API CLIENT
# ============================================================================

class ModalClient:
    """Client for the Modal ComfyUI headless API."""
    
    def __init__(self, url=None):
        self.url = (url or MODAL_URL).rstrip("/")
    
    def post(self, path, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(f"{self.url}{path}", data=body,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    
    def get(self, path):
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    
    def submit_image(self, prompt, width=1024, height=1024, steps=20, cfg=7.0, seed=-1, loras=None):
        """Submit image generation request."""
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "seed": seed if seed >= 0 else int(time.time()) % 2**32,
        }
        if loras:
            payload["loras"] = loras
        return self.post("/generate", payload)
    
    def submit_video(self, prompt, width=1024, height=1024, num_keyframes=4,
                     frames_per_keyframe=8, fps=24, steps=15, cfg=7.0, seed=-1, loras=None):
        """Submit video generation request."""
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_keyframes": num_keyframes,
            "frames_per_keyframe": frames_per_keyframe,
            "output_fps": fps,
            "steps": steps,
            "cfg": cfg,
            "seed": seed if seed >= 0 else int(time.time()) % 2**32,
        }
        if loras:
            payload["loras"] = loras
        return self.post("/generate-video", payload)
    
    def poll_result(self, prompt_id, timeout=600):
        """Poll for generation completion."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(5)
            result = self.get(f"/result/{prompt_id}")
            status = result.get("status")
            
            if status == "running":
                elapsed = result.get("elapsed_seconds", 0)
                print(f"\r  Running... {elapsed:.0f}s", end="", flush=True)
                continue
            
            if status == "done":
                return result
            
            if status == "error":
                raise RuntimeError(f"Generation failed: {result.get('error', 'unknown')}")
            
            raise RuntimeError(f"Unexpected status: {status}")
        
        raise TimeoutError(f"Generation timed out after {timeout}s")
    
    def download_files(self, result, output_dir="./output"):
        """Download all files from result to local directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        
        for f in result.get("images", []):
            fn = f["filename"]
            if "base64" in f and f.get("size_bytes", 0) > 0:
                data = base64.b64decode(f["base64"])
                path = out / fn
                path.write_bytes(data)
                saved.append(str(path))
                print(f"  {fn} ({len(data)/1024:.0f} KB) -> {path}")
        
        return saved


# ============================================================================
# MUSIC VIDEO PRODUCER
# ============================================================================

class TrapMusicVideoProducer:
    """Production music video generator for trap/rap channels."""
    
    def __init__(self, style="dark-trap", channel="youtube"):
        self.style = TRAP_STYLES.get(style, TRAP_STYLES["dark-trap"])
        self.channel = CHANNEL_CONFIGS.get(channel, CHANNEL_CONFIGS["youtube"])
        self.client = ModalClient()
    
    def generate_music_video(self, audio_path=None, custom_prompts=None, duration_seconds=180,
                             bpm=140, seed=-1, output_dir="./output"):
        """
        Generate a complete music video.
        
        Pipeline:
          1. Analyze audio (BPM, energy levels) or use manual prompts
          2. Generate keyframes with FLUX using style preset
          3. Interpolate between keyframes (if RIFE available)
          4. Post-process (upscale, color grade)
          5. Encode final video with audio
        
        Args:
            audio_path: Path to audio file (optional)
            custom_prompts: List of custom prompts per keyframe (optional)
            duration_seconds: Target video duration
            bpm: Beats per minute (for sync timing)
            seed: Random seed (-1 for random)
            output_dir: Output directory
        
        Returns:
            dict with paths to generated files
        """
        print(f"\n{'='*60}")
        print(f"  Trap Music Video Producer")
        print(f"  Style: {self.style['name']}")
        print(f"  Channel: {self.channel['description']}")
        print(f"  Resolution: {self.style['resolution'][0]}x{self.style['resolution'][1]}")
        print(f"  Keyframes: {self.style['num_keyframes']} @ {self.style['steps']} steps")
        print(f"{'='*60}\n")
        
        # Step 1: Generate prompts for each keyframe
        if custom_prompts:
            prompts = custom_prompts
        else:
            prompts = self._generate_prompts(bpm, duration_seconds)
        
        print(f"Generated {len(prompts)} keyframe prompts:")
        for i, p in enumerate(prompts):
            print(f"  [{i+1}] {p[:80]}...")
        
        # Step 2: Submit video generation
        width, height = self.style["resolution"]
        print(f"\nSubmitting to Modal ({width}x{height})...")
        
        result = self.client.submit_video(
            prompt=self.style["prompt_prefix"] + prompts[0] + self.style["prompt_suffix"],
            width=width,
            height=height,
            num_keyframes=self.style["num_keyframes"],
            frames_per_keyframe=self.style["frames_per_keyframe"],
            fps=self.style["fps"],
            steps=self.style["steps"],
            cfg=self.style["cfg"],
            seed=seed,
            loras=self.style.get("loras", []),
        )
        
        prompt_id = result["prompt_id"]
        print(f"Accepted. prompt_id={prompt_id}")
        
        # Step 3: Poll for completion
        print("Polling for completion...")
        result = self.client.poll_result(prompt_id, timeout=1200)
        
        # Step 4: Download files
        print(f"\nDownloading files to {output_dir}/")
        saved_files = self.client.download_files(result, output_dir)
        
        # Step 5: Post-process (client-side FFmpeg)
        video_path = None
        if saved_files:
            video_path = self._post_process(saved_files, audio_path, output_dir)
        
        print(f"\n{'='*60}")
        print(f"  Complete! {len(saved_files)} files generated")
        if video_path:
            print(f"  Video: {video_path}")
        print(f"{'='*60}\n")
        
        return {
            "files": saved_files,
            "video": video_path,
            "prompt_id": prompt_id,
            "style": self.style["name"],
            "channel": self.channel["description"],
        }
    
    def _generate_prompts(self, bpm, duration_seconds):
        """Generate keyframe prompts based on BPM and duration."""
        beat_interval = 60.0 / bpm
        total_beats = int(duration_seconds / beat_interval)
        keyframe_interval = max(1, total_beats // self.style["num_keyframes"])
        
        # Trap music video scene progression
        scene_templates = [
            "opening shot, establishing scene, dark atmosphere, cinematic",
            "building energy, neon lights emerging, urban landscape",
            "verse section, street scene, dramatic shadows, moody",
            "chorus peak, intense colors, dynamic movement, explosive energy",
            "bridge transition, abstract visuals, smoke and mirrors, ethereal",
            "final verse, powerful imagery, gold and black, luxury aesthetic",
            "outro, fading lights, city skyline at night, reflective mood",
        ]
        
        prompts = []
        for i in range(self.style["num_keyframes"]):
            template_idx = min(i, len(scene_templates) - 1)
            scene = scene_templates[template_idx]
            prompts.append(f"{self.style['prompt_prefix']}{scene}{self.style['prompt_suffix']}")
        
        return prompts
    
    def _post_process(self, image_files, audio_path, output_dir):
        """Post-process: upscale, color grade, encode video with audio."""
        if not image_files:
            return None
        
        out = Path(output_dir)
        
        # Use FFmpeg to create video from frames
        # Find the pattern in filenames
        frame_files = sorted([f for f in image_files if f.endswith(".png")])
        if not frame_files:
            return None
        
        # Create video with FFmpeg
        video_name = f"trap_music_video_{int(time.time())}.mp4"
        video_path = str(out / video_name)
        
        # Build FFmpeg command
        # Assume frames are named keyframe_NNN_00001_.png
        first_frame = Path(frame_files[0])
        parent = first_frame.parent
        
        # Try to find the pattern
        import glob
        frame_pattern = str(parent / "keyframe_*_00001_.png")
        matching = sorted(glob.glob(frame_pattern))
        
        if matching:
            # Create a file list for FFmpeg concat
            list_file = out / "frame_list.txt"
            with open(list_file, "w") as lf:
                for f in matching:
                    lf.write(f"file '{f}'\n")
                    lf.write(f"duration {1.0/self.style['fps']:.4f}\n")
            
            ffmpeg_cmd = (
                f"ffmpeg -y -f concat -safe 0 -i {list_file} "
                f"-c:v {self.channel['codec']} -crf {self.channel['crf']} "
                f"-preset {self.channel['preset']} -pix_fmt yuv420p "
                f"-movflags +faststart "
            )
            
            if audio_path and Path(audio_path).exists():
                ffmpeg_cmd += (
                    f"-i {audio_path} -c:a {self.channel['audio_codec']} "
                    f"-b:a {self.channel['audio_bitrate']} -shortest "
                )
            
            ffmpeg_cmd += video_path
            
            print(f"\nEncoding video with FFmpeg...")
            print(f"  Command: {ffmpeg_cmd[:100]}...")
            
            import subprocess
            result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and Path(video_path).exists():
                size_mb = Path(video_path).stat().st_size / 1024 / 1024
                print(f"  Video encoded: {video_path} ({size_mb:.1f} MB)")
                return video_path
            else:
                print(f"  FFmpeg failed: {result.stderr[:200]}")
                return None
        
        return None
    
    def generate_album_art(self, prompt, seed=-1, output_dir="./output"):
        """Generate album cover art."""
        width, height = 3000, 3000
        
        full_prompt = f"{self.style['prompt_prefix']}{prompt}{self.style['prompt_suffix']}, album cover art, square format, high detail, print quality"
        
        result = self.client.submit_image(
            prompt=full_prompt,
            width=width,
            height=height,
            steps=30,
            cfg=7.5,
            seed=seed,
        )
        
        print(f"Album art submitted. prompt_id={result['prompt_id']}")
        result = self.client.poll_result(result["prompt_id"], timeout=600)
        saved = self.client.download_files(result, output_dir)
        return saved
    
    def batch_generate(self, prompts, output_dir="./output"):
        """Generate multiple images in batch."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
            
            full_prompt = f"{self.style['prompt_prefix']}{prompt}{self.style['prompt_suffix']}"
            width, height = self.style["resolution"]
            
            result = self.client.submit_image(
                prompt=full_prompt,
                width=width,
                height=height,
                steps=self.style["steps"],
                cfg=self.style["cfg"],
            )
            
            result = self.client.poll_result(result["prompt_id"], timeout=600)
            saved = self.client.download_files(result, output_dir)
            results.extend(saved)
        
        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trap Music Video Producer")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--style", default="dark-trap", choices=list(TRAP_STYLES.keys()),
                        help="Visual style preset")
    parser.add_argument("--channel", default="youtube", choices=list(CHANNEL_CONFIGS.keys()),
                        help="Output channel")
    parser.add_argument("--duration", type=int, default=180, help="Video duration in seconds")
    parser.add_argument("--bpm", type=int, default=140, help="Beats per minute")
    parser.add_argument("--keyframes", type=int, default=None, help="Override number of keyframes")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--url", default=None, help="Modal app URL")
    parser.add_argument("--list-styles", action="store_true", help="List available styles")
    parser.add_argument("--list-channels", action="store_true", help="List available channels")
    parser.add_argument("--album-art", action="store_true", help="Generate album art instead of video")
    parser.add_argument("--prompt", default=None, help="Custom prompt")
    
    args = parser.parse_args()
    
    if args.url:
        global MODAL_URL
        MODAL_URL = args.url.rstrip("/")
    
    if args.list_styles:
        print("\nAvailable trap styles:")
        for key, style in TRAP_STYLES.items():
            res = style['resolution']
            print(f"  {key:15s} {style['name']:15s} {res[0]}x{res[1]} — {style['description']}")
        return
    
    if args.list_channels:
        print("\nAvailable channels:")
        for key, ch in CHANNEL_CONFIGS.items():
            print(f"  {key:10s} {ch['description']}")
        return
    
    producer = TrapMusicVideoProducer(style=args.style, channel=args.channel)
    
    if args.keyframes:
        producer.style["num_keyframes"] = args.keyframes
    
    if args.album_art:
        prompt = args.prompt or "dark trap album cover, neon lights, gold chains"
        saved = producer.generate_album_art(prompt, args.seed, args.output)
        print(f"\nGenerated {len(saved)} album art image(s)")
    else:
        result = producer.generate_music_video(
            audio_path=args.audio,
            custom_prompts=[args.prompt] if args.prompt else None,
            duration_seconds=args.duration,
            bpm=args.bpm,
            seed=args.seed,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
