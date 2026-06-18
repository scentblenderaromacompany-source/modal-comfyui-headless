#!/usr/bin/env python3
"""
encode_video.py - Combine keyframe images into video with FFmpeg crossfade transitions.

Usage:
    python encode_video.py --frames frame1.png frame2.png frame3.png --output video.mp4 --fps 24 --duration 2.0
    python encode_video.py --pattern "output/flux_*.png" --output video.mp4 --fps 24
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def encode_with_crossfade(frame_files, output_path, fps=24, transition_duration=1.0, hold_duration=2.0):
    """
    Combine images into video with crossfade transitions between each frame.
    
    Each frame is held for `hold_duration` seconds, then crossfades into the next
    frame over `transition_duration` seconds.
    
    Args:
        frame_files: List of image file paths (sorted)
        output_path: Output video path
        fps: Frames per second
        transition_duration: Crossfade duration in seconds
        hold_duration: How long to hold each frame before crossfading
    """
    if len(frame_files) < 2:
        print("Need at least 2 frames for crossfade", file=sys.stderr)
        return False
    
    frame_files = sorted(frame_files)
    print(f"Encoding {len(frame_files)} frames with crossfade:")
    for f in frame_files:
        print(f"  {f}")
    
    # Build FFmpeg filter complex for crossfade chain
    # For N frames, we need N-1 xfade filters
    filter_parts = []
    
    # Scale all inputs to same size and set fps
    for i, _ in enumerate(frame_files):
        filter_parts.append(f"[{i}:v]scale='trunc(iw/2)*2':'trunc(ih/2)*2',setsar=1,fps={fps},format=yuv420p[v{i}];")
    
    # Chain xfade transitions
    # [v0][v1]xfade=transition=fade:duration=D:offset=O[out0]
    # [out0][v2]xfade=transition=fade:duration=D:offset=O[out1]
    # etc.
    offset = hold_duration
    for i in range(len(frame_files) - 1):
        input_a = f"[v{i}]" if i == 0 else f"[out{i-1}]"
        input_b = f"[v{i+1}]"
        output = f"[out{i}]" if i < len(frame_files) - 2 else "[out]"
        
        filter_parts.append(
            f"{input_a}{input_b}xfade=transition=fade:duration={transition_duration}:offset={offset}{output};"
        )
        offset += hold_duration
    
    filter_complex = "".join(filter_parts)
    # Remove trailing semicolon
    filter_complex = filter_complex.rstrip(";")
    
    # Build command
    cmd = ["ffmpeg", "-y"]
    for f in frame_files:
        cmd.extend(["-loop", "1", "-t", str(hold_duration + transition_duration), "-i", f])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "medium",
        "-crf", "18",
        str(output_path)
    ])
    
    print(f"\nFFmpeg command: {' '.join(cmd[:10])}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[:500]}", file=sys.stderr)
        return False
    
    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"\nVideo encoded: {output_path} ({size_mb:.1f} MB)")
        return True
    else:
        print("Video file not created", file=sys.stderr)
        return False


def encode_simple(frame_files, output_path, fps=24, duration_per_frame=2.0):
    """
    Simple encoding: each frame held for a fixed duration, no transitions.
    Uses concat demuxer.
    """
    if not frame_files:
        print("No frames provided", file=sys.stderr)
        return False
    
    frame_files = sorted(frame_files)
    print(f"Encoding {len(frame_files)} frames (simple, {duration_per_frame}s each):")
    for f in frame_files:
        print(f"  {f}")
    
    # Create concat file list
    concat_file = Path(output_path).parent / "concat_list.txt"
    with open(concat_file, "w") as f:
        for frame in frame_files:
            f.write(f"file '{Path(frame).absolute()}'\n")
            f.write(f"duration {duration_per_frame}\n")
        # FFmpeg requires the last frame to be listed again without duration
        f.write(f"file '{Path(frame_files[-1]).absolute()}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "medium",
        "-crf", "18",
        "-r", str(fps),
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_file.unlink(missing_ok=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[:500]}", file=sys.stderr)
        return False
    
    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"Video encoded: {output_path} ({size_mb:.1f} MB)")
        return True
    return False


def upscale_frames(frame_files, scale_factor, model_path):
    """Upscale images. Falls back to FFmpeg lanczos if model not available."""
    if not Path(model_path).exists():
        print(f"  Upscale model not found, using FFmpeg lanczos")
        return _upscale_ffmpeg(frame_files, scale_factor)
    return _upscale_ffmpeg(frame_files, scale_factor)


def _upscale_ffmpeg(frame_files, scale_factor):
    """Upscale using FFmpeg lanczos scaling."""
    upscaled = []
    for f in frame_files:
        out = str(Path(f).with_suffix("")) + f"_x{int(scale_factor)}{Path(f).suffix}"
        cmd = ["ffmpeg", "-y", "-i", f, "-vf", f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=lanczos", out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and Path(out).exists():
            upscaled.append(out)
        else:
            upscaled.append(f)
    return upscaled


def main():
    parser = argparse.ArgumentParser(description="Encode video from keyframe images")
    parser.add_argument("--frames", nargs="+", help="Frame image files")
    parser.add_argument("--pattern", default=None, help="Glob pattern for frames (e.g., 'output/*.png')")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration per frame in seconds (simple mode)")
    parser.add_argument("--transition", type=float, default=1.0, help="Crossfade duration in seconds")
    parser.add_argument("--crossfade", action="store_true", help="Use crossfade transitions")
    parser.add_argument("--audio", default=None, help="Audio file to add")
    parser.add_argument("--upscale", type=float, default=None, help="Upscale factor (e.g., 2.0 for 2x)")
    parser.add_argument("--upscale-model", default="/tmp/4x-UltraSharp.pth", help="Path to upscale model (.pth)")
    
    args = parser.parse_args()
    
    # Get frame files
    if args.frames:
        frame_files = args.frames
    elif args.pattern:
        frame_files = sorted(glob.glob(args.pattern))
    else:
        print("Provide --frames or --pattern", file=sys.stderr)
        sys.exit(1)
    
    frame_files = [f for f in frame_files if Path(f).exists()]
    if not frame_files:
        print("No frame files found", file=sys.stderr)
        sys.exit(1)
    
    # Optional: upscale frames first
    if args.upscale and args.upscale > 1.0:
        print(f"Upscaling frames {args.upscale}x...")
        frame_files = upscale_frames(frame_files, args.upscale, args.upscale_model)
    
    # Encode
    if args.crossfade:
        success = encode_with_crossfade(frame_files, args.output, args.fps, args.transition, args.duration)
    else:
        success = encode_simple(frame_files, args.output, args.fps, args.duration)
    
    # Add audio if provided
    if success and args.audio and Path(args.audio).exists():
        temp_output = args.output + ".temp.mp4"
        os.rename(args.output, temp_output)
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-i", args.audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            args.output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(temp_output) if Path(temp_output).exists() else None
        if result.returncode == 0:
            print(f"Audio added: {args.audio}")
        else:
            print(f"Audio encoding failed: {result.stderr[:200]}")
            if Path(temp_output).exists():
                os.rename(temp_output, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
