#!/usr/bin/env python3
"""
extract_frames.py — Convert videos into sampled frames for training/testing.
Example: 3 frames per minute (~0.05 fps)
"""

import argparse
import subprocess
from pathlib import Path

def extract_frames(video_path, output_dir, fps=0.05):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-r", str(fps), "-qscale:v", "2",
        out_pattern
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video file or folder")
    parser.add_argument("--output", required=True, help="Output frame folder")
    parser.add_argument("--fps", type=float, default=0.05, help="Sampling rate (frames/sec)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        for video in input_path.glob("*.mp4"):
            extract_frames(video, Path(args.output) / video.stem, args.fps)
    else:
        extract_frames(input_path, Path(args.output), args.fps)
