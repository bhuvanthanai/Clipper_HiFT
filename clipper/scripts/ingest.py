"""
ingest.py - Ingest video from YouTube URL or local file for Universal Clipper.
Merges best video (up to 1080p) + best audio into MP4 container.
Applies retry logic and custom extractor args to bypass 403 / throttling errors.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FFMPEG_CANDIDATE_PATHS = [
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    "ffmpeg",
]

FFPROBE_CANDIDATE_PATHS = [
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe",
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe",
    "ffprobe",
]


def resolve_binary(candidates, name="ffmpeg"):
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return name


FFMPEG_BIN = resolve_binary(FFMPEG_CANDIDATE_PATHS, "ffmpeg")
FFPROBE_BIN = resolve_binary(FFPROBE_CANDIDATE_PATHS, "ffprobe")


def get_video_metadata(file_path: str) -> dict:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe = json.loads(result.stdout)
    
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    format_info = probe.get("format", {})
    
    duration = float(format_info.get("duration", 0))
    if duration == 0 and video_stream:
        duration = float(video_stream.get("duration", 0))
        
    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0
    
    # Calculate FPS
    fps = 30.0
    if video_stream:
        r_frame_rate = video_stream.get("r_frame_rate", "30/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(r_frame_rate)
            
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "format": format_info.get("format_name", "mp4"),
        "size_bytes": int(format_info.get("size", os.path.getsize(file_path))),
    }


def download_youtube(url: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(title).80s [%(id)s].%(ext)s")
    
    # yt-dlp arguments to bypass 403 throttling, enforce 1080p max MP4
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ffmpeg-location",
        os.path.dirname(FFMPEG_BIN) if os.path.isabs(FFMPEG_BIN) else "ffmpeg",
        "--extractor-args",
        "youtube:player_client=android,web",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format",
        "mp4",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--retry-sleep",
        "linear=1::2",
        "--no-check-certificates",
        "--no-playlist",
        "--print-json",
        "-o",
        out_template,
        url,
    ]
    
    print(f"[ingest] Fetching media from YouTube URL: {url}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    
    last_json_line = None
    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean.startswith("{") and line_clean.endswith("}"):
            last_json_line = line_clean
            
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (code {proc.returncode}): {stderr}")
        
    if not last_json_line:
        # Fallback: search output_dir for most recent mp4
        mp4s = sorted(Path(output_dir).glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if not mp4s:
            raise RuntimeError(f"yt-dlp completed but no MP4 found in {output_dir}")
        downloaded_file = str(mp4s[0])
        title = mp4s[0].stem
    else:
        info = json.loads(last_json_line)
        downloaded_file = info.get("_filename") or info.get("filename")
        if not downloaded_file or not os.path.exists(downloaded_file):
            # Check requested_downloads
            req = info.get("requested_downloads")
            if req and len(req) > 0 and os.path.exists(req[0].get("filepath", "")):
                downloaded_file = req[0]["filepath"]
            else:
                # Find matching files
                video_id = info.get("id")
                matches = list(Path(output_dir).glob(f"*{video_id}*.mp4"))
                if matches:
                    downloaded_file = str(matches[0])
                else:
                    raise FileNotFoundError(f"Cannot resolve downloaded file path for video ID {video_id}")
        title = info.get("title", Path(downloaded_file).stem)
        
    meta = get_video_metadata(downloaded_file)
    meta["title"] = title
    meta["file_path"] = os.path.abspath(downloaded_file)
    meta["source_type"] = "youtube"
    return meta


def ingest_local_file(input_path: str, output_dir: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Local file does not exist: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    input_file = Path(input_path)
    meta = get_video_metadata(str(input_file))
    
    # Destination in temp_work
    dest_path = os.path.join(output_dir, f"{input_file.stem}_ingested.mp4")
    
    # If already mp4 with valid streams, copy or remux
    print(f"[ingest] Normalizing local file into container: {dest_path}", file=sys.stderr)
    remux_cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_file),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        dest_path,
    ]
    try:
        subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        # Fallback to re-encode if copy fails
        reencode_cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            str(input_file),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            dest_path,
        ]
        subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
    new_meta = get_video_metadata(dest_path)
    new_meta["title"] = input_file.stem
    new_meta["file_path"] = os.path.abspath(dest_path)
    new_meta["source_type"] = "local"
    return new_meta


def main():
    parser = argparse.ArgumentParser(description="Ingest video from URL or local path for clipping.")
    parser.add_argument("--input", "-i", required=True, help="YouTube URL or local video file path")
    parser.add_argument("--output-dir", "-o", default="./temp_work", help="Output staging directory")
    parser.add_argument("--json", action="store_true", help="Output metadata as JSON to stdout")
    args = parser.parse_args()
    
    input_val = args.input.strip()
    is_url = input_val.startswith("http://") or input_val.startswith("https://") or "youtube.com" in input_val or "youtu.be" in input_val
    
    try:
        if is_url:
            result = download_youtube(input_val, args.output_dir)
        else:
            result = ingest_local_file(input_val, args.output_dir)
            
        result["status"] = "success"
        
        # Save metadata JSON into staging dir
        meta_json_path = os.path.join(args.output_dir, "ingest_metadata.json")
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        result["metadata_file"] = os.path.abspath(meta_json_path)
        
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
