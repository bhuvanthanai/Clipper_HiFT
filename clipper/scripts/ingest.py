"""
ingest.py - Ingest video from YouTube URL or local file for Universal Clipper.
Produces two H.264 MP4 files on the same timeline:
- the main video (up to 1080p, best audio) used for rendering, and
- a 360p analysis copy (video only) used for crop-location analysis.
Prefers H.264 (avc1) YouTube formats and transcodes anything else (AV1, VP9, ...) to
H.264, so OpenCV builds without AV1/VP9 decoders can read both files.
Uses yt-dlp's default YouTube player clients (forcing android,web hits YouTube's
SABR-only experiment and caps downloads at 360p) with retry logic for throttling errors.
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

MAIN_FORMAT_SELECTOR = (
    "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]"
    "/bestvideo[height<=1080][vcodec^=avc1]+bestaudio"
    "/bestvideo[height<=1080]+bestaudio"
    "/best[height<=1080]/best"
)
ANALYSIS_FORMAT_SELECTOR = (
    "bestvideo[height<=360][vcodec^=avc1]"
    "/bestvideo[height<=360]"
    "/best[height<=360]"
)
ANALYSIS_SHORT_SIDE_PX = 360


def resolve_binary(candidates, name="ffmpeg"):
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return name


FFMPEG_BIN = resolve_binary(FFMPEG_CANDIDATE_PATHS, name="ffmpeg")
FFPROBE_BIN = resolve_binary(FFPROBE_CANDIDATE_PATHS, name="ffprobe")


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
        "video_codec": video_stream.get("codec_name", "") if video_stream else "",
        "format": format_info.get("format_name", "mp4"),
        "size_bytes": int(format_info.get("size", os.path.getsize(file_path))),
    }


def is_h264_video(file_path: str) -> bool:
    return get_video_metadata(file_path)["video_codec"] == "h264"


def run_ffmpeg(cmd: list) -> None:
    """Runs an ffmpeg command, raising RuntimeError with the tail of ffmpeg's stderr on failure."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    if result.returncode != 0:
        stderr_lines = result.stderr.strip().splitlines()
        error_lines = [line for line in stderr_lines if "error" in line.lower()]
        stderr_tail = "\n".join((error_lines or stderr_lines)[-5:])
        raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {stderr_tail}")


def transcode_to_h264(input_path: str, output_path: str, is_audio_kept: bool = True) -> str:
    """
    Re-encodes the video stream to H.264 (and audio to AAC, or drops it).
    ffmpeg's default AV1 decoder (libdav1d) fails on some clips cut mid-GOP, so an
    AV1 source that fails is retried once with the libaom-av1 decoder.
    """
    try:
        return transcode_to_h264_with_decoder(input_path, output_path, is_audio_kept=is_audio_kept, video_decoder=None)
    except RuntimeError as err:
        if get_video_metadata(input_path)["video_codec"] != "av1":
            raise
        print(f"[ingest] Default AV1 decoder failed ({err}); retrying with libaom-av1", file=sys.stderr)
        return transcode_to_h264_with_decoder(input_path, output_path, is_audio_kept=is_audio_kept, video_decoder="libaom-av1")


def transcode_to_h264_with_decoder(input_path: str, output_path: str, is_audio_kept: bool, video_decoder: str | None) -> str:
    cmd = [FFMPEG_BIN, "-y"]
    cmd += ["-c:v", video_decoder] if video_decoder else []
    cmd += [
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if is_audio_kept else ["-an"]
    cmd += ["-movflags", "+faststart", output_path]
    print(f"[ingest] Transcoding to H.264: {output_path}", file=sys.stderr)
    run_ffmpeg(cmd)
    return output_path


def get_h264_video_path(video_path: str, is_audio_kept: bool = True) -> str:
    """Returns video_path if it is already H.264, else transcodes it and deletes the original."""
    if is_h264_video(video_path):
        return video_path
    source = Path(video_path)
    h264_path = str(source.with_name(f"{source.stem}_h264.mp4"))
    transcode_to_h264(video_path, h264_path, is_audio_kept=is_audio_kept)
    os.remove(video_path)
    return h264_path


def build_analysis_copy(input_path: str, output_path: str) -> str:
    """Downscales input_path to a 360p (short side) H.264 video-only copy."""
    short_side = ANALYSIS_SHORT_SIDE_PX
    scale_filter = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        input_path,
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        output_path,
    ]
    print(f"[ingest] Building 360p analysis copy: {output_path}", file=sys.stderr)
    run_ffmpeg(cmd)
    return output_path


def download_with_yt_dlp(url: str, output_dir: str, format_selector: str, out_template: str, extra_args: list) -> tuple:
    """Downloads url with yt-dlp; returns (downloaded_file_path, title)."""
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ffmpeg-location",
        os.path.dirname(FFMPEG_BIN) if os.path.isabs(FFMPEG_BIN) else "ffmpeg",
        "-f",
        format_selector,
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
        *extra_args,
        "-o",
        out_template,
        url,
    ]

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
        return str(mp4s[0]), mp4s[0].stem

    info = json.loads(last_json_line)
    downloaded_file = info.get("_filename") or info.get("filename")
    if not downloaded_file or not os.path.exists(downloaded_file):
        req = info.get("requested_downloads")
        if req and len(req) > 0 and os.path.exists(req[0].get("filepath", "")):
            downloaded_file = req[0]["filepath"]
        else:
            raise FileNotFoundError(f"Cannot resolve downloaded file path for video ID {info.get('id')}")
    return downloaded_file, info.get("title", Path(downloaded_file).stem)


def get_download_section_args(download_section: str | None) -> list:
    """yt-dlp args that limit the download to a START-END time range (seconds or HH:MM:SS), if one is given."""
    if not download_section:
        return []
    return ["--download-sections", f"*{download_section}", "--force-keyframes-at-cuts"]


def download_youtube(url: str, output_dir: str, download_section: str | None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    section_args = get_download_section_args(download_section=download_section)

    print(f"[ingest] Fetching media from YouTube URL: {url}", file=sys.stderr)
    main_file, title = download_with_yt_dlp(
        url,
        output_dir,
        format_selector=MAIN_FORMAT_SELECTOR,
        out_template=os.path.join(output_dir, "%(title).80s [%(id)s].%(ext)s"),
        extra_args=["--write-info-json", *section_args],
    )
    info_json_path = str(Path(main_file).with_suffix(".info.json"))
    with open(info_json_path, "r", encoding="utf-8") as info_file:
        video_id = json.load(info_file).get("id")
    main_file = get_h264_video_path(main_file, is_audio_kept=True)

    print("[ingest] Fetching 360p analysis copy from YouTube", file=sys.stderr)
    try:
        analysis_file, _ = download_with_yt_dlp(
            url,
            output_dir,
            format_selector=ANALYSIS_FORMAT_SELECTOR,
            out_template=os.path.join(output_dir, "%(title).80s [%(id)s].360p.%(ext)s"),
            extra_args=section_args,
        )
        analysis_file = get_h264_video_path(analysis_file, is_audio_kept=False)
    except (RuntimeError, FileNotFoundError) as err:
        print(f"[ingest] 360p download failed ({err}); downscaling the main video instead", file=sys.stderr)
        analysis_file = build_analysis_copy(main_file, str(Path(main_file).with_name(f"{Path(main_file).stem}_360p.mp4")))

    meta = get_video_metadata(main_file)
    meta["title"] = title
    meta["file_path"] = os.path.abspath(main_file)
    meta["source_type"] = "youtube"
    meta["video_id"] = video_id
    meta["info_json_path"] = os.path.abspath(info_json_path)
    meta["download_section"] = download_section
    add_analysis_metadata(meta, analysis_file)
    return meta


def ingest_local_file(input_path: str, output_dir: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Local file does not exist: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    input_file = Path(input_path)

    # Destination in temp_work
    dest_path = os.path.join(output_dir, f"{input_file.stem}_ingested.mp4")

    if is_h264_video(str(input_file)):
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
            run_ffmpeg(remux_cmd)
        except RuntimeError:
            transcode_to_h264(str(input_file), dest_path, is_audio_kept=True)
    else:
        transcode_to_h264(str(input_file), dest_path, is_audio_kept=True)

    analysis_path = build_analysis_copy(dest_path, os.path.join(output_dir, f"{input_file.stem}_360p.mp4"))

    new_meta = get_video_metadata(dest_path)
    new_meta["title"] = input_file.stem
    new_meta["file_path"] = os.path.abspath(dest_path)
    new_meta["source_type"] = "local"
    add_analysis_metadata(new_meta, analysis_path)
    return new_meta


def add_analysis_metadata(meta: dict, analysis_path: str) -> dict:
    analysis_meta = get_video_metadata(analysis_path)
    meta["analysis_file_path"] = os.path.abspath(analysis_path)
    meta["analysis_width"] = analysis_meta["width"]
    meta["analysis_height"] = analysis_meta["height"]
    return meta


def main():
    parser = argparse.ArgumentParser(description="Ingest video from URL or local path for clipping.")
    parser.add_argument("--input", "-i", required=True, help="YouTube URL or local video file path")
    parser.add_argument("--output-dir", "-o", default="./temp_work", help="Output staging directory")
    parser.add_argument("--json", action="store_true", help="Output metadata as JSON to stdout")
    parser.add_argument("--download-section", default=None, help="YouTube only: download just START-END (seconds or HH:MM:SS), e.g. 3600-4080")
    args = parser.parse_args()

    input_val = args.input.strip()
    is_url = input_val.startswith("http://") or input_val.startswith("https://") or "youtube.com" in input_val or "youtu.be" in input_val

    try:
        if is_url:
            result = download_youtube(input_val, args.output_dir, download_section=args.download_section)
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
