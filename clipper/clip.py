"""
clip.py - End-to-end Universal Clipper: URL in, vertical streamer Shorts out.

Flow:
1. ingest.py          download the 1080p H.264 video (+ 360p analysis copy, + yt-dlp info JSON)
2. transcribe.py      word-level transcript
3. auto_clip.py       rank the best clip windows
4. facecam_detect.py  find the streamer facecam once (on frames of the 1080p video)
5. smart_crop.py      per clip: streamer layout (facecam top 1/3, stream crop bottom 2/3)
6. render_captions.py per clip: burn karaoke captions (skipped with --no-captions)

Results land in <output-root>/<video id>/: clip_01.mp4 ..., clips.json, facecam.json, run.json.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLIPPER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CLIPPER_DIR / "scripts"


def run_step(step_name: str, script_name: str, script_args: list) -> dict:
    """Runs one pipeline script, streams its log to stderr and returns its stdout JSON result."""
    print(f"[clip] ▶ {step_name}", file=sys.stderr)
    started_at = time.time()
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), *[str(a) for a in script_args]],
        stdout=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{step_name} failed ({script_name} exited {completed.returncode})")
    result = json.loads(completed.stdout)
    print(f"[clip] ✔ {step_name} ({time.time() - started_at:.1f}s)", file=sys.stderr)
    return result


def get_output_folder_name(ingest_result: dict) -> str:
    """The YouTube video id when known, otherwise a filesystem-safe version of the title."""
    if ingest_result.get("video_id"):
        return ingest_result["video_id"]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ingest_result.get("title") or "video").strip("_")[:80]


def run_clip_pipeline(
    source: str,
    output_root: Path,
    work_root: Path,
    max_clips: int,
    min_clip_seconds: float,
    max_clip_seconds: float,
    is_captioned: bool,
    facecam_index: int,
    whisper_model: str,
    download_section: str | None,
) -> dict:
    run_work_dir = work_root / time.strftime("run_%Y%m%d_%H%M%S")
    run_work_dir.mkdir(parents=True, exist_ok=True)

    ingest_args = ["--input", source, "--output-dir", run_work_dir]
    if download_section:
        ingest_args += ["--download-section", download_section]
    ingest_result = run_step(step_name="Ingest", script_name="ingest.py", script_args=ingest_args)
    source_video_path = ingest_result["file_path"]

    output_dir = output_root / get_output_folder_name(ingest_result=ingest_result)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_work_dir / "transcript.json"
    run_step(
        step_name="Transcribe",
        script_name="transcribe.py",
        script_args=["--input", source_video_path, "--model", whisper_model, "--output", transcript_path],
    )

    auto_clip_args = [
        "--video", source_video_path,
        "--transcription", transcript_path,
        "--ingest-metadata", ingest_result["metadata_file"],
        "--min-length", min_clip_seconds,
        "--max-length", max_clip_seconds,
        "--max-clips", max_clips,
        "--output", output_dir / "clips.json",
    ]
    if ingest_result.get("info_json_path") and not download_section:
        auto_clip_args += ["--info-json", ingest_result["info_json_path"]]
    elif download_section:
        print("[clip] Section download: skipping the heatmap signal (its times are for the full video).", file=sys.stderr)
    clips_result = run_step(step_name="Auto clip", script_name="auto_clip.py", script_args=auto_clip_args)
    if not clips_result["clips"]:
        raise RuntimeError("auto_clip.py found no clip windows")

    facecam_detection = run_step(
        step_name="Facecam detection",
        script_name="facecam_detect.py",
        script_args=["--input", source_video_path, "--facecam-index", facecam_index],
    )
    facecam_detection_path = output_dir / "facecam.json"
    facecam_detection_path.write_text(json.dumps(facecam_detection, indent=2), encoding="utf-8")

    rendered_clips = []
    for clip in clips_result["clips"]:
        clip_name = f"clip_{clip['rank']:02d}"
        cropped_path = run_work_dir / f"{clip_name}_cropped.mp4"
        crop_result = run_step(
            step_name=f"Crop {clip_name} ({clip['start']:.1f}-{clip['end']:.1f}s)",
            script_name="smart_crop.py",
            script_args=[
                "--input", source_video_path,
                "--output", cropped_path,
                "--start", clip["start"],
                "--end", clip["end"],
                "--mode", "streamer",
                "--facecam-detection", facecam_detection_path,
                "--fps", ingest_result.get("fps") or 30,
            ],
        )
        final_path = output_dir / f"{clip_name}.mp4"
        if is_captioned:
            run_step(
                step_name=f"Captions {clip_name}",
                script_name="render_captions.py",
                script_args=[
                    "--video", cropped_path,
                    "--transcription", transcript_path,
                    "--start", clip["start"],
                    "--end", clip["end"],
                    "--output", final_path,
                    "--fps", ingest_result.get("fps") or 30,
                ],
            )
        else:
            shutil.copyfile(cropped_path, final_path)
        rendered_clips.append({
            **clip,
            "file": str(final_path),
            "layout": crop_result.get("layout"),
        })

    run_summary = {
        "status": "success",
        "source": source,
        "title": ingest_result.get("title"),
        "output_dir": str(output_dir),
        "work_dir": str(run_work_dir),
        "signals_used": clips_result.get("signals_used"),
        "facecam": facecam_detection.get("facecam"),
        "stream_box": facecam_detection.get("stream_box"),
        "clips": rendered_clips,
    }
    (output_dir / "run.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def main():
    parser = argparse.ArgumentParser(description="URL in, vertical streamer Shorts out.")
    parser.add_argument("--url", "-u", required=True, help="YouTube URL (a local video path also works)")
    parser.add_argument("--output-root", default=str(CLIPPER_DIR / "output"), help="Clips go to <output-root>/<video id>/")
    parser.add_argument("--work-root", default=str(CLIPPER_DIR / "temp_work"), help="Intermediate files go to <work-root>/run_<timestamp>/")
    parser.add_argument("--max-clips", type=int, default=5, help="Number of clips to render (default 5)")
    parser.add_argument("--min-length", type=float, default=15.0, help="Minimum clip length in seconds")
    parser.add_argument("--max-length", type=float, default=60.0, help="Maximum clip length in seconds")
    parser.add_argument("--no-captions", action="store_true", help="Skip burning karaoke captions")
    parser.add_argument("--facecam-index", type=int, default=0, help="Which stable facecam to use, largest first (default 0)")
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model size (default base)")
    parser.add_argument("--download-section", default=None, help="Only download START-END of the video (seconds or HH:MM:SS)")
    args = parser.parse_args()

    try:
        run_summary = run_clip_pipeline(
            source=args.url,
            output_root=Path(args.output_root),
            work_root=Path(args.work_root),
            max_clips=args.max_clips,
            min_clip_seconds=args.min_length,
            max_clip_seconds=args.max_length,
            is_captioned=not args.no_captions,
            facecam_index=args.facecam_index,
            whisper_model=args.whisper_model,
            download_section=args.download_section,
        )
        print(json.dumps(run_summary, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
