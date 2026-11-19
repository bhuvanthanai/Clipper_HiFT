"""
Integration test for clip.py: the whole pipeline on the local sample (real whisper, YOLO, ffmpeg).
Needs ffmpeg/ffprobe on PATH and network on first run (model weights).
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2

CLIPPER_DIR = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = CLIPPER_DIR / "temp_work" / "test_section.mp4"


def test_clip_renders_captioned_streamer_clip_from_sample(tmp_path):
    completed = subprocess.run(
        [
            sys.executable, str(CLIPPER_DIR / "clip.py"),
            "--url", str(SAMPLE_VIDEO),
            "--output-root", str(tmp_path / "output"),
            "--work-root", str(tmp_path / "work"),
            "--max-clips", "1",
            "--whisper-model", "tiny",
        ],
        stdout=subprocess.PIPE, text=True, check=True,
    )
    run_summary = json.loads(completed.stdout)
    output_dir = Path(run_summary["output_dir"])

    assert run_summary["facecam"] is not None and run_summary["stream_box"] is not None
    assert [clip["layout"] for clip in run_summary["clips"]] == ["streamer"]
    assert {"clips.json", "facecam.json", "run.json", "clip_01.mp4"} <= {p.name for p in output_dir.iterdir()}
    capture = cv2.VideoCapture(run_summary["clips"][0]["file"])
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1080, 1920)
    capture.release()
