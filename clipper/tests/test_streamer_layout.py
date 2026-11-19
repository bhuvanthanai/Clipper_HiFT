"""
Integration tests for the streamer layout: real facecam model, real ffmpeg, real sample video.
Needs ffmpeg/ffprobe on PATH and network access on first run (model weights download).
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SAMPLE_VIDEO = Path(__file__).resolve().parents[1] / "temp_work" / "test_section.mp4"
sys.path.insert(0, str(SCRIPTS_DIR))

from facecam_detect import get_stream_box  # noqa: E402


def transcode_sample(output_path: Path, extra_args: list) -> None:
    """Transcodes the first 15 s of the AV1 sample to H.264 (retrying with libaom if dav1d fails)."""
    base_args = ["-t", "15", "-i", str(SAMPLE_VIDEO), *extra_args, "-c:v", "libx264", "-preset", "veryfast", str(output_path)]
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *base_args], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libaom-av1", *base_args], check=True)


def test_streamer_mode_renders_facecam_over_stream_crop(tmp_path):
    source_video = tmp_path / "source_1080p.mp4"
    analysis_video = tmp_path / "analysis_360p.mp4"
    output_video = tmp_path / "streamer.mp4"
    transcode_sample(output_path=source_video, extra_args=[])
    transcode_sample(output_path=analysis_video, extra_args=["-vf", "scale=-2:360", "-an"])

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPTS_DIR / "smart_crop.py"),
            "--input", str(source_video),
            "--analysis-video", str(analysis_video),
            "--output", str(output_video),
            "--mode", "streamer",
            "--start", "2", "--end", "6", "--fps", "30",
        ],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    streamer_layout = result["streamer_layout"]

    assert result["layout"] == "streamer"
    assert result["resolution"] == "1080x1920"
    fx1, fy1, fx2, fy2 = streamer_layout["facecam_box"]
    sx1, sy1, sx2, sy2 = streamer_layout["stream_box"]
    assert fx1 < fx2 and fy1 < fy2
    assert sx2 - sx1 > 0 and sy2 - sy1 > 0
    assert (sx1 >= fx2) or (sx2 <= fx1) or (sy1 >= fy2) or (sy2 <= fy1), "stream crop must exclude the facecam"

    capture = cv2.VideoCapture(str(output_video))
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1080, 1920)
    capture.release()


@pytest.mark.parametrize(
    "facecam_box, expected_stream_box",
    [
        ([0.6, 0.1, 0.9, 0.3], [0.4, 0.3, 0.6, 0.7]),
        ([0.4, 0.4, 0.6, 0.6], None),
    ],
)
def test_get_stream_box_per_axis_rule(facecam_box, expected_stream_box):
    stream_box = get_stream_box(facecam_box=facecam_box)
    if expected_stream_box is None:
        assert stream_box is None
    else:
        assert stream_box == pytest.approx(expected_stream_box)


def test_streamer_mode_without_facecam_renders_stream_only(tmp_path):
    source_video = tmp_path / "source_1080p.mp4"
    output_video = tmp_path / "stream_only.mp4"
    detection_path = tmp_path / "facecam.json"
    transcode_sample(output_path=source_video, extra_args=[])
    detection_path.write_text(json.dumps({"facecam": None, "stream_box": None, "stable_facecams": [], "sampled_frames": 12, "elapsed_seconds": 0}))

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPTS_DIR / "smart_crop.py"),
            "--input", str(source_video), "--output", str(output_video),
            "--mode", "streamer", "--facecam-detection", str(detection_path),
            "--start", "2", "--end", "4", "--fps", "30",
        ],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)

    assert result["layout"] == "stream_only"
    assert result["streamer_layout"]["stream_box"] == [656, 0, 1264, 1080]
    capture = cv2.VideoCapture(str(output_video))
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1080, 1920)
    capture.release()
