"""
Integration test for ingest.py: runs the real CLI (ffmpeg, ffprobe, yt-dlp, network)
and checks the ingest_metadata.json contract that smart_crop / facecam analysis rely on.
Requires ffmpeg + ffprobe on PATH and yt-dlp, opencv-python in the environment.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

CLIPPER_DIR = Path(__file__).resolve().parents[1]
INGEST_SCRIPT = CLIPPER_DIR / "scripts" / "ingest.py"
AV1_SAMPLE_PATH = CLIPPER_DIR / "temp_work" / "test_section.mp4"
SHORT_YOUTUBE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def get_stream_info(video_path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
         "-show_entries", "format=duration", "-of", "json", video_path],
        stdout=subprocess.PIPE, text=True, check=True,
    )
    probe = json.loads(result.stdout)
    return {**probe["streams"][0], "duration": float(probe["format"]["duration"])}


@pytest.mark.parametrize("ingest_input", [str(AV1_SAMPLE_PATH), SHORT_YOUTUBE_URL], ids=["local-av1", "youtube"])
def test_ingest_produces_h264_main_and_360p_analysis_copy(ingest_input, tmp_path):
    subprocess.run([sys.executable, str(INGEST_SCRIPT), "-i", ingest_input, "-o", str(tmp_path)], check=True)
    ingest_metadata = json.loads((tmp_path / "ingest_metadata.json").read_text())

    main_info = get_stream_info(ingest_metadata["file_path"])
    analysis_info = get_stream_info(ingest_metadata["analysis_file_path"])

    assert main_info["codec_name"] == "h264"
    assert analysis_info["codec_name"] == "h264"
    assert min(analysis_info["width"], analysis_info["height"]) <= 360
    assert (analysis_info["width"], analysis_info["height"]) == (ingest_metadata["analysis_width"], ingest_metadata["analysis_height"])
    assert abs(analysis_info["duration"] - main_info["duration"]) < 0.5
    for video_path in (ingest_metadata["file_path"], ingest_metadata["analysis_file_path"]):
        capture = cv2.VideoCapture(video_path)
        is_frame_read, _ = capture.read()
        capture.release()
        assert is_frame_read
