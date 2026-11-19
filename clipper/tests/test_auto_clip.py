"""Integration test for auto_clip.py: runs the real script with ffmpeg on the sample video."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CLIPPER_DIR = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = CLIPPER_DIR / "temp_work" / "test_section.mp4"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_auto_clip_ranks_sentence_aligned_clips_from_all_signals(tmp_path):
    transcript_segments = [
        {"start": float(sec), "end": sec + 4.5, "text": "Oh my god, no way!" if sec == 30 else "We keep playing here.",
         "words": [{"word": "word", "start": sec + i * 0.5, "end": sec + i * 0.5 + 0.4} for i in range(8)]}
        for sec in range(0, 50, 5)
    ]
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({"duration": 50.0, "segments": transcript_segments}))
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps({"heatmap": [
        {"start_time": sec, "end_time": sec + 5, "value": 1.0 if 25 <= sec < 40 else 0.1} for sec in range(0, 50, 5)
    ]}))
    chat_path = tmp_path / "live_chat.json"
    chat_path.write_text("\n".join(
        json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(int(offset * 1000))}})
        for offset in [10.0] + [38.0 + i * 0.1 for i in range(40)]
    ) + "\nnot json\n")
    metadata_path = tmp_path / "ingest_metadata.json"
    metadata_path.write_text(json.dumps({"analysis_file_path": str(SAMPLE_VIDEO)}))
    clips_path = tmp_path / "clips.json"

    subprocess.run(
        [sys.executable, str(CLIPPER_DIR / "scripts" / "auto_clip.py"),
         "--video", str(SAMPLE_VIDEO), "--transcription", str(transcript_path),
         "--ingest-metadata", str(metadata_path), "--info-json", str(info_path), "--chat", str(chat_path),
         "--min-length", "15", "--max-length", "30", "--output", str(clips_path)],
        check=True, capture_output=True,
    )

    result = json.loads(clips_path.read_text())
    assert result["signals_used"] == ["audio", "chat", "heatmap", "motion", "speech"]
    clips = result["clips"]
    assert clips and clips[0]["start"] <= 30 < clips[0]["end"]
    segment_starts = {seg["start"] for seg in transcript_segments}
    segment_ends = {seg["end"] for seg in transcript_segments}
    for clip in clips:
        assert 15 <= clip["duration"] <= 30
        assert clip["start"] in segment_starts and clip["end"] in segment_ends
    spans = sorted((clip["start"], clip["end"]) for clip in clips)
    assert all(prev_end <= next_start for (_, prev_end), (next_start, _) in zip(spans, spans[1:]))
    assert [clip["score"] for clip in clips] == sorted((clip["score"] for clip in clips), reverse=True)
