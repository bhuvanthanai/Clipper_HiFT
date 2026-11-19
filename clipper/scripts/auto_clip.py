"""
auto_clip.py - Pick the best clip sections of a long video automatically.

Scores every second of the source on the signals that are available, then picks
non-overlapping 15-60 s windows whose boundaries sit on transcript sentence ends.

Signals (each optional except audio + transcript; weights renormalise over what exists):
- heatmap: YouTube "most replayed" curve from yt-dlp metadata (`heatmap` field).
- chat:    live-chat replay message rate (yt-dlp `--write-subs --sub-langs live_chat`).
- audio:   short-term loudness peaks relative to the local baseline (shouting, laughter, hype).
- speech:  words per second and exclamation/laughter cues from the transcript.
- motion:  frame-difference energy from the low-res analysis copy (scene changes, action).

Runs on CPU with no models: ffmpeg decodes audio (16 kHz mono) and the analysis video
(64x36 grey, 2 fps); everything else is numpy.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

FFMPEG_CANDIDATES = [
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    "ffmpeg",
]

SIGNAL_WEIGHTS = {"heatmap": 0.35, "chat": 0.25, "audio": 0.2, "speech": 0.1, "motion": 0.1}
AUDIO_SAMPLE_RATE = 16000
MOTION_FPS = 2
MOTION_SIZE = (64, 36)
CHAT_REACTION_LAG_SEC = 8.0
EXCITEMENT_TOKENS = ("!", "?!", "haha", "lol", "lmao", "omg", "oh my god", "what", "no way", "bro", "let's go", "yo")
SENTENCE_END_CHARS = (".", "!", "?")


def get_ffmpeg_bin():
    for candidate in FFMPEG_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return "ffmpeg"


FFMPEG_BIN = get_ffmpeg_bin()


def get_robust_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score using median/MAD so a few huge spikes don't flatten everything else."""
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    if mad < 1e-9:
        mad = values.std() or 1.0
    return (values - median) / mad


def get_normalized_signal(values: np.ndarray) -> np.ndarray:
    """Map a per-second signal to 0..1 by clipping its robust z-score to [0, 3]."""
    return np.clip(get_robust_zscore(values), 0.0, 3.0) / 3.0


def get_smoothed(values: np.ndarray, window_sec: int) -> np.ndarray:
    if window_sec <= 1 or len(values) < window_sec:
        return values
    kernel = np.ones(window_sec) / window_sec
    return np.convolve(values, kernel, mode="same")


def get_audio_signal(video_path: str, total_secs: int) -> np.ndarray:
    """Per-second loudness (dB RMS) minus a 60 s rolling baseline, so peaks are local."""
    cmd = [
        FFMPEG_BIN, "-v", "error", "-i", video_path, "-vn", "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    secs = min(total_secs, len(samples) // AUDIO_SAMPLE_RATE)
    frames = samples[: secs * AUDIO_SAMPLE_RATE].reshape(secs, AUDIO_SAMPLE_RATE)
    rms_db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    baseline = get_smoothed(rms_db, window_sec=60)
    signal = np.zeros(total_secs, dtype=np.float32)
    signal[:secs] = rms_db - baseline
    return get_normalized_signal(get_smoothed(signal, window_sec=3))


def get_motion_signal(analysis_video_path: str, total_secs: int) -> np.ndarray:
    """Per-second mean absolute frame difference on a tiny grey decode of the analysis copy."""
    width, height = MOTION_SIZE
    cmd = [
        FFMPEG_BIN, "-v", "error", "-threads", "1", "-i", analysis_video_path,
        "-vf", f"fps={MOTION_FPS},scale={width}:{height},format=gray",
        "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width).astype(np.float32)
    signal = np.zeros(total_secs, dtype=np.float32)
    if len(frames) < 2:
        return signal
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    for frame_idx, diff in enumerate(diffs, start=1):
        sec = int(frame_idx / MOTION_FPS)
        if sec < total_secs:
            signal[sec] = max(signal[sec], diff)
    return get_normalized_signal(get_smoothed(signal, window_sec=3))


def get_heatmap_signal(heatmap_points: list, total_secs: int) -> np.ndarray:
    signal = np.zeros(total_secs, dtype=np.float32)
    for point in heatmap_points:
        start = max(0, int(point["start_time"]))
        end = min(total_secs, int(np.ceil(point["end_time"])))
        signal[start:end] = float(point["value"])
    return signal


def get_chat_message_offsets(chat_path: str) -> list:
    """Message times (seconds) from a yt-dlp live_chat.json (one JSON object per line)."""
    offsets = []
    with open(chat_path, "r", encoding="utf-8") as chat_file:
        for line in chat_file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay_action = entry.get("replayChatItemAction", {})
            offset_msec = replay_action.get("videoOffsetTimeMsec")
            if offset_msec is not None:
                offsets.append(int(offset_msec) / 1000.0)
    return offsets


def get_chat_signal(chat_path: str, total_secs: int) -> np.ndarray:
    """Chat burst rate, shifted earlier by the typical viewer reaction lag."""
    counts = np.zeros(total_secs, dtype=np.float32)
    for offset in get_chat_message_offsets(chat_path):
        sec = int(offset - CHAT_REACTION_LAG_SEC)
        if 0 <= sec < total_secs:
            counts[sec] += 1
    return get_normalized_signal(get_smoothed(counts, window_sec=5))


def get_speech_signal(transcript_segments: list, total_secs: int) -> np.ndarray:
    """Words per second plus a bonus for exclamations / laughter cues."""
    signal = np.zeros(total_secs, dtype=np.float32)
    for segment in transcript_segments:
        for word in segment.get("words", []):
            sec = int(word["start"])
            if 0 <= sec < total_secs:
                signal[sec] += 1.0
        text = segment.get("text", "").lower()
        if any(token in text for token in EXCITEMENT_TOKENS):
            start, end = int(segment["start"]), min(total_secs, int(np.ceil(segment["end"])))
            signal[start:end] += 1.5
    return get_normalized_signal(get_smoothed(signal, window_sec=5))


def get_sentence_boundaries(transcript_segments: list) -> tuple:
    """Clip start candidates (segment starts) and end candidates (sentence ends or pauses)."""
    starts, ends = [], []
    for idx, segment in enumerate(transcript_segments):
        starts.append(float(segment["start"]))
        next_start = transcript_segments[idx + 1]["start"] if idx + 1 < len(transcript_segments) else None
        is_sentence_end = segment.get("text", "").rstrip().endswith(SENTENCE_END_CHARS)
        is_followed_by_pause = next_start is None or (next_start - segment["end"]) >= 0.5
        if is_sentence_end or is_followed_by_pause:
            ends.append(float(segment["end"]))
    return starts, ends


def get_clip_reason(window_signals: dict, start: float) -> str:
    parts = []
    for name, values in sorted(window_signals.items(), key=lambda kv: -kv[1].mean()):
        if values.mean() < 0.05:
            continue
        peak_sec = start + int(np.argmax(values))
        parts.append(f"{name} {values.mean():.2f} (peak at {peak_sec:.0f}s)")
    return "; ".join(parts[:3]) or "no strong signal"


def build_ranked_clips(
    signals: dict,
    transcript_segments: list,
    min_clip_sec: float,
    max_clip_sec: float,
    max_clips: int,
) -> list:
    total_weight = sum(SIGNAL_WEIGHTS[name] for name in signals)
    combined = sum(SIGNAL_WEIGHTS[name] * values for name, values in signals.items()) / total_weight
    clip_starts, clip_ends = get_sentence_boundaries(transcript_segments)
    candidates = []
    for start in clip_starts:
        for end in clip_ends:
            length = end - start
            if length < min_clip_sec:
                continue
            if length > max_clip_sec:
                break
            window = combined[int(start): int(np.ceil(end))]
            if len(window) == 0:
                continue
            peak = np.sort(window)[-5:].mean()
            score = 0.6 * window.mean() + 0.4 * peak
            candidates.append((score, start, end))
    candidates.sort(reverse=True)
    picked = []
    for score, start, end in candidates:
        if any(start < other_end and end > other_start for _, other_start, other_end in picked):
            continue
        picked.append((score, start, end))
        if len(picked) >= max_clips:
            break
    clips = []
    for rank, (score, start, end) in enumerate(picked, start=1):
        window_signals = {name: values[int(start): int(np.ceil(end))] for name, values in signals.items()}
        clips.append({
            "rank": rank,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "score": round(float(score), 4),
            "reason": get_clip_reason(window_signals, start),
        })
    return clips


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as json_file:
        return json.load(json_file)


def main():
    parser = argparse.ArgumentParser(description="Rank the best clip sections of a long video.")
    parser.add_argument("--video", "-v", required=True, help="Source video (same timeline as the transcript)")
    parser.add_argument("--transcription", "-t", required=True, help="Transcript JSON from transcribe.py")
    parser.add_argument("--ingest-metadata", help="ingest_metadata.json (uses analysis_file_path for motion)")
    parser.add_argument("--info-json", help="yt-dlp info JSON (uses the heatmap field when present)")
    parser.add_argument("--chat", help="yt-dlp live_chat.json replay file")
    parser.add_argument("--min-length", type=float, default=15.0, help="Minimum clip length in seconds")
    parser.add_argument("--max-length", type=float, default=60.0, help="Maximum clip length in seconds")
    parser.add_argument("--max-clips", type=int, default=5, help="Number of clips to return")
    parser.add_argument("--output", "-o", default="./temp_work/clips.json", help="Output clips JSON")
    args = parser.parse_args()

    try:
        transcript = load_json(args.transcription)
        transcript_segments = transcript.get("segments", [])
        if not transcript_segments:
            raise ValueError("Transcript has no segments")
        total_secs = int(np.ceil(float(transcript.get("duration") or transcript_segments[-1]["end"]))) + 1

        signals = {
            "audio": get_audio_signal(args.video, total_secs),
            "speech": get_speech_signal(transcript_segments, total_secs),
        }
        analysis_video_path = None
        if args.ingest_metadata:
            analysis_video_path = load_json(args.ingest_metadata).get("analysis_file_path")
        if analysis_video_path and os.path.exists(analysis_video_path):
            try:
                signals["motion"] = get_motion_signal(analysis_video_path, total_secs)
            except subprocess.CalledProcessError as motion_error:
                print(f"[auto_clip] Skipping motion signal, decode failed: {motion_error.stderr[-300:]!r}", file=sys.stderr)
        if args.info_json:
            heatmap_points = load_json(args.info_json).get("heatmap") or []
            if heatmap_points:
                signals["heatmap"] = get_heatmap_signal(heatmap_points, total_secs)
        if args.chat and os.path.exists(args.chat):
            signals["chat"] = get_chat_signal(args.chat, total_secs)

        clips = build_ranked_clips(
            signals=signals,
            transcript_segments=transcript_segments,
            min_clip_sec=args.min_length,
            max_clip_sec=args.max_length,
            max_clips=args.max_clips,
        )
        result = {
            "status": "success",
            "video_path": os.path.abspath(args.video),
            "signals_used": sorted(signals),
            "clips": clips,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
        result["clips_file"] = os.path.abspath(args.output)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
