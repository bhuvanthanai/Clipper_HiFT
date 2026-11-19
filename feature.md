# Universal Clipper — Current Flow

Turns a long-form video (YouTube URL or local file) into a 1080x1920 vertical Short with karaoke-style burned-in captions.

## Pipeline overview

```
YouTube URL / local file
        │
        ▼
1. ingest.py          → temp_work/<title>.mp4 + ingest_metadata.json
        │
        ▼
2. transcribe.py      → temp_work/<name>_transcript.json (word timestamps)
        │
        ▼
2b. auto_clip.py      → temp_work/clips.json (optional: ranked best windows)
        │
        ▼
3. smart_crop.py      → 9:16 clip for [start, end] with original audio
        │
        ▼
4. render_captions.py → final MP4 with burned-in karaoke captions
```

- Each step is a standalone CLI script in `clipper/scripts/`, run manually in order, or all chained by `clipper/clip.py` (see "End-to-end run (`clip.py`)").
- Each script prints a JSON result to stdout (`"status": "success"` or `"error"`) and exits non-zero on failure.
- The clip window (`--start` / `--end`) is chosen by the user, or taken from a ranked entry in `clips.json` from `auto_clip.py`; the same window is passed to steps 3 and 4.

## Step 1 — Ingest (`ingest.py`)

- Input: `--input` YouTube URL or local video path; `--output-dir` staging dir (default `./temp_work`).
- URL detection: starts with `http(s)://` or contains `youtube.com` / `youtu.be`.
- YouTube: downloads via `yt-dlp` with the default player clients (forcing `android,web` hits YouTube's SABR-only experiment and caps downloads at 360p), 5 retries, linear retry sleep, no playlists.
- YouTube main video: best video ≤1080p (prefers `avc1`) + best audio, merged into MP4.
- YouTube analysis copy: a separate 360p video-only H.264 download on the same timeline; if it fails, the main video is downscaled instead.
- H.264 only: any non-H.264 output (AV1, VP9, ...) is transcoded to H.264 (libx264 CRF 20, AAC 192k), so OpenCV builds without AV1/VP9 decoders can read it.
- AV1 retry: if the default decoder (libdav1d) fails on a mid-GOP cut, the transcode is retried with `libaom-av1`.
- Local file: if already H.264, remuxes to MP4 (video copied, audio → AAC 192k, `+faststart`); otherwise transcodes to H.264.
- Local file: falls back to a full re-encode (libx264, CRF 20) if the remux fails.
- Local file analysis copy: always built by downscaling the MP4 to 360p (short side), video only.
- Probes the result with `ffprobe` for duration, width, height, fps, video codec, format and size.
- Option: `--download-section START-END` (YouTube only; seconds or HH:MM:SS) downloads just that range (`--force-keyframes-at-cuts`), for both the main video and the analysis copy.
- YouTube also writes the yt-dlp info JSON (`--write-info-json`) next to the video.
- Output: the main MP4, the 360p analysis MP4 and `ingest_metadata.json` in the staging dir.
- Metadata also records `analysis_file_path`, `analysis_width` and `analysis_height`.
- YouTube metadata also records `video_id`, `info_json_path` and `download_section`.

## Step 2 — Transcribe (`transcribe.py`)

- Input: `--input` video/audio path; `--model` tiny | base | small | medium | large-v3 (default `base`).
- Options: `--language` (e.g. `en`, `hi`; auto-detect if omitted), `--beam-size` (default 1), `--cpu-threads` (default 4).
- Engine: `faster-whisper` with word-level timestamps and VAD (500 ms min silence).
- Device: CUDA float16 when available, otherwise CPU int8.
- Device fallback: retries on CPU int8 if CUDA fails at model load or during transcription.
- Initial prompt biases toward English, Hindi and Hinglish code-switching, slang and gaming speech.
- Output: `--output` JSON (default `./temp_work/<name>_transcript.json`).
- Transcript shape: detected language, duration, and `segments[]` each with `start`, `end`, `text`, `words[]`.
- Word shape: `word`, `start`, `end`, `probability`.

## Step 2b — Auto clip (`auto_clip.py`, optional)

- Input: `--video`, `--transcription` (both required); optional `--ingest-metadata`, `--info-json`, `--chat`.
- Options: `--min-length` (default 15 s), `--max-length` (default 60 s), `--max-clips` (default 5), `--output` (default `./temp_work/clips.json`).
- Signals, per second: `heatmap` (yt-dlp `heatmap` from `--info-json`), `chat` (live-chat replay rate, `--chat`), `audio` (loudness peaks), `speech` (words/s and excitement cues), `motion` (frame diffs on the analysis copy from `--ingest-metadata` `analysis_file_path`).
- Fusion weights: heatmap 0.35, chat 0.25, audio 0.2, speech 0.1, motion 0.1; renormalised over the signals present (audio + transcript always).
- Windows start at transcript segment starts, end at sentence ends or pauses, and are picked greedily without overlap.
- CPU only, no models: ffmpeg decodes audio and the low-res video; the rest is numpy.
- Output: `clips.json` with `signals_used` and ranked `clips[]` of `rank`, `start`, `end`, `duration`, `score`, `reason`; pass a clip's `start` / `end` to steps 3 and 4.
- Research and rationale: `docs/auto_clipping.md`.

## Step 3 — Smart crop (`smart_crop.py`)

- Input: `--input` video, `--output` path, optional `--start` / `--end` seconds.
- Options: `--mode` auto | face | pillarbox | streamer (default `auto`), `--analysis-video`, `--facecam-index` (default 0), `--facecam-detection`, `--fps` (default 60), `--alpha` (default 0.08), `--timeout` (default 1.5 s).
- Crop window: full source height, width sized for 9:16, scaled to 1080x1920.
- Face tracking: MediaPipe face detection on every frame; follows the largest face.
- Smoothing: EMA on crop x-position (`--alpha`) with a 20 px deadband to suppress jitter.
- `auto` mode: switches to the pillarbox layout after `--timeout` seconds without a face.
- Pillarbox layout: full frame scaled to 1080 width, centred over a blurred, dimmed copy of itself.
- Layout switch: cross-fades between face-crop and pillarbox (~0.4 s glide).
- `face` / `pillarbox` modes force one layout for the whole clip.
- `streamer` mode: static layout, no face tracking; facecam stretched to 1080x640 on top, stream crop stretched to 1080x1280 below.
- Facecam detection (`facecam_detect.py`): runs once, on `--analysis-video` (low-res copy, same timeline) or `--input`; boxes are scaled to the source frame.
- `--facecam-detection`: path to a saved `facecam_detect.py` result JSON, reused instead of detecting again (`clip.py` detects once and passes it to every clip).
- Detector: `highheat4/webcam-detect` (YOLO12s, pinned revision) on CPU at 640 px, over 12 frames spread across the first 3 minutes.
- Stable cams: keeps only boxes with IoU >= 0.8 in >= 75% of samples (rejects faces inside the stream content); median box per cam, largest first. `--facecam-index` picks one.
- Stream crop (per axis): cam fully on one side of the centre → crop from the cam's inner edge to the same distance past the centre; cam straddling the centre → full axis; straddling both axes → no crop.
- Fallback: no stable cam or no usable stream crop → logs it and renders the stream content only (`layout` = `stream_only`): a centred 9:16 crop of the source scaled (aspect kept) to fill the 1080x1920 frame.
- `facecam_detect.py` also runs standalone and prints the detection result JSON.
- MediaPipe is imported only for the face-tracking modes.
- Encoding: frames piped to ffmpeg; `h264_nvenc` (CQ 19) if available, else `libx264` (CRF 18).
- Audio: the matching segment of the source audio is muxed back in (AAC 192k).
- Output: 9:16 MP4 at `--output`; temp video-only file is deleted. The result JSON includes `layout` and `streamer_layout`.

## Step 4 — Captions (`render_captions.py`)

- Input: `--video` (cropped 9:16 clip), `--transcription` JSON, `--start` / `--end` (same window as step 3), `--output`.
- Options: `--words-per-phrase` (default 3), `--crf` (default 20), `--fps` (default 60).
- Word selection: keeps words overlapping the clip window and shifts timestamps to clip-relative.
- Missing word timestamps: spreads a segment's words evenly across its duration.
- Phrasing: breaks after N words, at sentence-ending punctuation, or at gaps > 0.4 s.
- Gap bridging: gaps < 0.25 s between words in a phrase are closed to avoid flicker.
- Style: uppercase Arial Black 78, white text, black outline 5, shadow 3, bottom-centred.
- Safe area: 280 px bottom margin to clear the Shorts UI.
- Highlight: one subtitle event per word; the active word is gold (#FFD700), others white.
- Burn-in: ffmpeg `ass` filter, libx264 CRF 20, AAC 192k, `yuv420p`, `+faststart`.
- Output: final MP4 at `--output`; the temporary `.ass` file is deleted.

## End-to-end run (`clip.py`)

- `python clipper/clip.py --url <url>` chains ingest → transcribe → auto clip → facecam detection (once) → smart crop (streamer mode) → captions, per clip. A local video path also works as `--url`.
- Options: `--max-clips` (default 5), `--min-length` (default 15 s), `--max-length` (default 60 s), `--no-captions`, `--facecam-index` (default 0), `--whisper-model` (default `base`), `--download-section START-END`, `--output-root` (default `clipper/output`), `--work-root` (default `clipper/temp_work`).
- Intermediates go to `<work-root>/run_<timestamp>/`; results go to `<output-root>/<video id>/` (a sanitised title for local files).
- Output: `clip_NN.mp4` per ranked clip, plus `clips.json`, `facecam.json`, `run.json` (source, title, signals used, facecam / stream box, per-clip file and layout, `preview` path) and `preview.jpg`.
- `preview.jpg` is a contact sheet: one row per clip, frames at 10%, 50% and 90%, each labelled with clip name, layout and source time.
- `python clipper/validate.py` renders the reference inputs (`facecam`: `temp_work/test_section.mp4`; `no-facecam`: IShowSpeed VOD minute 60) and prints each run's `preview.jpg` path. Options: `--only facecam|no-facecam`, `--max-clips` (default 2), `--output-root` (default `clipper/temp_work/validation`). See `TESTING.md` for what to check in the previews.
- Facecam detection runs on frames of the 1080p video, not the 360p copy: at 360p the detector merged two stacked cams into one box in the sample; 1080p frames separate them.
- With `--download-section`, the heatmap signal is skipped (its times are for the full video).
- With `--no-captions`, the cropped clip is copied as the final file.
- Each step runs as a subprocess; any failure aborts the run with a JSON error on stderr.

## Example end-to-end run

```bash
cd clipper
python clip.py --url "https://www.youtube.com/watch?v=<id>"   # all steps in one command
```

Or with [Task](https://taskfile.dev) (repo root; uses `.venv`, override with `VENV=<path>`; uses the venv's `bin/python` on Linux/macOS and `Scripts/python.exe` on Windows; `install` builds the venv with `python3` (`python` on Windows), override with `SYSTEM_PYTHON=<path>`, and stops early if that python is older than 3.10 or lacks `venv`/`ensurepip` (e.g. missing `python3-venv` on Ubuntu/Debian), with a per-OS fix hint; `clip`/`test`/`validate` check that `ffmpeg` / `ffprobe` run, with a per-OS install hint (winget on Windows, apt/brew elsewhere); tasks set `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` so progress symbols don't crash a Windows console):

```bash
task install                                   # create .venv, install requirements.txt
task clip URL="https://www.youtube.com/watch?v=<id>" [-- extra clip.py flags]
task clip:section URL=<url> SECTION=3600-3780  # adds --download-section
task test                                      # integration tests
task validate                                  # clipper/validate.py
```

Or step by step:

```bash
cd clipper
python scripts/ingest.py -i "https://www.youtube.com/watch?v=<id>" -o ./temp_work
python scripts/transcribe.py -i "./temp_work/<title> [<id>].mp4" -o ./temp_work/transcript.json
python scripts/smart_crop.py -i "./temp_work/<title> [<id>].mp4" -o ./temp_work/cropped.mp4 -s 120 -e 165
python scripts/render_captions.py -v ./temp_work/cropped.mp4 -t ./temp_work/transcript.json -s 120 -e 165 -o ./output/short.mp4
```

## Dependencies

- System: `ffmpeg` and `ffprobe` on PATH (scripts first check hardcoded Windows WinGet paths).
- Python: `yt-dlp`, `faster-whisper` (with `ctranslate2`), `mediapipe`, `opencv-python`, `numpy`; streamer mode also needs `ultralytics` and `huggingface_hub`. Install via the root `requirements.txt` (pins `av==14.2.0`, `mediapipe==0.10.21`; mediapipe is skipped on Python 3.13+, which has no wheel for it, so only the manual `auto`/`face` modes are unavailable there and `clip.py` still works).
- Optional: NVIDIA GPU for CUDA transcription and `h264_nvenc` encoding.

## Test assets (`clipper/temp_work/`)

- `test_section.mp4`: sample source clip.
- `frame_sample.jpg`, `frame_gameplay.jpg`: sample frames (talking head and gameplay).
- `info.json`: yt-dlp metadata for a sample YouTube video.

## Known gaps

- Run step by step, `auto_clip.py` is not wired into steps 3 and 4 (`start` / `end` are passed by hand); only `clip.py` chains them.
- `auto_clip.py` has no semantic (LLM) scoring; it ranks on cheap signals only.
- `clip.py` always renders in streamer mode (no `--mode` option); when no stable facecam is found it renders the stream content only (`stream_only` layout).
- No README.
- `clip.py` does not pass `--analysis-video`, so facecam detection runs on the 1080p video, not the 360p copy.
- `--mode stacked` / `split` are accepted by the CLI, but `make_stacked_layout` is never called, so both behave like `auto`.
- ffmpeg candidate paths are hardcoded to the author's Windows machine.
- Smart crop feeds frames at the source fps and outputs 60 fps, so 30 fps sources get duplicated frames.
- A compiled `__pycache__/*.pyc` file is committed.
