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
3. smart_crop.py      → 9:16 clip for [start, end] with original audio
        │
        ▼
4. render_captions.py → final MP4 with burned-in karaoke captions
```

- Each step is a standalone CLI script in `clipper/scripts/`, run manually in order.
- Each script prints a JSON result to stdout (`"status": "success"` or `"error"`) and exits non-zero on failure.
- The clip window (`--start` / `--end`) is chosen by the user; the same window is passed to steps 3 and 4.

## Step 1 — Ingest (`ingest.py`)

- Input: `--input` YouTube URL or local video path; `--output-dir` staging dir (default `./temp_work`).
- URL detection: starts with `http(s)://` or contains `youtube.com` / `youtu.be`.
- YouTube: downloads via `yt-dlp`, best video ≤1080p + best audio, merged into MP4.
- YouTube: uses `android,web` player clients, 5 retries, linear retry sleep, no playlists.
- Local file: remuxes to MP4 (video copied, audio → AAC 192k, `+faststart`).
- Local file: falls back to a full re-encode (libx264, CRF 20) if the remux fails.
- Probes the result with `ffprobe` for duration, width, height, fps, format and size.
- Output: the MP4 plus `ingest_metadata.json` in the staging dir.

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

## Step 3 — Smart crop (`smart_crop.py`)

- Input: `--input` video, `--output` path, optional `--start` / `--end` seconds.
- Options: `--mode` auto | face | pillarbox (default `auto`), `--fps` (default 60), `--alpha` (default 0.08), `--timeout` (default 1.5 s).
- Crop window: full source height, width sized for 9:16, scaled to 1080x1920.
- Face tracking: MediaPipe face detection on every frame; follows the largest face.
- Smoothing: EMA on crop x-position (`--alpha`) with a 20 px deadband to suppress jitter.
- `auto` mode: switches to the pillarbox layout after `--timeout` seconds without a face.
- Pillarbox layout: full frame scaled to 1080 width, centred over a blurred, dimmed copy of itself.
- Layout switch: cross-fades between face-crop and pillarbox (~0.4 s glide).
- `face` / `pillarbox` modes force one layout for the whole clip.
- Encoding: frames piped to ffmpeg; `h264_nvenc` (CQ 19) if available, else `libx264` (CRF 18).
- Audio: the matching segment of the source audio is muxed back in (AAC 192k).
- Output: 9:16 MP4 at `--output`; temp video-only file is deleted.

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

## Example end-to-end run

```bash
cd clipper
python scripts/ingest.py -i "https://www.youtube.com/watch?v=<id>" -o ./temp_work
python scripts/transcribe.py -i "./temp_work/<title> [<id>].mp4" -o ./temp_work/transcript.json
python scripts/smart_crop.py -i "./temp_work/<title> [<id>].mp4" -o ./temp_work/cropped.mp4 -s 120 -e 165
python scripts/render_captions.py -v ./temp_work/cropped.mp4 -t ./temp_work/transcript.json -s 120 -e 165 -o ./output/short.mp4
```

## Dependencies

- System: `ffmpeg` and `ffprobe` on PATH (scripts first check hardcoded Windows WinGet paths).
- Python: `yt-dlp`, `faster-whisper` (with `ctranslate2`), `mediapipe`, `opencv-python`, `numpy`.
- Optional: NVIDIA GPU for CUDA transcription and `h264_nvenc` encoding.

## Test assets (`clipper/temp_work/`)

- `test_section.mp4`: sample source clip.
- `frame_sample.jpg`, `frame_gameplay.jpg`: sample frames (talking head and gameplay).
- `info.json`: yt-dlp metadata for a sample YouTube video.

## Known gaps

- No highlight detection: clip start/end must be picked by hand.
- No orchestrator: the four steps are not chained by a single command.
- No README or `requirements.txt`.
- `--mode stacked` / `split` are accepted by the CLI, but `make_stacked_layout` is never called, so both behave like `auto`.
- ffmpeg candidate paths are hardcoded to the author's Windows machine.
- Smart crop feeds frames at the source fps and outputs 60 fps, so 30 fps sources get duplicated frames.
- A compiled `__pycache__/*.pyc` file is committed.
