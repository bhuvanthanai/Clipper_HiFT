# TODO

## 1. Auto clipping — pick the best clip sections automatically

Today the clip window (`--start` / `--end`) is chosen by hand. Replace that with automatic highlight detection.

- [x] Research approaches for finding the best clip sections and compare them on quality, cost and speed (`docs/auto_clipping.md`).
- [ ] Transcript-based: score windows of the Whisper transcript with an LLM for hook, payoff and standalone sense. (Only cheap word-rate/exclamation cues are done.)
- [x] Audio-based: find energy, loudness and laughter/shouting spikes (loudness peaks; no laughter model).
- [x] Video-based: find scene cuts, motion spikes (motion only; no face-expression changes).
- [x] Engagement-based: use YouTube "most replayed" heatmap and chat-replay activity when available.
- [x] Combined: blend the signals above into one score per candidate window.
- [x] Decide clip length rules (e.g. 15–60 s) and snap boundaries to sentence ends and silences.
- [x] Output a ranked list of `{start, end, score, reason}` that steps 3 and 4 can consume.
- [x] Write up the chosen approach and the reasons (`docs/auto_clipping.md`, `feature.md`).

## Ingest — 360p analysis copy

- [x] In `ingest.py`, download a 360p copy alongside the 1080p video (same source, same timeline).
- [x] Record both paths in `ingest_metadata.json`.
- [x] Run frame scanning on the 360p copy (`auto_clip.py` motion); facecam detection moved to 1080p frames for accuracy.
- [x] Boxes are normalised 0-1 and scaled onto the 1080p frame; rendering uses the 1080p copy.

## Bug — AV1 sources can't be decoded

- [x] `test_section.mp4` is AV1; OpenCV builds without an AV1 decoder (no dav1d) fail to read it, so `smart_crop.py` breaks on such input.
- [x] Prefer H.264 in `ingest.py`'s yt-dlp format string (e.g. `bestvideo[vcodec^=avc1][height<=1080]`), or transcode non-H.264 input to H.264 during ingest.

## 2. Streamer layout — facecam top third, stream content bottom two thirds

New fixed layout for streamer content in the 1080x1920 output. No face tracking: both regions are static crops of the source frame.

- [x] Identify the streamer's facecam overlay region in the source frame (a fixed box for the whole video), detected automatically once at the start.
- [x] Detection model: `highheat4/webcam-detect` (YOLO12s, Apache-2.0), run on CPU at 640 px on frames of the 1080p video (the 360p copy merged stacked cams).
- [x] Sampling: ~12 frames spread across the first few minutes (not consecutive), so stream content changes between samples.
- [x] Reject faces in the stream content: keep only boxes that appear in the same place (IoU ≥ 0.8 with the cluster median) in ≥ 75% of samples.
- [x] Rationale: the facecam is a fixed overlay, while faces in gameplay, videos or cutscenes move, change size or disappear between samples.
- [x] Cluster the stable boxes; the facecam box is the median of its cluster.
- [x] When several stable cams are found, pick the largest one (or expose `--facecam-index`).
- [x] When no stable cam is found, log it and render the stream content only (centred 9:16 crop over the whole frame) instead of guessing or falling back to `auto`.
- [ ] Benchmark on real VODs (single cam, multi-cam, borderless cams, react streams with faces in the content) before wiring it in.
- [x] Bottom two thirds (1080x1280): crop the stream content so the facecam overlay is left out.
- [x] Top third (1080x640): crop the facecam region and stretch it to 1080x640 (no aspect-ratio preservation for now).
- [x] Stack the two crops into one 1080x1920 frame.
- [x] Add it as a `--mode` option in `smart_crop.py` and document it in `feature.md`.

## 3. End-to-end flow — URL in, clips out

One command that runs the whole pipeline for a URL.

- [x] New entry script (e.g. `clipper/clip.py --url <url>`).
- [x] Ingest: download the 1080p H.264 video plus the 360p analysis copy.
- [x] Transcribe, then detect the optimal clip time frames with `auto_clip.py`.
- [x] Detect the streamer facecam box once (on frames of the 1080p video: at 360p two stacked cams merged into one box).
- [x] Stream crop box, per axis: from the cam's inner edge to the frame centre, then the same distance past the centre.
- [x] Stream crop box: on an axis where the cam straddles the centre, use the full frame on that axis.
- [x] Render each clip in the streamer layout (stream crop bottom 1080x1280, facecam top 1080x640) with captions.
- [x] Store results in an output folder (e.g. `clipper/output/<video-id>/clip_01.mp4`, plus `clips.json`).
- [x] Defaults: top 5 clips (`--max-clips`), captions on (`--no-captions` to skip).
