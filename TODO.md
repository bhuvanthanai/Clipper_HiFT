# TODO

## 1. Auto clipping — pick the best clip sections automatically

Today the clip window (`--start` / `--end`) is chosen by hand. Replace that with automatic highlight detection.

- [ ] Research approaches for finding the best clip sections and compare them on quality, cost and speed.
- [ ] Transcript-based: score windows of the Whisper transcript with an LLM for hook, payoff and standalone sense.
- [ ] Audio-based: find energy, loudness and laughter/shouting spikes.
- [ ] Video-based: find scene cuts, motion spikes and face-expression changes.
- [ ] Engagement-based: use YouTube "most replayed" heatmap and chat-replay activity when available.
- [ ] Combined: blend the signals above into one score per candidate window.
- [ ] Decide clip length rules (e.g. 15–60 s) and snap boundaries to sentence ends and silences.
- [ ] Output a ranked list of `{start, end, score, reason}` that steps 3 and 4 can consume.
- [ ] Write up the chosen approach and the reasons in `feature.md`.

## Ingest — 360p analysis copy

- [x] In `ingest.py`, download a 360p copy alongside the 1080p video (same source, same timeline).
- [x] Record both paths in `ingest_metadata.json`.
- [ ] Run crop-location analysis (facecam detection, any frame scanning) on the 360p copy only.
- [ ] Scale detected boxes from 360p to the 1080p frame before cropping; render from the 1080p copy.

## Bug — AV1 sources can't be decoded

- [x] `test_section.mp4` is AV1; OpenCV builds without an AV1 decoder (no dav1d) fail to read it, so `smart_crop.py` breaks on such input.
- [x] Prefer H.264 in `ingest.py`'s yt-dlp format string (e.g. `bestvideo[vcodec^=avc1][height<=1080]`), or transcode non-H.264 input to H.264 during ingest.

## 2. Streamer layout — facecam top third, stream content bottom two thirds

New fixed layout for streamer content in the 1080x1920 output. No face tracking: both regions are static crops of the source frame.

- [ ] Identify the streamer's facecam overlay region in the source frame (a fixed box for the whole video), detected automatically once at the start.
- [ ] Detection model: `highheat4/webcam-detect` (YOLO12s, Apache-2.0), run on CPU at 640 px on the 360p analysis copy.
- [ ] Sampling: ~12 frames spread across the first few minutes (not consecutive), so stream content changes between samples.
- [ ] Reject faces in the stream content: keep only boxes that appear in the same place (IoU ≥ 0.8 with the cluster median) in ≥ 75% of samples.
- [ ] Rationale: the facecam is a fixed overlay, while faces in gameplay, videos or cutscenes move, change size or disappear between samples.
- [ ] Cluster the stable boxes; the facecam box is the median of its cluster.
- [ ] When several stable cams are found, pick the largest one (or expose `--facecam-index`).
- [ ] When no stable cam is found, log it and fall back to the existing `auto` layout instead of guessing.
- [ ] Benchmark on real VODs (single cam, multi-cam, borderless cams, react streams with faces in the content) before wiring it in.
- [ ] Bottom two thirds (1080x1280): crop the stream content so the facecam overlay is left out.
- [ ] Top third (1080x640): crop the facecam region and stretch it to 1080x640 (no aspect-ratio preservation for now).
- [ ] Stack the two crops into one 1080x1920 frame.
- [x] Add it as a `--mode` option in `smart_crop.py` and document it in `feature.md`.

## 3. End-to-end flow — URL in, clips out

One command that runs the whole pipeline for a URL.

- [ ] New entry script (e.g. `clipper/clip.py --url <url>`).
- [ ] Ingest: download the 1080p H.264 video plus the 360p analysis copy.
- [ ] Transcribe, then detect the optimal clip time frames with `auto_clip.py`.
- [ ] Detect the streamer facecam box once on the 360p copy.
- [ ] Stream crop box, per axis: from the cam's inner edge to the frame centre, then the same distance past the centre.
- [ ] Stream crop box: on an axis where the cam straddles the centre, use the full frame on that axis.
- [ ] Render each clip in the streamer layout (stream crop bottom 1080x1280, facecam top 1080x640) with captions.
- [ ] Store results in an output folder (e.g. `clipper/output/<video-id>/clip_01.mp4`, plus `clips.json`).
- [ ] Defaults: top 3 clips (`--max-clips`), captions on (`--no-captions` to skip).
