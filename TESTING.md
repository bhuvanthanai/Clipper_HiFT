# Testing

Every change goes through both steps below before it is called done.

## 1. Integration tests

- Run: `python -m pytest -q clipper/tests` (needs `ffmpeg` / `ffprobe` on PATH and `requirements.txt` installed). Shortcut: `task test` (after `task install`).
- Tests use real ffmpeg, real models and the sample video; no mocks.

## 2. Visual validation (screenshot)

Tests check sizes, layouts and JSON, but they cannot tell a good crop from a bad one. So every change that can affect the rendered clips also gets a visual check.

- Run: `python clipper/validate.py` (add `--only facecam` or `--only no-facecam` for one input). Shortcut: `task validate [-- --only facecam]`.
- It renders the reference inputs with `clip.py` and prints the path of each run's `preview.jpg`.
- `preview.jpg` is a contact sheet: one row per clip, frames at 10%, 50% and 90%, labelled with clip, layout and source time.
- Reference inputs:
  - `facecam`: `clipper/temp_work/test_section.mp4` (stream with facecam overlays) → expects the `streamer` layout.
  - `no-facecam`: IShowSpeed VOD minute 60 (Ome.tv split-screen, no overlay) → expects the `stream_only` layout.
- Open every `preview.jpg` and check:
  - The streamer's face is fully in the top third (streamer layout).
  - The main stream content is visible and readable, not cut through the middle of a person or the action.
  - No facecam overlay leaks into the stream crop.
  - Captions sit in the lower part of the frame and don't cover the key content.
- Attach the `preview.jpg` (or a frame from it) to the PR or change report, with one line per reference input on what it shows.
- A run that passes tests but looks wrong in the preview is a failed change.

`clip.py` also writes `preview.jpg` for every normal run, so any output folder can be checked the same way.
