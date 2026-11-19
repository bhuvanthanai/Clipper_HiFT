"""
validate.py - Visual validation run for clipper changes.

Runs clip.py on a fixed set of reference inputs that cover each layout path and
collects every run's preview.jpg contact sheet. Open each sheet and check it by eye
before calling a change done; tests alone don't show a bad crop.

Reference inputs:
- facecam: temp_work/test_section.mp4 (Rocket League stream with facecam overlays) -> streamer layout
- no-facecam: IShowSpeed VOD minute 60 (Ome.tv split-screen, no overlay) -> stream-only layout
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CLIPPER_DIR = Path(__file__).resolve().parent
REFERENCE_INPUTS = [
    {"name": "facecam", "url": str(CLIPPER_DIR / "temp_work" / "test_section.mp4"), "extra_args": []},
    {
        "name": "no-facecam",
        "url": "https://www.youtube.com/watch?v=o0UeucVxYCA",
        "extra_args": ["--download-section", "3600-3660"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Render the reference inputs and collect preview sheets.")
    parser.add_argument("--output-root", default=str(CLIPPER_DIR / "temp_work" / "validation"), help="Validation runs go to <output-root>/<timestamp>/")
    parser.add_argument("--max-clips", type=int, default=2, help="Clips per reference input (default 2)")
    parser.add_argument("--only", choices=[r["name"] for r in REFERENCE_INPUTS], help="Run a single reference input")
    args = parser.parse_args()

    validation_dir = Path(args.output_root) / time.strftime("%Y%m%d_%H%M%S")
    reference_inputs = [r for r in REFERENCE_INPUTS if args.only in (None, r["name"])]
    results = []
    for reference_input in reference_inputs:
        print(f"[validate] ▶ {reference_input['name']}", file=sys.stderr)
        completed = subprocess.run(
            [
                sys.executable, str(CLIPPER_DIR / "clip.py"),
                "--url", reference_input["url"],
                "--output-root", str(validation_dir / reference_input["name"] / "output"),
                "--work-root", str(validation_dir / reference_input["name"] / "work"),
                "--max-clips", str(args.max_clips),
                *reference_input["extra_args"],
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            results.append({"name": reference_input["name"], "status": "error"})
            continue
        run_summary = json.loads(completed.stdout)
        results.append({
            "name": reference_input["name"],
            "status": "success",
            "preview": run_summary["preview"],
            "layouts": [clip["layout"] for clip in run_summary["clips"]],
        })

    print(json.dumps({"validation_dir": str(validation_dir), "runs": results}, indent=2))
    sys.exit(0 if all(r["status"] == "success" for r in results) else 1)


if __name__ == "__main__":
    main()
