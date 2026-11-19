"""
facecam_detect.py - Find the streamer's facecam overlay once per video.

Runs the highheat4/webcam-detect YOLO12s model on CPU over frames sampled across
the first few minutes, then keeps only boxes that stay in the same place in most
samples. A facecam is a fixed overlay, so this rejects faces that belong to the
stream content (gameplay, videos, cutscenes), which move, resize or disappear.

Boxes are returned normalised to 0-1 so they can be scaled onto any rendition of
the same video (e.g. detect on the 360p analysis copy, crop the 1080p source).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

WEBCAM_MODEL_REPO_ID = "highheat4/webcam-detect"
WEBCAM_MODEL_FILENAME = "yolo12s_finetuned.pt"
WEBCAM_MODEL_REVISION = "5a54ecc6280dba8182648fe00e81cb05f314f3a2"
DEFAULT_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "clipper", "models")


def get_webcam_model_path(model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR) -> str:
    """Downloads (once) the pinned webcam-detect weights and returns the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=WEBCAM_MODEL_REPO_ID,
        filename=WEBCAM_MODEL_FILENAME,
        revision=WEBCAM_MODEL_REVISION,
        cache_dir=model_cache_dir,
    )


def load_webcam_model(model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR):
    from ultralytics import YOLO

    return YOLO(get_webcam_model_path(model_cache_dir=model_cache_dir))


def get_sample_times(video_duration: float, sample_count: int, sampling_window_seconds: float) -> list:
    """Evenly spaced timestamps across the first `sampling_window_seconds`, skipping both ends."""
    window = min(video_duration, sampling_window_seconds)
    return [window * (i + 1) / (sample_count + 1) for i in range(sample_count)]


def read_sample_frames(input_paths: list, sample_count: int, sampling_window_seconds: float) -> list:
    """
    Returns frames to analyse. Each input is tried as a still image first; anything
    OpenCV cannot read as an image is opened as a video and sampled over time.
    """
    frames = []
    for input_path in input_paths:
        image = cv2.imread(input_path)
        if image is not None:
            frames.append(image)
            continue
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open input as image or video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        video_duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        for sample_time in get_sample_times(
            video_duration=video_duration,
            sample_count=sample_count,
            sampling_window_seconds=sampling_window_seconds,
        ):
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            is_frame_read, frame = capture.read()
            if is_frame_read and frame is not None:
                frames.append(frame)
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from: {input_paths}")
    return frames


def get_box_iou(box_a, box_b) -> float:
    inter_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    union = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter
    return inter / union if union > 0 else 0.0


def get_stable_facecam_boxes(boxes_per_frame: list, iou_threshold: float, min_presence_ratio: float) -> list:
    """
    Clusters per-frame boxes by IoU against each cluster's running median and keeps
    clusters present (in distinct frames) in at least `min_presence_ratio` of frames.
    Returns stable clusters as dicts, largest box first.
    """
    clusters = []
    for frame_index, frame_boxes in enumerate(boxes_per_frame):
        for box in frame_boxes:
            matching_cluster = None
            for cluster in clusters:
                if get_box_iou(box, np.median(cluster["boxes"], axis=0)) >= iou_threshold:
                    matching_cluster = cluster
                    break
            if matching_cluster is None:
                clusters.append({"boxes": [box], "frame_indices": {frame_index}})
            else:
                matching_cluster["boxes"].append(box)
                matching_cluster["frame_indices"].add(frame_index)

    frame_count = len(boxes_per_frame)
    stable_boxes = []
    for cluster in clusters:
        presence_ratio = len(cluster["frame_indices"]) / frame_count
        if presence_ratio < min_presence_ratio:
            continue
        median_box = [float(v) for v in np.median(cluster["boxes"], axis=0)]
        stable_boxes.append({
            "box": median_box,
            "presence_ratio": round(presence_ratio, 3),
            "area": (median_box[2] - median_box[0]) * (median_box[3] - median_box[1]),
        })
    stable_boxes.sort(key=lambda b: b["area"], reverse=True)
    return stable_boxes


def get_stream_box(facecam_box: list) -> list | None:
    """
    Returns the normalised stream-content crop for a normalised facecam box, or None
    when the cam straddles the centre on both axes.

    Per axis (centre = 0.5): a cam fully on one side gives a crop from the cam's inner
    edge to the same distance past the centre (symmetric about the centre); a cam that
    straddles the centre gives the full axis.
    """

    def get_axis_span(cam_start: float, cam_end: float) -> tuple:
        if cam_end < 0.5:
            return cam_end, 1.0 - cam_end, False
        if cam_start > 0.5:
            return 1.0 - cam_start, cam_start, False
        return 0.0, 1.0, True

    x1, x2, is_x_straddling = get_axis_span(cam_start=facecam_box[0], cam_end=facecam_box[2])
    y1, y2, is_y_straddling = get_axis_span(cam_start=facecam_box[1], cam_end=facecam_box[3])
    if is_x_straddling and is_y_straddling:
        return None
    return [x1, y1, x2, y2]


def detect_facecam(
    input_paths: list,
    facecam_index: int = 0,
    sample_count: int = 12,
    sampling_window_seconds: float = 180.0,
    iou_threshold: float = 0.8,
    min_presence_ratio: float = 0.75,
    min_confidence: float = 0.3,
    model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
) -> dict:
    """
    Returns {"facecam": <stable box or None>, "stream_box": <crop or None>, "stable_facecams": [...], ...}.
    Box coordinates are normalised [x1, y1, x2, y2] in 0-1.
    """
    started_at = time.time()
    frames = read_sample_frames(
        input_paths=input_paths,
        sample_count=sample_count,
        sampling_window_seconds=sampling_window_seconds,
    )
    webcam_model = load_webcam_model(model_cache_dir=model_cache_dir)

    boxes_per_frame = []
    for frame in frames:
        frame_h, frame_w = frame.shape[:2]
        prediction = webcam_model.predict(source=frame, imgsz=640, conf=min_confidence, device="cpu", verbose=False)[0]
        frame_boxes = []
        for x1, y1, x2, y2 in prediction.boxes.xyxy.cpu().numpy().tolist():
            frame_boxes.append([x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h])
        boxes_per_frame.append(frame_boxes)

    stable_facecams = get_stable_facecam_boxes(
        boxes_per_frame=boxes_per_frame,
        iou_threshold=iou_threshold,
        min_presence_ratio=min_presence_ratio,
    )
    selected_facecam = stable_facecams[facecam_index] if facecam_index < len(stable_facecams) else None
    return {
        "facecam": selected_facecam,
        "stream_box": get_stream_box(facecam_box=selected_facecam["box"]) if selected_facecam else None,
        "stable_facecams": stable_facecams,
        "sampled_frames": len(frames),
        "detections_per_frame": [len(b) for b in boxes_per_frame],
        "elapsed_seconds": round(time.time() - started_at, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect the streamer facecam overlay box (normalised 0-1).")
    parser.add_argument("--input", "-i", required=True, nargs="+", help="Video (sampled over time) or still images")
    parser.add_argument("--facecam-index", type=int, default=0, help="Which stable cam to pick, largest first (default 0)")
    parser.add_argument("--samples", type=int, default=12, help="Frames sampled from a video (default 12)")
    parser.add_argument("--window", type=float, default=180.0, help="Seconds from the start to sample over (default 180)")
    args = parser.parse_args()

    try:
        result = detect_facecam(
            input_paths=args.input,
            facecam_index=args.facecam_index,
            sample_count=args.samples,
            sampling_window_seconds=args.window,
        )
        result["status"] = "success"
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
