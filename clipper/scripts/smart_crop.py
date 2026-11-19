"""
smart_crop.py - Intelligent 9:16 vertical video reframing for YouTube Shorts.
Features:
- MediaPipe Face & Subject Detection with EMA (Exponential Moving Average) temporal smoothing.
- Micro-jitter suppression via motion deadband.
- Automatic fallback to 1080x1920 blurred-background pillarbox layout when no face
  is detected for > 1.5 seconds (ideal for gaming, screen shares, and B-roll).
- Cinematic cross-dissolve transition between tracked face and pillarbox layouts.
- Streams processed frames directly into FFmpeg for hardware-accelerated H.264 encode.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import numpy as np

FFMPEG_CANDIDATES = [
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    r"C:\Users\Bhuvan Thanai\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
    "ffmpeg",
]


def get_ffmpeg_bin():
    for c in FFMPEG_CANDIDATES:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return "ffmpeg"


FFMPEG_BIN = get_ffmpeg_bin()


def check_nvenc_support():
    try:
        cmd = [FFMPEG_BIN, "-encoders"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def make_stacked_layout(frame, faces, target_w=1080, target_h=1920):
    """
    Stacks two speakers (podcast/interview) or speaker + center gameplay into a 1080x1920 frame.
    Top half: 1080x960, Bottom half: 1080x960.
    """
    orig_h, orig_w = frame.shape[:2]
    half_h = target_h // 2  # 960
    
    # Calculate crop box for 1080x960 (aspect 9:8)
    sub_crop_h = orig_h
    sub_crop_w = int(round(orig_h * (target_w / half_h)))
    if sub_crop_w % 2 != 0:
        sub_crop_w += 1
    sub_crop_w = min(orig_w, sub_crop_w)
    max_x = max(0, orig_w - sub_crop_w)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    if len(faces) >= 2:
        # Sort faces left to right
        sorted_faces = sorted(faces, key=lambda f: f["cx"])
        f1, f2 = sorted_faces[0], sorted_faces[1]
        
        # Crop 1 (Top)
        x1 = max(0, min(max_x, int(f1["cx"] - sub_crop_w / 2)))
        slice1 = frame[:, x1 : x1 + sub_crop_w]
        canvas[0:half_h, :] = cv2.resize(slice1, (target_w, half_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop 2 (Bottom)
        x2 = max(0, min(max_x, int(f2["cx"] - sub_crop_w / 2)))
        slice2 = frame[:, x2 : x2 + sub_crop_w]
        canvas[half_h:target_h, :] = cv2.resize(slice2, (target_w, half_h), interpolation=cv2.INTER_LINEAR)
    elif len(faces) == 1:
        # Streamer mode: Face on top, center context on bottom
        f1 = faces[0]
        x1 = max(0, min(max_x, int(f1["cx"] - sub_crop_w / 2)))
        slice1 = frame[:, x1 : x1 + sub_crop_w]
        canvas[0:half_h, :] = cv2.resize(slice1, (target_w, half_h), interpolation=cv2.INTER_LINEAR)
        
        # Center context for bottom
        x2 = max(0, min(max_x, (orig_w - sub_crop_w) // 2))
        slice2 = frame[:, x2 : x2 + sub_crop_w]
        canvas[half_h:target_h, :] = cv2.resize(slice2, (target_w, half_h), interpolation=cv2.INTER_LINEAR)
    else:
        return make_blurred_pillarbox(frame, target_w, target_h)
        
    # Add divider line
    canvas[half_h - 2 : half_h + 2, :] = (30, 30, 30)
    return canvas


def make_blurred_pillarbox(frame, target_w=1080, target_h=1920):
    """
    Creates a 1080x1920 canvas with blurred background and centered foreground.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background: Fast downsample + GaussianBlur + upsample
    down_w = 135
    down_h = int(down_w * target_h / target_w)
    bg_small = cv2.resize(frame, (down_w, down_h), interpolation=cv2.INTER_AREA)
    bg_blurred_small = cv2.GaussianBlur(bg_small, (21, 21), 10)
    bg = cv2.resize(bg_blurred_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Dim background slightly for contrast
    bg = cv2.convertScaleAbs(bg, alpha=0.75, beta=-10)
    
    # 2. Foreground: Scale to fit target width (1080)
    fg_w = target_w
    fg_h = int(round(orig_h * (target_w / orig_w)))
    if fg_h % 2 != 0:
        fg_h += 1
    fg_scaled = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR)
    
    # Center vertically
    y_offset = (target_h - fg_h) // 2
    
    # Optional subtle top and bottom border shadow
    if y_offset > 4:
        bg[y_offset - 2: y_offset, :] = (20, 20, 20)
        bg[y_offset + fg_h: y_offset + fg_h + 2, :] = (20, 20, 20)
        
    bg[y_offset: y_offset + fg_h, 0:fg_w] = fg_scaled
    return bg


STREAMER_FACECAM_HEIGHT_RATIO = 1.0 / 3.0


def resolve_streamer_layout(
    analysis_video_path: str,
    frame_w: int,
    frame_h: int,
    target_h: int,
    facecam_index: int,
) -> dict | None:
    """
    Detects the facecam once (on the analysis video) and returns pixel boxes for the
    source frame, or None when no stable facecam (or no usable stream crop) is found.
    """
    from facecam_detect import detect_facecam

    detection = detect_facecam(input_paths=[analysis_video_path], facecam_index=facecam_index)
    print(
        f"[smart_crop] Facecam detection: {len(detection['stable_facecams'])} stable cam(s) from "
        f"{detection['sampled_frames']} frames in {detection['elapsed_seconds']}s",
        file=sys.stderr,
    )
    if detection["facecam"] is None:
        return None
    if detection["stream_box"] is None:
        print("[smart_crop] Facecam straddles the frame centre on both axes; no stream crop possible.", file=sys.stderr)
        return None

    def get_box_px(box_normalised):
        return [
            int(round(box_normalised[0] * frame_w)),
            int(round(box_normalised[1] * frame_h)),
            int(round(box_normalised[2] * frame_w)),
            int(round(box_normalised[3] * frame_h)),
        ]

    return {
        "facecam_box": get_box_px(detection["facecam"]["box"]),
        "stream_box": get_box_px(detection["stream_box"]),
        "facecam_height": int(round(target_h * STREAMER_FACECAM_HEIGHT_RATIO)),
        "stable_facecam_count": len(detection["stable_facecams"]),
    }


def make_streamer_frame(frame, streamer_layout: dict, target_w: int, target_h: int):
    """Facecam stretched over the top third, stream crop stretched over the bottom two thirds."""
    fx1, fy1, fx2, fy2 = streamer_layout["facecam_box"]
    sx1, sy1, sx2, sy2 = streamer_layout["stream_box"]
    facecam_height = streamer_layout["facecam_height"]
    canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
    canvas[:facecam_height] = cv2.resize(frame[fy1:fy2, fx1:fx2], (target_w, facecam_height), interpolation=cv2.INTER_LINEAR)
    canvas[facecam_height:] = cv2.resize(frame[sy1:sy2, sx1:sx2], (target_w, target_h - facecam_height), interpolation=cv2.INTER_LINEAR)
    return canvas


def process_smart_crop(
    input_path: str,
    output_path: str,
    start_sec: float = None,
    end_sec: float = None,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: float = 60.0,
    ema_alpha: float = 0.08,
    deadband_px: float = 20.0,
    no_face_timeout: float = 1.5,
    forced_mode: str = "auto",
    analysis_video_path: str | None = None,
    facecam_index: int = 0,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {input_path}")
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / orig_fps if orig_fps > 0 else 0
    
    start_time = max(0.0, start_sec) if start_sec is not None else 0.0
    end_time = min(video_duration, end_sec) if end_sec is not None else video_duration
    
    if end_time <= start_time:
        end_time = video_duration
        
    clip_duration = end_time - start_time
    start_frame = int(start_time * orig_fps)
    end_frame = int(end_time * orig_fps)
    frames_to_process = max(1, end_frame - start_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate 9:16 crop window dimensions from original frame
    crop_h = orig_h
    crop_w = int(round(orig_h * (target_w / target_h)))
    if crop_w % 2 != 0:
        crop_w += 1
    max_crop_x = max(0, orig_w - crop_w)
    
    streamer_layout = None
    if forced_mode == "streamer":
        streamer_layout = resolve_streamer_layout(
            analysis_video_path=analysis_video_path or input_path,
            frame_w=orig_w,
            frame_h=orig_h,
            target_h=target_h,
            facecam_index=facecam_index,
        )
        if streamer_layout is None:
            print("[smart_crop] No stable facecam found; falling back to auto layout.", file=sys.stderr)
            forced_mode = "auto"
        else:
            print(f"[smart_crop] Streamer layout: {streamer_layout}", file=sys.stderr)

    # Initialize MediaPipe Face Detection
    face_detector = None
    if streamer_layout is None:
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    # Tracking state
    smoothed_center_x = orig_w / 2.0
    smoothed_crop_x = max(0.0, min(float(max_crop_x), smoothed_center_x - (crop_w / 2.0)))
    no_face_duration = 0.0
    
    # Layout blend weight: 1.0 = Face Track 9:16, 0.0 = Blurred Pillarbox
    current_layout_weight = 1.0 if forced_mode == "face" else (0.0 if forced_mode == "pillarbox" else 1.0)
    
    # Check encoder support (nvenc vs libx264)
    use_nvenc = check_nvenc_support()
    vcodec = "h264_nvenc" if use_nvenc else "libx264"
    preset = "p4" if use_nvenc else "veryfast"
    print(f"[smart_crop] Video encoder: {vcodec} (preset={preset})", file=sys.stderr)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_crop_video = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"temp_crop_{int(time.time())}.mp4")
    
    target_fps = fps if fps is not None else 60.0
    
    # FFmpeg video-only pipe from raw RGB frames
    ffmpeg_video_cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", f"{orig_fps}",
        "-i", "-",
        "-r", f"{target_fps}",
        "-c:v", vcodec,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
    ]
    if not use_nvenc:
        ffmpeg_video_cmd.extend(["-crf", "18"])
    else:
        ffmpeg_video_cmd.extend(["-cq", "19"])
    ffmpeg_video_cmd.append(temp_crop_video)
    
    ffmpeg_proc = subprocess.Popen(ffmpeg_video_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    frame_idx = 0
    start_proc_time = time.time()
    
    try:
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            frame_idx += 1
            dt = 1.0 / orig_fps

            if streamer_layout is not None:
                ffmpeg_proc.stdin.write(make_streamer_frame(frame, streamer_layout, target_w, target_h).tobytes())
                continue

            # --- Face Detection ---
            has_face = False
            detected_face_x = None
            
            if forced_mode != "pillarbox":
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_results = face_detector.process(rgb_frame)
                
                if detection_results and detection_results.detections:
                    # Pick the largest bounding box (dominant speaker)
                    best_area = -1
                    best_cx = None
                    for detection in detection_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        area = bboxC.width * bboxC.height
                        if area > best_area:
                            best_area = area
                            best_cx = (bboxC.xmin + (bboxC.width / 2.0)) * orig_w
                    if best_cx is not None:
                        has_face = True
                        detected_face_x = best_cx
                        
            # --- State & Smoothing Update ---
            if has_face:
                no_face_duration = 0.0
                target_center_x = detected_face_x
                target_crop_x = max(0.0, min(float(max_crop_x), target_center_x - (crop_w / 2.0)))
                
                # Apply deadband
                delta = target_crop_x - smoothed_crop_x
                if abs(delta) > deadband_px:
                    smoothed_crop_x += ema_alpha * delta
                smoothed_crop_x = max(0.0, min(float(max_crop_x), smoothed_crop_x))
            else:
                no_face_duration += dt
                
            # Determine target layout weight
            if forced_mode == "pillarbox":
                target_layout_weight = 0.0
            elif forced_mode == "face":
                target_layout_weight = 1.0
            else:
                # Auto mode: if no face for > timeout, transition to pillarbox
                if no_face_duration > no_face_timeout:
                    target_layout_weight = 0.0
                else:
                    target_layout_weight = 1.0
                    
            # Smooth transition between layouts (approx 0.4s glide)
            transition_speed = 0.08
            current_layout_weight += (target_layout_weight - current_layout_weight) * transition_speed
            current_layout_weight = max(0.0, min(1.0, current_layout_weight))
            
            # --- Frame Generation ---
            # 1. Generate 9:16 Face-Track Frame if needed
            crop_frame = None
            if current_layout_weight > 0.01:
                int_crop_x = int(round(smoothed_crop_x))
                int_crop_x = max(0, min(max_crop_x, int_crop_x))
                cropped_slice = frame[:, int_crop_x : int_crop_x + crop_w]
                crop_frame = cv2.resize(cropped_slice, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
            # 2. Generate Blurred Pillarbox Frame if needed
            pillar_frame = None
            if current_layout_weight < 0.99:
                pillar_frame = make_blurred_pillarbox(frame, target_w, target_h)
                
            # 3. Composite according to current layout weight
            if current_layout_weight >= 0.99:
                out_frame = crop_frame
            elif current_layout_weight <= 0.01:
                out_frame = pillar_frame
            else:
                out_frame = cv2.addWeighted(
                    crop_frame, current_layout_weight,
                    pillar_frame, 1.0 - current_layout_weight,
                    0,
                )
                
            # Pipe to FFmpeg
            ffmpeg_proc.stdin.write(out_frame.tobytes())
            
    finally:
        cap.release()
        if face_detector is not None:
            face_detector.close()
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        
    proc_time = time.time() - start_proc_time
    print(f"[smart_crop] Video frames processed in {proc_time:.2f}s ({frame_idx / proc_time if proc_time > 0 else 0:.1f} fps).", file=sys.stderr)
    
    # Now mux audio from original input segment into the final output
    print("[smart_crop] Muxing synchronized audio from original video...", file=sys.stderr)
    mux_cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{clip_duration:.3f}",
        "-i", input_path,
        "-i", temp_crop_video,
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # Cleanup temp video file
    if os.path.exists(temp_crop_video):
        os.remove(temp_crop_video)
        
    print(f"[smart_crop] 9:16 export ready: {output_path}", file=sys.stderr)
    return {
        "status": "success",
        "output_path": os.path.abspath(output_path),
        "duration": round(clip_duration, 2),
        "resolution": f"{target_w}x{target_h}",
        "frames": frame_idx,
        "processing_time": round(proc_time, 2),
        "layout": "streamer" if streamer_layout is not None else forced_mode,
        "streamer_layout": streamer_layout,
    }


def main():
    parser = argparse.ArgumentParser(description="Smart 9:16 Face Tracking & Gameplay Pillarbox Crop")
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Path to output 9:16 video")
    parser.add_argument("--start", "-s", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, default=None, help="End time in seconds")
    parser.add_argument("--mode", "-m", choices=["auto", "face", "stacked", "split", "pillarbox", "streamer"], default="auto", help="Cropping mode")
    parser.add_argument("--analysis-video", default=None, help="Low-res copy of --input (same timeline) used for facecam detection in streamer mode")
    parser.add_argument("--facecam-index", type=int, default=0, help="Streamer mode: which stable facecam to use, largest first (default 0)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target framerate (default 60)")
    parser.add_argument("--alpha", type=float, default=0.08, help="EMA temporal smoothing factor")
    parser.add_argument("--timeout", type=float, default=1.5, help="Seconds without face before pillarbox fallback")
    args = parser.parse_args()
    
    try:
        res = process_smart_crop(
            input_path=args.input,
            output_path=args.output,
            start_sec=args.start,
            end_sec=args.end,
            fps=args.fps,
            ema_alpha=args.alpha,
            no_face_timeout=args.timeout,
            forced_mode=args.mode,
            analysis_video_path=args.analysis_video,
            facecam_index=args.facecam_index,
        )
        print(json.dumps(res, indent=2))
        sys.exit(0)
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
