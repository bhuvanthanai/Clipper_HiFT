"""
transcribe.py - Production-grade audio transcription with faster-whisper.
Extracts word-level timestamps, supports English/Hindi/Hinglish code-switching,
and automatically selects CUDA with seamless fallback to CPU int8.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))


def detect_compute_device():
    """
    Detects if CUDA is viable for ctranslate2; returns (device, compute_type).
    """
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            return "cuda", "float16"
    except Exception as e:
        print(f"[transcribe] CUDA check failed ({e}), falling back to CPU.", file=sys.stderr)
    return "cpu", "int8"


def get_whisper_model(model_size: str = "base", cpu_threads: int = 4):
    """
    Initializes faster_whisper.WhisperModel with automatic CUDA->CPU fallback.
    """
    from faster_whisper import WhisperModel
    
    device, compute_type = detect_compute_device()
    print(f"[transcribe] Attempting to load faster-whisper '{model_size}' on {device} ({compute_type})...", file=sys.stderr)
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type,
                             cpu_threads=cpu_threads if device == "cpu" else 0)
        print(f"[transcribe] Successfully initialized model on {device}.", file=sys.stderr)
        return model, device, compute_type
    except Exception as err:
        if device == "cuda":
            print(f"[transcribe] CUDA initialization failed: {err}. Retrying on CPU (int8)...", file=sys.stderr)
            model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
            print("[transcribe] Successfully initialized model on CPU.", file=sys.stderr)
            return model, "cpu", "int8"
        raise err


def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: str = None,
    output_path: str = None,
    cpu_threads: int = 4,
    beam_size: int = 1,
) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    start_time = time.time()
    model, device, compute_type = get_whisper_model(model_size, cpu_threads=cpu_threads)
    
    # Context prompt for code-switching (English, Hindi, Hinglish)
    initial_prompt = "English, Hindi, and Hinglish dialogue with code-switching, slang, gaming and colloquial speech."
    
    print(f"[transcribe] Transcribing: {video_path} (language={language or 'auto-detect'}, beam_size={beam_size})...", file=sys.stderr)
    try:
        segments_gen, info = model.transcribe(
            video_path,
            beam_size=beam_size,
            word_timestamps=True,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        detected_lang = info.language
        detected_prob = info.language_probability
        duration = info.duration
        print(f"[transcribe] Detected language: {detected_lang} (confidence: {detected_prob:.2f}), duration: {duration:.2f}s", file=sys.stderr)
        
        results = []
        total_words = 0
        
        for seg_idx, seg in enumerate(segments_gen):
            words_data = []
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3) if hasattr(w, "probability") else 1.0,
                    })
                    total_words += 1
                    
            results.append({
                "id": seg_idx,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words_data,
            })
    except Exception as cuda_err:
        if device == "cuda" and ("cublas" in str(cuda_err).lower() or "cuda" in str(cuda_err).lower() or "cannot be loaded" in str(cuda_err).lower()):
            print(f"[transcribe] CUDA execution failed ({cuda_err}). Seamlessly falling back to CPU int8...", file=sys.stderr)
            from faster_whisper import WhisperModel
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments_gen, info = model.transcribe(
                video_path,
                beam_size=beam_size,
                word_timestamps=True,
                language=language,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            detected_lang = info.language
            detected_prob = info.language_probability
            duration = info.duration
            print(f"[transcribe] Detected language: {detected_lang} (confidence: {detected_prob:.2f}), duration: {duration:.2f}s", file=sys.stderr)
            
            results = []
            total_words = 0
            for seg_idx, seg in enumerate(segments_gen):
                words_data = []
                if seg.words:
                    for w in seg.words:
                        words_data.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 3) if hasattr(w, "probability") else 1.0,
                        })
                        total_words += 1
                        
                results.append({
                    "id": seg_idx,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "words": words_data,
                })
        else:
            raise cuda_err
        
    elapsed = time.time() - start_time
    print(f"[transcribe] Transcription complete: {len(results)} segments, {total_words} words in {elapsed:.2f}s.", file=sys.stderr)
    
    payload = {
        "status": "success",
        "video_path": os.path.abspath(video_path),
        "device": device,
        "compute_type": compute_type,
        "model": model_size,
        "detected_language": detected_lang,
        "language_probability": round(detected_prob, 3),
        "duration": round(duration, 3),
        "total_segments": len(results),
        "total_words": total_words,
        "elapsed_seconds": round(elapsed, 2),
        "segments": results,
    }
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[transcribe] Transcript saved to: {output_path}", file=sys.stderr)
        
    return payload


def main():
    parser = argparse.ArgumentParser(description="Transcribe video with faster-whisper and word timestamps.")
    parser.add_argument("--input", "-i", required=True, help="Input video or audio file path")
    parser.add_argument("--model", "-m", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size")
    parser.add_argument("--language", "-l", default=None, help="Force language (e.g., 'en', 'hi') or omit for auto-detect")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path for transcript")
    parser.add_argument("--cpu-threads", "-c", default=4, type=int, help="CPU threads for inference (default: 4)")
    parser.add_argument("--beam-size", "-b", default=1, type=int, help="Beam size for decoding (1=greedy/fast, 5=accurate)")
    args = parser.parse_args()
    
    # Default output path if not provided
    if not args.output:
        base_name = Path(args.input).stem
        args.output = os.path.join("./temp_work", f"{base_name}_transcript.json")
        
    try:
        data = transcribe_video(
            video_path=args.input,
            model_size=args.model,
            language=args.language,
            output_path=args.output,
            cpu_threads=args.cpu_threads,
            beam_size=args.beam_size,
        )
        print(json.dumps({
            "status": "success",
            "transcript_file": os.path.abspath(args.output),
            "detected_language": data["detected_language"],
            "total_segments": data["total_segments"],
            "total_words": data["total_words"],
            "duration": data["duration"],
        }, indent=2))
        sys.exit(0)
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
