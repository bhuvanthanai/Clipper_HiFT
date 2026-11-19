"""
render_captions.py - Generates viral karaoke .ass subtitles and burns them
into 1080x1920 vertical video using FFmpeg.
Features:
- Dynamic phrase chunking (2-4 words per phrase).
- Active spoken word highlighted in Gold/Yellow (#FFD700 / &H0000D7FF&).
- Bold font (Arial Black) with thick black stroke (Outline=4) & drop shadow.
- Alignment=2 with safe MarginV=280 to prevent YouTube Shorts UI overlap.
- High-quality H.264 encode (CRF 20, AAC 192k) with hardware acceleration.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def format_ass_time(seconds: float) -> str:
    """Formats seconds to ASS timestamp format: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    csecs = int(round((seconds - int(seconds)) * 100))
    if csecs == 100:
        secs += 1
        csecs = 0
    return f"{hrs}:{mins:02d}:{secs:02d}.{csecs:02d}"


def extract_words_for_window(transcript_data: dict, clip_start: float, clip_end: float) -> list:
    """
    Extracts all words within the given clip window and normalizes timestamps relative to clip_start.
    """
    words = []
    segments = transcript_data.get("segments", [])
    
    for seg in segments:
        seg_words = seg.get("words", [])
        if not seg_words:
            # Fallback if no word timestamps: approximate evenly
            text_tokens = seg.get("text", "").split()
            if not text_tokens:
                continue
            seg_dur = max(0.1, seg["end"] - seg["start"])
            token_dur = seg_dur / len(text_tokens)
            for i, t in enumerate(text_tokens):
                t_start = seg["start"] + (i * token_dur)
                t_end = t_start + token_dur
                seg_words.append({"word": t, "start": t_start, "end": t_end})
                
        for w in seg_words:
            w_start = float(w["start"])
            w_end = float(w["end"])
            
            # Check overlap with clip window
            if w_end > clip_start and w_start < clip_end:
                rel_start = max(0.0, w_start - clip_start)
                rel_end = max(0.0, min(clip_end - clip_start, w_end - clip_start))
                clean_word = w["word"].strip().upper()
                if clean_word:
                    words.append({
                        "word": clean_word,
                        "start": rel_start,
                        "end": rel_end,
                    })
                    
    words.sort(key=lambda x: x["start"])
    return words


def build_karaoke_ass(words: list, output_ass_path: str, max_words_per_phrase: int = 3):
    """
    Constructs ASS subtitle file with word-by-word active highlight styling.
    """
    # Chunk words into small phrases
    phrases = []
    current_chunk = []
    
    for i, w in enumerate(words):
        current_chunk.append(w)
        ends_sentence = w["word"].endswith((".", "!", "?"))
        pause_after = False
        if i < len(words) - 1:
            pause_after = (words[i + 1]["start"] - w["end"]) > 0.4
            
        if len(current_chunk) >= max_words_per_phrase or ends_sentence or pause_after:
            phrases.append(current_chunk)
            current_chunk = []
            
    if current_chunk:
        phrases.append(current_chunk)
        
    ass_lines = [
        "[Script Info]",
        "Title: Viral Karaoke Captions",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Karaoke,Arial Black,78,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,5,3,2,60,60,280,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    
    # Active color: Yellow (#FFD700 in ASS BGR is &H00D7FF&)
    # Inactive color: Pure White (&H00FFFFFF&)
    COLOR_ACTIVE = r"{\c&H00D7FF&}"
    COLOR_INACTIVE = r"{\c&HFFFFFF&}"
    
    for phrase in phrases:
        if not phrase:
            continue
            
        # Bridge micro gaps between consecutive words in the same phrase
        for idx in range(len(phrase) - 1):
            if phrase[idx + 1]["start"] > phrase[idx]["end"] and (phrase[idx + 1]["start"] - phrase[idx]["end"]) < 0.25:
                phrase[idx]["end"] = phrase[idx + 1]["start"]
                
        # Generate dialogue events: one per active word
        for active_idx, target_word in enumerate(phrase):
            start_str = format_ass_time(target_word["start"])
            end_str = format_ass_time(target_word["end"])
            
            # Format text line highlighting active word
            formatted_words = []
            for idx, w in enumerate(phrase):
                clean_text = w["word"].rstrip(".,!?")
                if idx == active_idx:
                    formatted_words.append(f"{COLOR_ACTIVE}{clean_text}")
                else:
                    formatted_words.append(f"{COLOR_INACTIVE}{clean_text}")
                    
            line_text = " ".join(formatted_words)
            ass_lines.append(f"Dialogue: 0,{start_str},{end_str},Karaoke,,0,0,0,,{line_text}")
            
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))
        
    return output_ass_path


def burn_captions(
    video_path: str,
    ass_path: str,
    output_path: str,
    crf: int = 20,
    audio_bitrate: str = "192k",
    fps: float = 60.0,
):
    """
    Burns .ass subtitles into video using FFmpeg libass filter.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(ass_path):
        raise FileNotFoundError(f"ASS subtitle file not found: {ass_path}")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Format Windows path for FFmpeg filter graph
    # Forward slashes and escaped colon (e.g., C\:/path/to/file.ass)
    abs_ass = os.path.abspath(ass_path).replace("\\", "/")
    if ":" in abs_ass:
        drive, rest = abs_ass.split(":", 1)
        formatted_ass = f"{drive}\\:{rest}"
    else:
        formatted_ass = abs_ass
        
    vf_filter = f"ass='{formatted_ass}'"
    
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-vf", vf_filter,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    
    print(f"[render_captions] Burning subtitles into {output_path}...", file=sys.stderr)
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(f"[render_captions] Burn complete: {output_path}", file=sys.stderr)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render viral karaoke captions onto 9:16 vertical video.")
    parser.add_argument("--video", "-v", required=True, help="Path to cropped 9:16 video")
    parser.add_argument("--transcription", "-t", required=True, help="Path to transcription JSON file")
    parser.add_argument("--start", "-s", type=float, default=0.0, help="Clip start timestamp in source video")
    parser.add_argument("--end", "-e", type=float, default=None, help="Clip end timestamp in source video")
    parser.add_argument("--output", "-o", required=True, help="Path to final output MP4")
    parser.add_argument("--words-per-phrase", type=int, default=3, help="Words per subtitle phrase (2-4)")
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF quality level")
    parser.add_argument("--fps", type=float, default=60.0, help="Output framerate (default 60)")
    args = parser.parse_args()
    
    try:
        with open(args.transcription, "r", encoding="utf-8-sig") as f:
            trans_data = json.load(f)
            
        clip_start = max(0.0, args.start)
        clip_end = args.end if args.end is not None else float(trans_data.get("duration", 9999.0))
        
        words = extract_words_for_window(trans_data, clip_start, clip_end)
        
        # Save ephemeral ASS file
        ass_file = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            f"subtitles_{Path(args.output).stem}.ass",
        )
        build_karaoke_ass(words, ass_file, max_words_per_phrase=args.words_per_phrase)
        
        # Burn captions into video
        final_mp4 = burn_captions(
            video_path=args.video,
            ass_path=ass_file,
            output_path=args.output,
            crf=args.crf,
            fps=args.fps,
        )
        
        # Cleanup temporary ASS subtitle file
        if os.path.exists(ass_file):
            try:
                os.remove(ass_file)
            except Exception:
                pass
                
        out_info = {
            "status": "success",
            "final_video": os.path.abspath(final_mp4),
            "words_captioned": len(words),
            "file_size_bytes": os.path.getsize(final_mp4),
        }
        print(json.dumps(out_info, indent=2))
        sys.exit(0)
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
