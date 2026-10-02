"""Stage 5: export aligned transcript to SRT / VTT / word-timing JSON.

TODO:
- write .srt
- write .vtt
- write word-level JSON for karaoke UI
"""

from pathlib import Path
import json

def _format_timestamp(seconds: float, decimal_sep: str) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{decimal_sep}{millis:03}"

def write_srt(segments: list[dict], output_path: Path) -> Path:
    """One subtitle block per segment, SRT timecode format (HH:MM:SS,mmm)."""
    with open(output_path, "w") as f:
        for i, segment in enumerate(segments, start=1):
            end_time = segment["end"] if segment["end"] is not None else segment["start"] + 1.0
            start = _format_timestamp(segment["start"], ",")
            end = _format_timestamp(end_time, ",")
            f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")
    return output_path

def write_vtt(segments: list[dict], output_path: Path) -> Path:
    """Same idea, WebVTT format (HH:MM:SS.mmm, 'WEBVTT' header)."""
    with open(output_path, "w") as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(segments, start=1):
            start = _format_timestamp(segment["start"], ".")
            end_time = segment["end"] if segment["end"] is not None else segment["start"] + 1.0
            end = _format_timestamp(end_time, ".")
            f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")
    return output_path

def write_word_json(words: list[dict], output_path: Path) -> Path:
    """Word-level timing as JSON, for the Gradio karaoke UI to consume directly."""
    with open(output_path, "w") as f:
        json.dump(words, f, indent=2)
    return output_path

