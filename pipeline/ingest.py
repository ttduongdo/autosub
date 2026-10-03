"""
TODO:
- resolve Spotify/Apple Music links to a playable source
- re-enable YouTube URL support here (see _download_audio's note below).
  finetune/dataset.py re-enabled YouTube search (ytsearch1:) for building
  training data after confirming 4/4 clean downloads with no bot-detection
  in live testing (2026-10) — but that was a small sample, so the main
  app's URL-based path was deliberately left untouched pending more
  evidence before trusting it for end users.
"""
import subprocess
from pathlib import Path

import yt_dlp

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _download_audio(url: str) -> Path:
    """Download audio for a URL via yt-dlp.
    Returns the raw downloaded file path.

    Note: YouTube is not reliably supported right now — its bot detection
    has been blocking yt-dlp even with browser cookies, with no workaround
    found so far (see project history as of 2026-10). SoundCloud and other
    platforms yt-dlp supports work fine. Revisit YouTube support if/when
    yt-dlp or YouTube's behavior changes.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(OUTPUT_DIR / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info_dict))



def _normalize_audio(src: Path, sample_rate: int) -> Path:
    """Convert src to a mono WAV at the target sample rate via ffmpeg."""
    dst = OUTPUT_DIR / f"{src.stem}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ac", "1",
                "-ar", str(sample_rate),
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error with normalizing audio: {e.stderr.decode()}")
        raise
    return dst


def resolve_input(input_source: str, sample_rate: int = 16000) -> Path:
    """
    Resolve the input source (local file or YouTube URL) and extract a
    normalized mono WAV for downstream processing.

    Args:
        input_source: Path to a local file, or a URL.
        sample_rate: Target sample rate for the output WAV.

    Returns:
        Path to the normalized mono WAV file.
    """
    input_path = Path(input_source)

    if input_path.is_file():
        raw_path = input_path
    elif input_source.startswith(("http://", "https://")):
        raw_path = _download_audio(input_source)
    else:
        raise FileNotFoundError(
            f"'{input_source}' is not an existing local file or a recognized URL."
        )

    return _normalize_audio(raw_path, sample_rate)
