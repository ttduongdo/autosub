"""Stage 2: Demucs vocal separation.

TODO:
- run Demucs on extracted audio
- produce isolated vocal stem, keep original mix
"""

from pathlib import Path
import subprocess


OUTPUT_DIR = Path(__file__).parents[1] / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def separate_vocals(audio_path: Path) -> tuple[Path, Path, Path]:
    """Run Demucs to separate vocals from the input audio.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        (vocals_path, instrumental_path, original_mix_path). Demucs's
        --two-stems=vocals mode already produces both vocals.wav and
        no_vocals.wav (the instrumental) in one run. original_mix_path is
        audio_path itself, returned unchanged for callers that want to
        compare transcription with vs. without vocal isolation.
    """
    model = "htdemucs"
    try:
        subprocess.run(
            [
                "demucs",
                "-n", model,
                "--two-stems", "vocals",
                "-o", str(OUTPUT_DIR),
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error with separating vocals: {e.stderr.decode()}")
        raise

    track_dir = OUTPUT_DIR / model / audio_path.stem
    vocals_path = track_dir / "vocals.wav"
    instrumental_path = track_dir / "no_vocals.wav"
    for path in (vocals_path, instrumental_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected Demucs output at {path}, but it wasn't created.")

    return vocals_path, instrumental_path, audio_path