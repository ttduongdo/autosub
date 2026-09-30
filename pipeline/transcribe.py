"""Stage 3: Whisper transcription (checkpoint-agnostic: baseline or fine-tuned).

TODO:
- load Whisper model via Hugging Face transformers
- transcribe audio to segment-level text + timestamps
- support swapping in a fine-tuned checkpoint
"""
from transformers import pipeline, Pipeline
from pathlib import Path
import torch


# Example usage:
# pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

_PIPELINE_CACHE: dict[str, Pipeline] = {}

def _get_pipeline(model_id: str):
    if model_id not in _PIPELINE_CACHE:
        device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1
        _PIPELINE_CACHE[model_id] = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
            return_timestamps=True,
        )
    return _PIPELINE_CACHE[model_id]

def transcribe(audio_path: Path, model_id: str = "openai/whisper-base") -> list[dict]:
    """Transcribe the given audio file using the specified Whisper model.

    Args:
        audio_path: Path to the input audio file.
        model_id: Hugging Face model ID for the Whisper model.

    Returns:
        A list of segments, each {"text": str, "start": float, "end": float}.
    """
    pipe = _get_pipeline(model_id)
    result = pipe(str(audio_path))
    return [
        {
            "text": segment["text"].strip(),
            "start": segment["timestamp"][0],
            "end": segment["timestamp"][1],
        }
        for segment in result["chunks"]
    ]