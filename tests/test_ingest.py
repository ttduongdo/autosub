"""Tests for pipeline.ingest (stage 1)."""
import wave

import pytest

from pipeline.ingest import resolve_input


def test_resolve_input_local_file_produces_mono_wav(sample_audio_path):
    result = resolve_input(str(sample_audio_path), sample_rate=16000)

    assert result.exists()
    assert result.suffix == ".wav"

    with wave.open(str(result), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() > 0


def test_resolve_input_rejects_unrecognized_source():
    with pytest.raises(FileNotFoundError):
        resolve_input("not_a_real_file_or_url.mp3")
