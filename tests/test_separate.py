"""Tests for pipeline.separate (stage 2)."""
from pipeline.ingest import resolve_input
from pipeline.separate import separate_vocals


def test_separate_vocals_produces_vocal_stem_and_returns_original(sample_audio_path):
    normalized = resolve_input(str(sample_audio_path))

    vocals_path, mix_path = separate_vocals(normalized)

    assert vocals_path.exists()
    assert vocals_path.suffix == ".wav"
    assert mix_path == normalized
