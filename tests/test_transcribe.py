"""Tests for pipeline.transcribe (stage 3)."""
from pipeline.transcribe import transcribe


def test_transcribe_returns_segments_with_expected_words(sample_audio_path, sample_transcript):
    segments = transcribe(sample_audio_path)

    assert isinstance(segments, list)
    assert len(segments) > 0
    for segment in segments:
        assert set(segment.keys()) == {"text", "start", "end"}
        assert segment["end"] >= segment["start"]

    full_text = " ".join(s["text"] for s in segments).lower()
    for word in sample_transcript.split():
        assert word in full_text
