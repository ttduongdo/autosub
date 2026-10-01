"""Tests for pipeline.align (stage 4)."""
import wave

from pipeline.align import align_segments
from pipeline.transcribe import transcribe


def test_align_segments_produces_ordered_word_timestamps(sample_audio_path, sample_transcript):
    segments = transcribe(sample_audio_path)
    words = align_segments(sample_audio_path, segments)

    assert isinstance(words, list)
    assert len(words) > 0

    with wave.open(str(sample_audio_path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()

    prev_end = -1.0
    for w in words:
        assert set(w.keys()) == {"word", "start", "end"}
        assert w["start"] >= 0
        assert w["end"] >= w["start"]
        assert w["end"] <= duration + 0.5  # small slack for rounding
        assert w["start"] >= prev_end - 0.05  # words should not overlap meaningfully
        prev_end = w["end"]

    recognized_words = [w["word"].lower() for w in words]
    expected_words = sample_transcript.lower().split()
    # forced alignment should recover roughly the same word count as the transcript
    assert abs(len(recognized_words) - len(expected_words)) <= 2
