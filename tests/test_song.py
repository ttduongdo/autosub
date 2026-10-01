"""Full-pipeline smoke test against a real downloaded song.

Slow, needs network, and has no known ground-truth transcript — this checks
that the pipeline runs end-to-end on real music and produces plausible
output, not exact correctness (see test_ingest/test_transcribe/test_align
for ground-truth checks against synthesized speech).

Skipped by default. Run explicitly with: pytest -m song
Requires SONG_TEST_URL in tests/conftest.py to point at a URL you have
rights to use.
"""
import pytest

from pipeline.align import align_segments
from pipeline.separate import separate_vocals
from pipeline.transcribe import transcribe

pytestmark = pytest.mark.song


def test_full_pipeline_on_real_song(song_audio_path):
    vocals_path, mix_path = separate_vocals(song_audio_path)
    assert vocals_path.exists()
    assert mix_path == song_audio_path

    segments = transcribe(vocals_path)
    assert len(segments) > 0
    assert any(segment["text"].strip() for segment in segments)

    words = align_segments(vocals_path, segments)
    assert len(words) > 0

    prev_end = -1.0
    for w in words:
        assert w["start"] >= 0
        assert w["end"] >= w["start"]
        assert w["start"] >= prev_end - 0.05
        prev_end = w["end"]
