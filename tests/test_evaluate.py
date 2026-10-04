"""Tests for finetune.evaluate's pure logic (chunking reuse, WER
normalization). Model-loading/inference paths aren't covered here — they're
exercised directly when you actually run compare_baseline_vs_finetuned.
"""
import jiwer

from finetune.evaluate import _build_eval_chunks, _WER_NORMALIZATION


def test_wer_normalization_ignores_case_and_punctuation():
    ref = "And I'm tellin' you, baby, this the part!"
    hyp = "and im telling you baby this the part"

    raw_wer = jiwer.wer(ref, hyp)
    normalized_wer = jiwer.wer(_WER_NORMALIZATION(ref), _WER_NORMALIZATION(hyp))

    assert normalized_wer < raw_wer


def test_wer_normalization_still_counts_real_word_differences():
    ref = "hello world"
    hyp = "goodbye planet"

    normalized_wer = jiwer.wer(_WER_NORMALIZATION(ref), _WER_NORMALIZATION(hyp))

    assert normalized_wer == 1.0  # every word wrong, normalization doesn't hide that


def test_build_eval_chunks_expands_segments_across_records():
    records = [
        {
            "vocals_path": "/fake/song1.wav",
            "lyrics_segments": [
                {"start": 0.0, "end": 2.0, "text": "hello"},
                {"start": 2.0, "end": 4.0, "text": "world"},
            ],
        },
        {
            "vocals_path": "/fake/song2.wav",
            "lyrics_segments": [
                {"start": 0.0, "end": 2.0, "text": "another song"},
            ],
        },
    ]

    chunks = _build_eval_chunks(records)

    assert len(chunks) == 2  # each short song stays as one chunk
    assert chunks[0]["audio_path"] == "/fake/song1.wav"
    assert chunks[0]["reference_text"] == "hello world"
    assert chunks[1]["audio_path"] == "/fake/song2.wav"
    assert chunks[1]["reference_text"] == "another song"
