"""Tests for finetune.train.

These test mechanical correctness (shapes, masking, splitting, freezing),
not training quality — that's what finetune/evaluate.py's WER comparison
is for. Fast tests use a loaded WhisperProcessor/model (real, but small and
CPU-friendly) against the project's existing tests/fixtures/sample.wav.

test_train_runs_one_step_without_crashing is marked @pytest.mark.train and
deselected by default (see pytest.ini) — it's a real training step, not
mocked, so it's slow enough to keep out of the default fast suite. Run
explicitly with: pytest -m train
"""
import json
from pathlib import Path

import pytest
import torch
from transformers import WhisperProcessor

from finetune.train import (
    _build_hf_dataset,
    _build_lora_model,
    _DataCollatorSpeechSeq2SeqWithPadding,
    _group_into_chunks,
    load_manifest,
    train,
)

SAMPLE_AUDIO = Path(__file__).parent / "fixtures" / "sample.wav"
MODEL_ID = "openai/whisper-base"


@pytest.fixture(scope="module")
def processor():
    return WhisperProcessor.from_pretrained(MODEL_ID, task="transcribe")


# --- load_manifest ---

def test_load_manifest_splits_into_train_and_eval(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([
        {"song_id": "1", "split": "train"},
        {"song_id": "2", "split": "eval"},
        {"song_id": "3", "split": "train"},
    ]))

    train_records, eval_records = load_manifest(manifest_path)

    assert [r["song_id"] for r in train_records] == ["1", "3"]
    assert [r["song_id"] for r in eval_records] == ["2"]


def test_load_manifest_handles_empty_split(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([{"song_id": "1", "split": "train"}]))

    train_records, eval_records = load_manifest(manifest_path)

    assert len(train_records) == 1
    assert eval_records == []


# --- _group_into_chunks ---

def test_group_into_chunks_splits_long_song_into_windows():
    # 30 lines, 4s apart, 3.5s each -> spans ~119.5s total
    segments = [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": f"line {i}"} for i in range(30)]

    chunks = _group_into_chunks(segments, chunk_duration=30.0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["end"] - chunk["start"] <= 30.0 + 1e-6
    # chunks should be in order and cover the whole song without gaps in text
    all_text = " ".join(c["text"] for c in chunks)
    assert "line 0" in all_text
    assert "line 29" in all_text


def test_group_into_chunks_keeps_short_song_as_one_chunk():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]
    chunks = _group_into_chunks(segments, chunk_duration=30.0)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 4.0


def test_group_into_chunks_handles_empty_segments():
    assert _group_into_chunks([]) == []


def test_group_into_chunks_handles_single_oversized_segment():
    # one segment longer than chunk_duration becomes its own chunk, not split further
    segments = [{"start": 0.0, "end": 45.0, "text": "a very long held note"}]
    chunks = _group_into_chunks(segments, chunk_duration=30.0)
    assert len(chunks) == 1
    assert chunks[0]["end"] == 45.0


# --- _build_hf_dataset ---

def test_build_hf_dataset_produces_correct_shapes(processor):
    records = [{
        "vocals_path": str(SAMPLE_AUDIO),
        "lyrics_segments": [
            {"start": 0.0, "end": 0.5, "text": "hello world"},
            {"start": 0.5, "end": 1.0, "text": "this is a test"},
        ],
    }]

    dataset = _build_hf_dataset(records, processor)

    assert dataset.column_names == ["input_features", "labels"]
    example = dataset[0]
    # Whisper's log-mel spectrogram: 80 mel bins x 3000 frames (30s window)
    assert len(example["input_features"]) == 80
    assert len(example["input_features"][0]) == 3000
    # labels should start with Whisper's special tokens (bos, language, no-timestamps)
    assert len(example["labels"]) > 0


def test_build_hf_dataset_produces_one_example_per_chunk_not_per_song(processor):
    # a song whose segments span more than 30s should yield multiple examples
    long_segments = [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": f"line {i}"} for i in range(10)]
    records = [{"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": long_segments}]

    dataset = _build_hf_dataset(records, processor)

    assert len(dataset) > 1


def test_build_hf_dataset_handles_multiple_records(processor):
    records = [
        {"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "first example"}]},
        {"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "second example here"}]},
    ]

    dataset = _build_hf_dataset(records, processor)

    assert len(dataset) == 2
    # different text lengths should produce different label lengths
    assert len(dataset[0]["labels"]) != len(dataset[1]["labels"]) or \
        dataset[0]["labels"] != dataset[1]["labels"]


# --- _DataCollatorSpeechSeq2SeqWithPadding ---

def test_collator_pads_labels_and_masks_with_minus_100(processor):
    records = [
        {"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "short"}]},
        {"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "a much longer sentence with many more words in it"}]},
    ]
    dataset = _build_hf_dataset(records, processor)
    collator = _DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    batch = collator([dataset[0], dataset[1]])

    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].shape[0] == 2
    # the shorter sequence's padded tail should be masked with -100, not
    # left as a real token id or as the tokenizer's pad token
    short_labels = batch["labels"][0]
    assert (short_labels == -100).any(), "expected padded positions to be masked with -100"


def test_collator_handles_single_example(processor):
    records = [{"vocals_path": str(SAMPLE_AUDIO), "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "hello world"}]}]
    dataset = _build_hf_dataset(records, processor)
    collator = _DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    batch = collator([dataset[0]])

    assert batch["input_features"].shape == (1, 80, 3000)
    assert batch["labels"].shape[0] == 1
    # no padding needed with a single example, so no -100 expected
    assert not (batch["labels"][0] == -100).any()


# --- _build_lora_model ---

def test_build_lora_model_freezes_most_parameters():
    model = _build_lora_model(MODEL_ID)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    assert trainable > 0
    assert trainable < total
    # LoRA should only touch a small fraction of total parameters
    assert trainable / total < 0.05


def test_build_lora_model_only_targets_attention_projections():
    model = _build_lora_model(MODEL_ID)

    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]

    assert len(trainable_param_names) > 0
    assert all("lora" in name for name in trainable_param_names)
    assert all(("q_proj" in name or "v_proj" in name) for name in trainable_param_names)


# --- Live smoke test: one real training step ---

@pytest.mark.train
def test_train_runs_one_step_without_crashing(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([
        {
            "vocals_path": str(SAMPLE_AUDIO),
            "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "hello world this is a test"}],
            "split": "train",
        },
        {
            "vocals_path": str(SAMPLE_AUDIO),
            "lyrics_segments": [{"start": 0.0, "end": 0.5, "text": "another short example"}],
            "split": "eval",
        },
    ]))
    output_dir = tmp_path / "checkpoint"

    result_dir = train(
        manifest_path=manifest_path,
        output_dir=output_dir,
        base_model_id=MODEL_ID,
        batch_size=1,
        max_steps=1,
    )

    assert result_dir == output_dir
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "processor_config.json").exists()
