"""LoRA fine-tuning script for Whisper on song lyrics.

Loads the manifest produced by finetune/dataset.py (vocals_path +
lyrics_segments pairs, pre-split by artist into train/eval), fine-tunes a
Whisper checkpoint with a LoRA adapter (peft), and saves the resulting
adapter weights — not the full model, just the small set of trainable LoRA
params on top of the frozen base checkpoint.

Whisper processes audio in ~30s windows and caps decoder labels at 448
tokens, so a full song can't be used as one training example — a 3-4
minute song's full lyrics alone can exceed 448 tokens. _group_into_chunks
groups each song's line-level lyrics_segments into ~30s windows, and
_build_hf_dataset slices the corresponding audio per chunk, producing one
training example per chunk instead of one per song.

Targets CUDA GPU training (fp16 mixed precision). See finetune/evaluate.py
for the baseline-vs-fine-tuned WER comparison that uses this checkpoint.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SAMPLE_RATE = 16000
CHUNK_DURATION = 30.0  # seconds — matches Whisper's native input window


def load_manifest(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """Load the dataset manifest, return (train_records, eval_records)."""
    records = json.loads(manifest_path.read_text())
    train_records = [r for r in records if r["split"] == "train"]
    eval_records = [r for r in records if r["split"] == "eval"]
    return train_records, eval_records


def _group_into_chunks(segments: list[dict], chunk_duration: float = CHUNK_DURATION) -> list[dict]:
    """Group a song's line-level lyrics_segments into ~chunk_duration-long
    windows, each with a combined start/end and concatenated text — e.g.
    turning 40 individual sung lines into ~6 chunks of ~30s each. This is
    what lets one song produce several training examples, each short
    enough to fit Whisper's native 30s window and 448-token label cap.

    A single segment longer than chunk_duration (unusual, but possible —
    e.g. a sparse verse with a long pause) becomes its own chunk rather
    than being split further.
    """
    if not segments:
        return []

    chunks = []
    current = [segments[0]]
    chunk_start = segments[0]["start"]

    for segment in segments[1:]:
        if segment["end"] - chunk_start > chunk_duration:
            chunks.append(current)
            current = [segment]
            chunk_start = segment["start"]
        else:
            current.append(segment)
    chunks.append(current)

    return [
        {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(s["text"] for s in chunk),
        }
        for chunk in chunks
    ]


def _build_hf_dataset(records: list[dict], processor: WhisperProcessor) -> Dataset:
    """Turn manifest records into a HF Dataset with input_features + labels.
    Each song's lyrics_segments are grouped into ~30s chunks
    (_group_into_chunks); each chunk becomes one training example, with its
    audio sliced from vocals_path and its concatenated line text as the
    label — so a single song contributes several (chunk audio, chunk text)
    examples rather than one (whole song, whole lyrics) example that would
    exceed Whisper's native window and label length limits.
    """
    chunk_audio_paths = []
    chunk_starts = []
    chunk_ends = []
    chunk_texts = []

    for record in records:
        chunks = _group_into_chunks(record["lyrics_segments"])
        for chunk in chunks:
            chunk_audio_paths.append(record["vocals_path"])
            chunk_starts.append(chunk["start"])
            chunk_ends.append(chunk["end"])
            chunk_texts.append(chunk["text"])

    dataset = Dataset.from_dict({
        "audio_path": chunk_audio_paths,
        "start": chunk_starts,
        "end": chunk_ends,
        "text": chunk_texts,
    })

    def _prepare(example):
        waveform, sample_rate = torchaudio.load(example["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        start_sample = int(example["start"] * SAMPLE_RATE)
        end_sample = min(int(example["end"] * SAMPLE_RATE), waveform.shape[1])
        chunk_waveform = waveform[0, start_sample:end_sample].numpy()

        example["input_features"] = processor.feature_extractor(
            chunk_waveform, sampling_rate=SAMPLE_RATE
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    return dataset.map(_prepare, remove_columns=dataset.column_names)


@dataclass
class _DataCollatorSpeechSeq2SeqWithPadding:
    """Pads input_features and labels independently (they have different
    natural lengths/padding conventions), and masks padded label positions
    with -100 so they're ignored by the loss.
    """
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


class _ProgressTimerCallback(TrainerCallback):
    """Prints a clear progress line (step count, % done, elapsed time,
    estimated remaining time, current loss) on every training log event —
    more visible than the default tqdm bar alone, and keeps working even
    when output is piped/redirected rather than shown in an interactive
    terminal.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return  # eval-only log events don't carry training loss/progress

        elapsed = time.monotonic() - self._start_time
        total_steps = state.max_steps
        current_step = state.global_step
        fraction_done = current_step / total_steps if total_steps else 0.0

        remaining = (elapsed / current_step) * (total_steps - current_step) if current_step else 0.0

        print(
            f"[train] step {current_step}/{total_steps} ({fraction_done:.0%}) "
            f"| loss {logs['loss']:.4f} "
            f"| elapsed {_format_duration(elapsed)} "
            f"| est. remaining {_format_duration(remaining)}"
        )


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_lora_model(base_model_id: str) -> PeftModel:
    """Load the base Whisper model and wrap it with a LoRA adapter —
    targets the attention projection layers, freezing everything else.
    """
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    return get_peft_model(model, lora_config)


def train(
    manifest_path: Path,
    output_dir: Path,
    base_model_id: str = "openai/whisper-base",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = -1,
    resume_from_checkpoint: str | None = None,
) -> Path:
    """Fine-tune base_model_id with LoRA on the manifest's train split,
    using the eval split for periodic validation. Saves the LoRA adapter
    (not the full model) to output_dir. Returns output_dir.

    max_steps, if set (> 0), overrides num_epochs and stops after that many
    training steps regardless of dataset size — used by the smoke test to
    run a single real step without waiting for a full epoch.

    resume_from_checkpoint, if set, picks up training from a previously
    saved checkpoint-N directory (e.g. "data/checkpoints/whisper-lora-v2/checkpoint-72")
    instead of starting over from the base model — skips re-running
    already-completed epochs/steps.
    """
    train_records, eval_records = load_manifest(manifest_path)

    processor = WhisperProcessor.from_pretrained(base_model_id, task="transcribe")
    train_dataset = _build_hf_dataset(train_records, processor)
    eval_dataset = _build_hf_dataset(eval_records, processor)

    model = _build_lora_model(base_model_id)
    data_collator = _DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if max_steps < 0 else "no",
        save_strategy="epoch" if max_steps < 0 else "no",
        predict_with_generate=True,
        generation_max_length=225,
        label_names=["labels"],
        remove_unused_columns=False,
        logging_steps=1,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[_ProgressTimerCallback()],
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return output_dir
