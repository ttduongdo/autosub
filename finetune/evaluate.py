"""WER evaluation: baseline vs. fine-tuned Whisper on held-out songs.

Loads the manifest's eval split (held-out artists, never seen during
training — see finetune/dataset.py's _split_by_artist), chunks each song's
lyrics_segments the same way finetune/train.py does for training (Whisper
can't process a whole song or an unbounded label sequence in one pass), and
transcribes each chunk with both the baseline checkpoint and the
fine-tuned LoRA checkpoint. Reports aggregate WER for each, so you can see
whether fine-tuning actually improved transcription accuracy on sung lyrics
versus the baseline.
"""
import json
from pathlib import Path

import jiwer
import torch
import torchaudio
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from finetune.train import CHUNK_DURATION, SAMPLE_RATE, _group_into_chunks

BASELINE_MODEL_ID = "openai/whisper-base"

# Normalizes case/punctuation/spacing before computing WER, so stylistic
# differences that aren't real transcription errors (e.g. "Telling" vs
# "telling", a trailing comma, double spaces) don't inflate the error rate.
# Confirmed empirically to matter: on one real eval chunk, raw WER was
# 63.6% for a transcription that was semantically close to the reference;
# normalized WER was 36.4% for the same pair. This does NOT paper over
# genuine word-level errors (e.g. "tellin'" vs "telling" still differ as
# distinct words after normalization — mishearing a word is still counted).
_WER_NORMALIZATION = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def load_eval_records(manifest_path: Path) -> list[dict]:
    """Load just the eval-split records from a dataset manifest."""
    records = json.loads(manifest_path.read_text())
    return [r for r in records if r["split"] == "eval"]


def _build_eval_chunks(records: list[dict]) -> list[dict]:
    """Expand eval records into individual (audio_path, start, end,
    reference_text) chunks, using the same ~30s grouping finetune.train
    uses for training — so eval exercises the model under the same input
    shape it was trained/will run on, not whole untruncated songs.
    """
    chunks = []
    for record in records:
        for chunk in _group_into_chunks(record["lyrics_segments"], chunk_duration=CHUNK_DURATION):
            chunks.append({
                "audio_path": record["vocals_path"],
                "start": chunk["start"],
                "end": chunk["end"],
                "reference_text": chunk["text"],
            })
    return chunks


def _load_chunk_waveform(audio_path: str, start: float, end: float) -> "torch.Tensor":
    """Load and slice one chunk's audio, mirroring finetune.train's
    _build_hf_dataset preprocessing exactly, so both eval and training see
    identically-prepared audio."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    start_sample = int(start * SAMPLE_RATE)
    end_sample = min(int(end * SAMPLE_RATE), waveform.shape[1])
    return waveform[0, start_sample:end_sample]


def _transcribe_chunk(waveform: "torch.Tensor", model, processor: WhisperProcessor) -> str:
    """Run one audio chunk through a Whisper model, return its transcription."""
    inputs = processor.feature_extractor(
        waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    with torch.inference_mode():
        predicted_ids = model.generate(inputs.input_features, max_length=225)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def _load_baseline_model(model_id: str = BASELINE_MODEL_ID) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(model_id, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.forced_decoder_ids = None
    model.eval()
    return model, processor


def _load_finetuned_model(checkpoint_dir: Path, base_model_id: str = BASELINE_MODEL_ID) -> tuple[PeftModel, WhisperProcessor]:
    """Load the fine-tuned checkpoint: fresh base model + the saved LoRA
    adapter on top. The processor is loaded from the checkpoint directory
    (saved alongside the adapter by finetune.train) rather than re-fetched
    from the base model id, though for this project they're equivalent.
    """
    processor = WhisperProcessor.from_pretrained(checkpoint_dir, task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    base_model.generation_config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    return model, processor


def evaluate_model(chunks: list[dict], model, processor: WhisperProcessor) -> dict:
    """Transcribe every chunk with the given model, compute aggregate WER
    against the reference lyric text (after normalizing case/punctuation/
    spacing on both sides — see _WER_NORMALIZATION). Returns {wer,
    predictions, references} with the original, unnormalized text in
    predictions/references for readability when inspecting results.
    """
    references = []
    predictions = []
    for chunk in chunks:
        waveform = _load_chunk_waveform(chunk["audio_path"], chunk["start"], chunk["end"])
        prediction = _transcribe_chunk(waveform, model, processor)
        references.append(chunk["reference_text"])
        predictions.append(prediction)

    normalized_references = [_WER_NORMALIZATION(r) for r in references]
    normalized_predictions = [_WER_NORMALIZATION(p) for p in predictions]
    wer = jiwer.wer(normalized_references, normalized_predictions)
    return {"wer": wer, "predictions": predictions, "references": references}


def compare_baseline_vs_finetuned(
    manifest_path: Path,
    checkpoint_dir: Path,
    base_model_id: str = BASELINE_MODEL_ID,
) -> dict:
    """Top-level: load the eval split, chunk it, transcribe with both the
    baseline and fine-tuned checkpoints, and report WER for each.

    Returns {baseline: {wer, predictions, references},
             finetuned: {wer, predictions, references}}.
    """
    eval_records = load_eval_records(manifest_path)
    if not eval_records:
        raise ValueError(
            f"No eval-split records found in {manifest_path} — "
            "nothing to evaluate against."
        )
    chunks = _build_eval_chunks(eval_records)

    baseline_model, baseline_processor = _load_baseline_model(base_model_id)
    baseline_result = evaluate_model(chunks, baseline_model, baseline_processor)

    finetuned_model, finetuned_processor = _load_finetuned_model(checkpoint_dir, base_model_id)
    finetuned_result = evaluate_model(chunks, finetuned_model, finetuned_processor)

    print(f"Baseline WER:    {baseline_result['wer']:.1%}")
    print(f"Fine-tuned WER:  {finetuned_result['wer']:.1%}")
    improvement = baseline_result["wer"] - finetuned_result["wer"]
    print(f"Improvement:     {improvement:+.1%} (positive = fine-tuning helped)")

    return {"baseline": baseline_result, "finetuned": finetuned_result}
