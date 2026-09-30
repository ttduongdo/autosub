"""Stage 4: word-level forced alignment for karaoke-style highlighting.

TODO:
- align transcript words to precise start/end timestamps
- output [{word, start, end}, ...]
"""

from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

_PROCESSOR_CACHE: dict[str, Wav2Vec2Processor] = {}
_MODEL_CACHE: dict[str, Wav2Vec2ForCTC] = {}
INPUT_DIR = Path(__file__).parents[1] / "data" / "processed"

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def align_segments(audio_path: Path, segments: list[dict], 
                   model_id: str = "facebook/wav2vec2-base-960h") -> list[dict]:
    """Top-level entry point: align every segment's words to precise timestamps."""
    model, processor = _get_alignment_model(model_id)
    waveform, sample_rate = _load_waveform(audio_path)
    words = []
    for segment in segments: 
        words.extend(_align_single_segment(waveform, sample_rate, segment, model, processor))

    return words

def _get_alignment_model(model_id: str):
    """Load (and cache) the wav2vec2 CTC model + processor, same caching pattern 
    as transcribe._get_pipeline."""
    if model_id not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[model_id] = Wav2Vec2Processor.from_pretrained(model_id)
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = Wav2Vec2ForCTC.from_pretrained(model_id)
    return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]

def _align_single_segment(waveform: "torch.Tensor",
                          sample_rate: int, 
                          segment: dict,
                          model,
                          processor,
                          ) -> list[dict]:
    """Slice audio for this segment's time range, run CTC forced alignment
    against segment['text'], return per-word timestamps offset by segment['start']."""
    start_sample = int(segment["start"] * sample_rate)
    end_sample = int(segment["end"] * sample_rate)
    segment_waveform = waveform[:, start_sample:end_sample]

    # 1. get emissions
    with torch.inference_mode():
        emissions, _ = model(segment_waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0] # drop batch dim -> [num_frames, vocab_size]

    # 2. tokenize the known transcript for this segment
    cleaned = _clean_text(segment["text"])
    tokens = processor.tokenizer(cleaned).input_ids # list[int]

    # 3. forced_align
    targets = torch.tensor([tokens], dtype=torch.int32)
    aligned_tokens, scores = torchaudio.functional.forced_align(
        emission.unsqueeze(0), targets, blank=processor.tokenizer.pad_token_id
    )

    # 4. merge repeated frames into per-token spans
    token_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    # 5. convert token spans -> word spans -> timestamps
    return _spans_to_words(processor, token_spans, cleaned, segment["start"], sample_rate, emission.shape[0], segment_waveform.shape[1])

def _clean_text(text: str) -> str:
    """Lowercase and remove punctuation from the text."""
    import re
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", "", text)
    return text

def _spans_to_words(processor, token_spans, text, segment_start, sample_rate, num_frames, num_samples) -> list[dict]:
    ratio = num_samples / num_frames / sample_rate # seconds per frame
    words = text.split(" ")
    word_spans = []
    current = []
    for span in token_spans:
        if span.token == processor.tokenizer.pad_token_id: # blank/pad token from CTC decoding (not a real character -> skip)
            continue
        char = processor.tokenizer.convert_ids_to_tokens(span.token)
        if char == "|": # wav2vec2 internal space token
            if current:
                word_spans.append(current)
                current = []
        else:
            current.append(span)
    if current:
        word_spans.append(current)

    results = []
    for word, span in zip(words, word_spans):
        results.append({
            "word": word,
            "start": segment_start + span[0].start * ratio,
            "end": segment_start + span[-1].end * ratio,
        })
    return results

def _load_waveform(audio_path: Path, target_sample_rate: int = 16000) -> tuple["torch.Tensor", int]:
    """Load an audio file as a waveform tensor and resample to the target sample rate."""
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # just in case, should already be mono

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return waveform, sample_rate

