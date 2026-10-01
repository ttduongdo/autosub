# Pipeline Reference

A function-by-function walkthrough of `pipeline/`, covering both the audio/ML
concept behind each step and what the code is literally doing. Meant as a
learning reference alongside the code — see [ARCHITECTURE.md](ARCHITECTURE.md)
for the overall system design and stage ordering.

Stage 5 (`export.py`) is not yet implemented, so it's not covered here.

---

## Tensor dimensions, start to finish

This is the part that trips people up most, so it gets its own section
before the stage-by-stage walkthrough. A PyTorch tensor's "shape" is just a
tuple describing how many elements it has along each axis — the question is
always "what does each axis *mean*" for the tensor at hand. Here's how
shapes evolve as audio flows through stages 1-4.

**Raw audio, as loaded (`torchaudio.load`)**
`waveform.shape == [channels, num_samples]` — a 2D tensor. For mono audio
at 16kHz, a 3-second clip is `[1, 48000]`: 1 channel, 48,000 individual
amplitude values (16,000 samples/sec × 3 sec). This is `align.py`'s
`_load_waveform` return value.

**Slicing a segment out of the full track**
`waveform[:, start_sample:end_sample]` keeps the same `[channels, ...]`
shape, just with fewer samples along the second axis — e.g. `[1, 32000]`
for a 2-second slice at 16kHz. This is where the stage-4 bug above came
from: if `start_sample >= end_sample` (segment timestamps outside the real
audio), the slice becomes `[1, 0]` — a tensor that exists but holds zero
samples. Nothing raises an error at slicing time; it only blows up later,
once something (the wav2vec2 model) tries to actually process a
zero-length input and finds it has no data to run its convolution over.

**Feeding audio into wav2vec2 (`model(segment_waveform)`)**
`transformers` models expect a **batch dimension** as the first axis, even
when you're only processing one clip at a time — that's why nothing above
ever has shape `[48000]` alone; it's always `[1, 48000]`, where that leading
`1` means "batch of 1." The model internally runs its convolutional feature
extractor over the sample axis, downsampling it from raw audio samples
(16,000/sec) to a much coarser sequence of "frames" (roughly 50/sec, i.e.
one frame per ~20ms) — this is the mechanism that turns continuous audio
into the discrete per-time-step grid CTC needs.

**Model output: logits / emissions**
`model(segment_waveform).logits` has shape
`[batch, num_frames, vocab_size]` — e.g. `[1, 150, 32]` for a 3-second clip
(roughly 150 frames at ~50 frames/sec) and wav2vec2-base-960h's 32-token
vocabulary. Each of the 150 "rows" is one frame's raw score for each of the
32 possible characters. `torch.log_softmax(..., dim=-1)` normalizes each
frame's 32 scores into log-probabilities that sum to 1 when exponentiated
— still the same shape, just turned into something probability-like.
`emission = emissions[0]` drops the batch dimension (since there's only
ever 1 item in the batch here), leaving `[num_frames, vocab_size]` — this
is the actual "emissions grid" forced alignment searches over.

**Target tokens**
`tokens = processor.tokenizer(cleaned).input_ids` is a flat Python list of
integers, one per character in the cleaned transcript (e.g. `"HI THERE"` →
8 tokens, including the space). `torch.tensor([tokens], dtype=torch.int32)`
wraps it in an extra list before converting to a tensor specifically to add
a batch dimension — giving shape `[1, num_characters]`, matching the
`[batch, ...]` convention `forced_align` expects, consistent with how the
emissions tensor also carries a batch dimension via `.unsqueeze(0)`.

**Forced alignment output**
`aligned_tokens` has shape `[1, num_frames]` — for every single frame, which
token (character) was most likely active there. This is naturally full of
repeats (the same letter is usually "active" for several consecutive
frames, since a 20ms slice is much shorter than how long a spoken letter
actually lasts). `merge_tokens` collapses those runs into one span per
distinct character occurrence — no longer a tensor at this point, but a
plain Python list of `TokenSpan` objects, each carrying its own start
frame, end frame, and confidence score.

**Why `ratio = num_samples / num_frames / sample_rate`**
This converts a frame index back into real seconds. `num_samples /
num_frames` gives "how many raw audio samples does one frame correspond
to" (since the model's internal downsampling from samples→frames is
roughly fixed-rate but not a round number you should hardcode), and
dividing that by `sample_rate` converts "samples per frame" into "seconds
per frame." Multiply any frame index by `ratio` and you get its timestamp
in seconds, relative to the start of whatever slice was fed into the
model — which is why `_spans_to_words` then adds `segment_start` on top, to
convert back to absolute time in the full track.

---

## Stage 1: `pipeline/ingest.py`

**Concept.** Before anything ML-related can happen, every input — a local
file or a YouTube URL — needs to become the same kind of audio data: a mono
(single-channel) WAV file at a known sample rate. Models like Whisper and
wav2vec2 are trained on audio at a specific sample rate (16kHz); feed them
audio at the wrong rate and "one second" of audio contains the wrong number
of samples, so the model's internal timing and pattern-matching breaks
silently rather than erroring. Mono matters because stereo (2-channel) audio
is really two separate waveforms (left/right speakers summed differently),
and speech/music models expect a single waveform, not two.

### `_download_audio(url: str) -> Path`

- **Code:** Calls `yt_dlp.YoutubeDL` with options asking for the best
  available audio-only stream, and downloads it to `data/raw/`. Returns the
  path to whatever file yt-dlp actually wrote (`ydl.prepare_filename`, since
  the real extension depends on what format was available for that URL).
- **Concept:** YouTube serves video in multiple quality/format tracks; `yt-dlp`
  picks the best audio-only one so you're not downloading/decoding video data
  you don't need. The output format varies (webm, m4a, etc.) — this function
  doesn't normalize it, it just gets *some* audio file onto disk.

### `_normalize_audio(src: Path, sample_rate: int) -> Path`

- **Code:** Shells out to `ffmpeg` with `-ac 1` (force 1 audio channel = mono)
  and `-ar {sample_rate}` (resample to the target rate), writing a new `.wav`
  file. Wraps the subprocess call in a try/except so a failure prints
  `ffmpeg`'s actual stderr before re-raising — otherwise you'd just see a bare
  `CalledProcessError` with no indication of what went wrong inside ffmpeg.
- **Concept:** This is the "normalization" step — collapsing whatever
  channel count and sample rate the source had into the single consistent
  format every downstream stage assumes. Resampling (`-ar`) is a real signal
  processing operation: ffmpeg is reconstructing the waveform at new sample
  points via interpolation, not just relabeling the file.

### `resolve_input(input_source: str, sample_rate: int = 16000) -> Path`

- **Code:** The public entry point. Checks whether `input_source` is an
  existing local file; if so, skips downloading. Otherwise, if it looks like
  an `http(s)://` URL, downloads it via `_download_audio`. Anything else
  raises `FileNotFoundError` with a clear message, rather than silently
  trying (and confusingly failing) to treat a typo'd path as a URL. Whichever
  path was taken, the result is passed through `_normalize_audio` — so *every*
  input, local or downloaded, ends up as the same mono/target-rate WAV.
- **Concept:** This function is the boundary between "arbitrary user input"
  and "clean audio the rest of the pipeline can trust." Every later stage can
  assume mono audio at a known sample rate without re-checking, because this
  function's contract guarantees it.

---

## Stage 2: `pipeline/separate.py`

**Concept.** A music recording is one waveform that's actually the *sum* of
many sources — vocals, drums, bass, other instruments — all mixed together
before it was ever saved as a file. You cannot "unmix" this with simple
signal processing (there's no clean mathematical inverse once things are
summed and the original stems are gone). Demucs is a neural network trained
on pairs of (mixed track, isolated stems) to *predict* what the isolated
vocal track probably sounded like, given only the mix. The output is an
estimate, not a perfect recovery — a reconstruction, not a recorded isolated
track. This matters for the pipeline because Whisper and the aligner do much
better on clean speech than on speech buried under instrumentation.

### `separate_vocals(audio_path: Path) -> tuple[Path, Path]`

- **Code:** Shells out to the `demucs` CLI with `-n htdemucs` (model choice)
  and `--two-stems vocals` (only split into "vocals" vs. "everything else,"
  rather than the full 4-way drums/bass/vocals/other split, which is slower
  and unnecessary here). Demucs writes its own output file layout
  (`<output_dir>/<model>/<track_name>/vocals.wav`), so the function
  reconstructs that expected path and checks it actually exists before
  returning — if Demucs's CLI behavior ever changes (different version,
  different flag), this fails loudly here instead of silently breaking
  whatever calls it next.
- **Concept:** `htdemucs` is Demucs's current default pretrained model —
  a "hybrid transformer" architecture (hence "ht"), trained on a large set of
  multitrack recordings. `--two-stems vocals` only runs the parts of the
  model's output needed to produce "vocals" + "everything else combined,"
  rather than fully separating all four stems. The function returns
  `audio_path` itself (unmodified) as the "original mix" — no new file is
  created for this, since the original normalized audio from stage 1 already
  *is* the mix; this just hands that same reference back so a caller can
  compare transcription with vs. without isolation later.

---

## Stage 3: `pipeline/transcribe.py`

**Concept.** Whisper is a sequence-to-sequence (encoder-decoder) model: it
reads a chunk of audio and *generates* text token-by-token, the same
architectural family as machine translation models. Because it's generating
free-form text rather than aligning to fixed time slices, its timestamps are
only accurate at the segment level (a few words to a sentence) — a side
effect of how it processes audio in fixed-size windows, not a precise
per-word measurement. This is the fundamental reason stage 4 (forced
alignment) has to exist as a separate step using a different kind of model.

### `_get_pipeline(model_id: str) -> Pipeline`

- **Code:** Lazily constructs (and caches, in `_PIPELINE_CACHE`) a Hugging
  Face `pipeline("automatic-speech-recognition", ...)` object for a given
  `model_id`. Picks a device at call time — CUDA GPU if available, Apple
  Silicon's MPS backend next, CPU (`-1`) as the fallback — so the same code
  runs correctly across different machines without hardcoding a device.
  Caching means repeated calls with the same `model_id` don't reload the
  (large) model weights from disk every time.
- **Concept:** The HF `pipeline()` helper bundles together the feature
  extractor (turns raw waveform into the spectrogram-like input Whisper
  expects), the model itself, and the decoding logic (turning Whisper's
  output tokens back into text) — doing this by hand would mean managing all
  three separately. `return_timestamps=True` tells Whisper's generation
  process to also emit the approximate start/end time of each decoded
  segment, not just the text.

### `transcribe(audio_path: Path, model_id: str = "openai/whisper-base") -> list[dict]`

- **Code:** Gets the cached pipeline for `model_id`, runs it on the audio
  file, and reshapes HF's raw output (`result["chunks"]`, a list of
  `{"text": ..., "timestamp": (start, end)}`) into this project's own flat
  shape: `[{"text": str, "start": float, "end": float}, ...]`. This
  reshaping matters so that `align.py` and `export.py` don't need to know
  anything about Hugging Face's specific dict format — they only ever see
  this project's own segment shape.
- **Concept:** `model_id` being a parameter (not hardcoded) is what makes
  this function "checkpoint-agnostic" — swapping in a fine-tuned Whisper
  checkpoint later (once `finetune/train.py` produces one) means passing a
  different string here, with zero other code changes. The segment-level
  text here is Whisper's best guess at *what* was said; stage 4 is entirely
  about figuring out more precisely *when* each word within that text was
  actually spoken.

---

## Stage 4: `pipeline/align.py`

**Concept.** This is the step that produces real word-level timestamps for
karaoke-style highlighting, using a fundamentally different kind of model
than Whisper: a CTC (Connectionist Temporal Classification) model,
`wav2vec2`. Unlike Whisper's free-form generation, a CTC model outputs a
prediction *for every fixed time-slice of audio* (a "frame," here ~20ms) —
effectively a grid of "which character is probably being spoken right now,"
one row per frame. Because you get one prediction per time step, and you
already know the correct transcript (from Whisper, stage 3), you can do
**forced alignment**: find the most likely path through that frame-by-frame
grid that spells out the known text, in order. This is fundamentally a
constrained search, not a transcription — you're never asking the model
"what did it hear," only "when did it most likely hear each known word."

### `_get_alignment_model(model_id: str) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]`

- **Code:** Same lazy-load-and-cache pattern as stage 3's `_get_pipeline`,
  but for two objects: the CTC model itself (`Wav2Vec2ForCTC`) and its
  `Wav2Vec2Processor`, which bundles the audio feature extractor and the
  character-level tokenizer this model's vocabulary uses.
- **Concept:** wav2vec2's vocabulary (for `facebook/wav2vec2-base-960h`) is
  tiny and specific: 32 tokens total — uppercase `A`-`Z`, apostrophe, a
  special `|` token that represents a space, and a handful of
  special/blank tokens (`<pad>`, `<s>`, `</s>`, `<unk>`). It has no digits,
  no lowercase, no other punctuation — a fact that directly shapes
  `_clean_text` below.

### `_load_waveform(audio_path: Path, target_sample_rate: int = 16000) -> tuple[Tensor, int]`

- **Code:** Loads the audio file into a PyTorch tensor via `torchaudio.load`,
  defensively collapses to mono if more than one channel is present, and
  resamples if the file's actual sample rate doesn't match the target.
- **Concept:** This is mostly a safety net — stage 1 already guarantees mono
  audio at a known rate, but this function doesn't assume that guarantee
  holds (e.g. if called directly against a Demucs vocals file that wasn't
  routed back through `ingest.py`). 16kHz specifically matters because that's
  the rate `wav2vec2-base-960h` was trained on; mismatch it and the model's
  frame-to-time-duration assumptions silently break.

### `_clean_text(text: str) -> str`

- **Code:** Uppercases the text, then strips out every character that isn't
  `A`-`Z`, an apostrophe, or a space using a regex.
- **Concept:** This maps Whisper's natural-language output text onto exactly
  the character set wav2vec2's vocabulary can represent. If you skipped this
  (e.g. left in a comma, or a digit), the tokenizer would map that character
  to an unknown/pad token, which would corrupt the alignment for that
  segment — forced alignment requires the *target sequence* to be expressible
  in the model's vocabulary.

### `_align_single_segment(waveform, sample_rate, segment, model, processor) -> list[dict]`

This is where the actual CTC forced-alignment machinery runs, for one
Whisper segment at a time. Step by step:

1. **Slicing:** converts the segment's `start`/`end` (in seconds) into sample
   indices and slices just that span out of the full-track waveform — so the
   model only processes this segment's audio, not the whole track at once.
2. **Emissions:** runs the sliced audio through the wav2vec2 model
   (`torch.inference_mode()` disables gradient tracking since this is
   inference, not training — saves memory and time), then applies
   `log_softmax` to turn the model's raw output scores into log-probabilities
   per character per frame. This probability grid is called the "emission."
3. **Tokenizing the known text:** cleans the segment's Whisper text and
   converts it into the sequence of token IDs the forced-alignment search
   will try to match against the emission grid.
4. **`torchaudio.functional.forced_align`:** the actual alignment algorithm
   (a dynamic-programming search, conceptually similar to Viterbi decoding)
   — given the emission grid and the known token sequence, finds the most
   probable frame-by-frame path that produces exactly that sequence in
   order. Returns which token is "active" at each frame, plus a confidence
   score per frame.
5. **`torchaudio.functional.merge_tokens`:** because CTC naturally repeats
   predictions across several consecutive frames for the same character (a
   spoken letter usually spans more than one 20ms frame), this collapses
   consecutive same-token frames into a single span per character occurrence,
   each with its own start/end frame and a confidence score.
6. **`_spans_to_words`** (next): takes those character-level spans and
   regroups them into word-level spans.

### `_spans_to_words(processor, token_spans, text, segment_start, sample_rate, num_frames, num_samples) -> list[dict]`

- **Code:** Walks the flat list of character-level `token_spans` in order.
  Blank/pad tokens are skipped (they're CTC's "no new character here"
  marker, not real content). Each real character span is accumulated into
  `current`; hitting the `|` (space) token flushes `current` as one
  completed word and starts a new one. Finally, each word's overall `start`
  is its first character's start frame, and `end` is its last character's
  end frame — converted from frame indices into seconds via `ratio`
  (computed as `num_samples / num_frames / sample_rate`, i.e. how many
  seconds one frame actually spans for this specific audio slice), then
  offset by `segment_start` to convert from "seconds into this segment" back
  to "seconds into the whole track."
- **Concept:** This is the step that turns "the model knows which frames
  correspond to which letters" into "the model knows which frames correspond
  to which *words*" — the actual deliverable for karaoke-style highlighting,
  where you want to highlight whole words in sync with audio, not individual
  letters.

### `align_segments(audio_path, segments, model_id) -> list[dict]`

- **Code:** The public entry point for this stage. Loads the alignment
  model once and the waveform once (not per-segment — reused across the
  loop), then calls `_align_single_segment` for every Whisper segment,
  flattening all their word-lists into one list spanning the whole track.
- **Concept:** This is the function the rest of the pipeline (eventually
  `app/demo.py`) calls — it hides all the CTC/frame/token mechanics behind
  a simple contract: give it audio + Whisper's segments, get back a flat
  list of `{"word", "start", "end"}` covering the entire track, ready to
  drive word-by-word highlighting during playback.

---

## Not yet implemented: Stage 5, `pipeline/export.py`

Will take the word-level list from `align_segments` (and/or the segment
list from `transcribe`) and write out `.srt`, `.vtt`, and a word-timing
JSON file for the Gradio demo's karaoke UI to consume.
