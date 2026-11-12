# AutoSub

A karaoke-style subtitle generator for music, with a Whisper fine-tuning track for sung vocals.

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.11-blue)

End-to-end audio pipeline + custom web app + a fine-tuning experiment.

The fine-tuning side was inspired by a 2026 MSc thesis on Greek lyrics transcription ([Frangiadaki, Athena Research Center](https://arxiv.org/abs/2609.11302)), which took Whisper from 92% to 30% WER on sung lyrics through full fine-tuning and more training epochs. This project runs the same experiment at a smaller scale.

## What it does

Give it a song and it separates the vocals from the instrumental (Demucs), transcribes the vocals (Whisper), and aligns the transcript to the audio at the word level using a wav2vec2 CTC model, which is what makes karaoke-style highlighting possible. From there it exports `.srt`/`.vtt` subtitles plus the word-timing data the UI uses.

It runs as a local web app with a zoomable waveform, a synced lyrics strip, and a toggle between the full mix, vocals, and instrumental.

## Why

Whisper is trained on speech, not singing, and pitch, sustain, and backing instrumentation break a lot of its assumptions. This project attacks that from two angles: cleaning up the signal by isolating vocals before transcription, and adapting the model itself by fine-tuning Whisper on sung lyrics.

## How it's built

**Pipeline** (`pipeline/`): ingest → Demucs → Whisper → forced alignment → export.

**Fine-tuning** (`finetune/`): 
- a dataset builder that pulls song picks from the Billboard Hot 100 and gets real timestamps from an independent lyrics database, 
- a training script that supports either LoRA or full fine-tuning,
- an eval script that compares baseline vs. fine-tuned WER on a held-out set split by artist.

## Progress

The pipeline and demo work end to end. The table below is every fine-tuning run so far, each measured as baseline vs. fine-tuned WER on a held-out set of artists the model never trained on.

| Run | Method | Songs | Epochs | WER (baseline → fine-tuned) |
|---|---|---|---|---|
| v1 | LoRA | 9 | 3 | 69.2% → 88.4% |
| v2 | LoRA | 50 | 3 | 88.3% → 94.7% |
| v3 | LoRA | 100 | 3 | 68.7% → 80.4% |
| v4 | Full fine-tune | 81 | 5 | 68.7% → 66.4% |
| v5 | Full fine-tune | 34 (cleaned) | 5 | 66.2% → 55.1% |
| LoRA ablation | LoRA | 34 (cleaned) | 5 | 66.2% → 58.6% |
| v6-short | Full fine-tune | 34 (cleaned) | 2 | 66.2% → 63.1% |

**v4** Switching from LoRA (which only trains ~1.6% of the model's weights) to full fine-tuning, and from 3 epochs to 5, finally crossed into a net win: 66.4% vs. 68.7% baseline.

**v5** is the best result so far — same method/epochs as v4, but on a cleaned 34-song set after dropping tracks where the audio didn't actually match the lyrics' timing (see Data notes below). 55.1% WER, down from a 92%-ish out-of-the-box baseline on sung lyrics generally.

**LoRA ablation** isolated what actually drove v4/v5's win: same data and epochs as v5, LoRA instead of full fine-tuning. Still a real improvement (58.6%), but full fine-tuning has a genuine edge (+11.2pp vs. LoRA's +7.7pp) — it's not just more epochs or cleaner data.

**v6-short** tested whether v5's full fine-tune was overfitting (its train loss collapses to ~0.03 while eval loss stays flat). Cutting to 2 epochs did make that loss gap healthier, but WER got worse (63.1%), not better — so the "overfitting" is partly just the model continuing to learn useful things past the point its training loss stopped being meaningful as a signal. v5 remains the benchmark to beat; more/cleaner training data is the next lever, not fewer epochs.

