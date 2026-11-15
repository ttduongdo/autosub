"""CLI entry point for a full fine-tuning run: train, evaluate against
baseline, and log the result to notebooks/logs/run_log.json.

Run via the Colab CLI against a provisioned GPU runtime, e.g.:
    colab exec -s autosub-trainer -f scripts/train_model.py
or
    colab run --gpu T4 scripts/train_model.py

On a Colab session (not a real repo checkout), this process won't have
data/processed/'s audio on disk -- set AUTOSUB_DRIVE_ROOT to resolve
audio, checkpoints, and logs against Drive instead (see finetune/train.py's
DRIVE_ROOT docstring), e.g.:
    AUTOSUB_DRIVE_ROOT=/content/drive/MyDrive colab exec -s autosub-trainer -f scripts/train_model.py
Still need the manifest + finetune/ package itself on the session's
filesystem (colab upload), since those are small, git-trackable files --
only the (large, binary) audio and run outputs go through Drive.

Edit the RUN_NAME/config block below to match the run you're about to
make, then run this file -- the same config lives in git history instead
of being buried in notebook cell outputs.
"""
from pathlib import Path

from finetune.evaluate import compare_baseline_vs_finetuned
from finetune.train import append_run_log, train

MANIFEST_PATH = Path("data/finetune_manifest_v3.json")  # 248-song set, 100 weeks of Billboard history; see data/finetune_manifest_v2.json for v7's 102-song set
RUN_NAME = "whisper-full-v9"
NUM_EPOCHS = 5  # back to v7's epoch count -- v8 showed 8 epochs overfits even on more data, so isolate chunk_duration as the only new variable here
FULL_FINETUNE = True
LEARNING_RATE = 1e-4
# v7/v8 both used the 30s default (multi-line chunks). The reference paper
# (arXiv:2609.11302) trains on much shorter spans -- mean 4s, close to one
# lyric line -- not 30s blocks of several concatenated lines. Our own
# lyrics_segments are already line-level (confirmed: 1-3s each), so this
# tests whether matching the paper's granularity helps, independent of
# data size or epoch count. See EXPERIMENTS.md's chunk-granularity entry.
CHUNK_DURATION = 5.0

result = train(
    manifest_path=MANIFEST_PATH,
    num_epochs=NUM_EPOCHS,
    full_finetune=FULL_FINETUNE,
    learning_rate=LEARNING_RATE,
    run_name=RUN_NAME,
    chunk_duration=CHUNK_DURATION,
    # W&B off by default here: each Colab CLI session is a fresh, disposable
    # VM (see scripts/colab_run.sh's recycling note), so `wandb login`
    # wouldn't persist between runs anyway without extra auth plumbing.
    # make_run_key's live train JSONL + run_log.json (both on Drive when
    # AUTOSUB_DRIVE_ROOT is set) already cover per-step/per-run tracking.
    report_to_wandb=False,
)
print(result)
print("run_key:", result["run_key"])

eval_result = compare_baseline_vs_finetuned(
    manifest_path=MANIFEST_PATH,
    checkpoint_dir=result["output_dir"],
    run_name=result["run_key"],
    chunk_duration=CHUNK_DURATION,  # must match training's chunk_duration for a valid comparison
)

append_run_log({
    "run": result["run_key"],
    "method": "Full fine-tune" if FULL_FINETUNE else "LoRA",
    "compute": "GPU (fp16)",
    "songs": result["num_train_songs"],
    "epochs": NUM_EPOCHS,
    "lr": LEARNING_RATE,
    "train_loss": result["final_train_loss"],
    "eval_loss": result["final_eval_loss"],
    "baseline_wer": eval_result["baseline"]["wer"],
    "finetuned_wer": eval_result["finetuned"]["wer"],
    "notes": (
        "Tests chunk granularity in isolation: same v3 manifest, epochs, "
        "and LR as v7 (which got 52.4% WER), but chunk_duration=5s instead "
        "of the 30s default -- v7/v8 trained on multi-line ~30s chunks; "
        "the reference paper (arXiv:2609.11302) trains on ~4s spans close "
        "to single lyric lines, which our own lyrics_segments already "
        "support natively (they're line-level, 1-3s each, pre-chunking). "
        "v8 already showed more epochs/data alone isn't the answer "
        "(54.6% WER, severe overfitting) -- this checks whether matching "
        "the paper's training-example granularity, not just its data "
        "scale, is what's actually driving their lower WER."
    ),
})
