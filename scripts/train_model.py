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

MANIFEST_PATH = Path("data/finetune_manifest_v2.json")  # 120-song set; see data/finetune_manifest.json for the earlier 44-song set
RUN_NAME = "whisper-full-v7"
NUM_EPOCHS = 5
FULL_FINETUNE = True
LEARNING_RATE = 1e-4

result = train(
    manifest_path=MANIFEST_PATH,
    num_epochs=NUM_EPOCHS,
    full_finetune=FULL_FINETUNE,
    learning_rate=LEARNING_RATE,
    run_name=RUN_NAME,
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
        "First run on the expanded 120-song dataset (up from 34 cleaned "
        "songs for v5/LoRA-ablation/v6-short) -- more training diversity "
        "is the lever being tested here, not epochs or method."
    ),
})
