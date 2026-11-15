"""Re-run baseline-vs-finetuned eval against an already-trained checkpoint
and append the result to notebooks/logs/run_log.json.

Use this when training finished and saved a checkpoint, but the eval step
or the SSH session dropped before the comparison/run-log step completed
(e.g. whisper-full-v7_20261113-0031: training completed to Drive, but the
`colab ssh` connection was closed by the remote host partway through eval).

Run the same way as scripts/train_model.py, e.g.:
    AUTOSUB_DRIVE_ROOT=/content/drive/MyDrive colab exec -s autosub-trainer -f scripts/eval_checkpoint.py
"""
from pathlib import Path

from finetune.evaluate import compare_baseline_vs_finetuned
from finetune.train import append_run_log

MANIFEST_PATH = Path("data/finetune_manifest_v2.json")
RUN_KEY = "whisper-full-v7_20261113-0031"
CHECKPOINT_DIR = Path("/content/drive/MyDrive/autosub_runs/checkpoints/whisper-full-v7_20261113-0031/checkpoint-450")
# The top-level checkpoint dir is missing model.safetensors -- train.py's
# final model.save_pretrained() call (finetune/train.py:448) apparently
# didn't finish syncing to the Drive FUSE mount before the SSH session that
# launched training was torn down (same connection drop seen during eval).
# checkpoint-450 is the Trainer's own periodic checkpoint at the final step
# (450/450) and has a complete model.safetensors, so it's equivalent to what
# the final save would have produced.
NUM_TRAIN_SONGS = 102
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
FINAL_TRAIN_LOSS = 0.08001068979501724
FINAL_EVAL_LOSS = 1.3220767974853516

eval_result = compare_baseline_vs_finetuned(
    manifest_path=MANIFEST_PATH,
    checkpoint_dir=CHECKPOINT_DIR,
    run_name=RUN_KEY,
)
print(eval_result)

append_run_log({
    "run": RUN_KEY,
    "method": "Full fine-tune",
    "compute": "GPU (fp16)",
    "songs": NUM_TRAIN_SONGS,
    "epochs": NUM_EPOCHS,
    "lr": LEARNING_RATE,
    "train_loss": FINAL_TRAIN_LOSS,
    "eval_loss": FINAL_EVAL_LOSS,
    "baseline_wer": eval_result["baseline"]["wer"],
    "finetuned_wer": eval_result["finetuned"]["wer"],
    "notes": (
        "First run on the expanded 119-song deduped dataset (up from 34 "
        "cleaned songs for v5/LoRA-ablation/v6-short) -- more training "
        "diversity is the lever being tested here, not epochs or method. "
        "Training crashed once before this (Taylor Swift 'Babylon' produced "
        "a 509-token label chunk against Whisper's 448-token cap); fixed by "
        "dropping over-cap chunks in _build_hf_dataset. This run completed "
        "training cleanly but the SSH session dropped during/after eval, so "
        "this eval was re-run standalone against the saved checkpoint."
    ),
})
