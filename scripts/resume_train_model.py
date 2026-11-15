"""Resume a full fine-tuning run that got cut off mid-training (SSH
session dropped before `nohup`-backgrounding was added to colab_run.sh --
see that script's note), picking up from the Trainer's last epoch
checkpoint instead of restarting from the base model.

Use this once, pointed at the specific dead run; scripts/train_model.py
is still the normal entry point for a fresh run.

Run the same way as scripts/train_model.py:
    AUTOSUB_DRIVE_ROOT=/content/drive/MyDrive colab exec -s autosub-trainer -f scripts/resume_train_model.py
"""
from pathlib import Path

from finetune.evaluate import compare_baseline_vs_finetuned
from finetune.train import append_run_log, train

MANIFEST_PATH = Path("data/finetune_manifest_v3.json")
RUN_NAME = "whisper-full-v8"
NUM_EPOCHS = 8  # same total as the original run -- Trainer resumes step/epoch state from the checkpoint and only runs the remaining epochs (5-8)
FULL_FINETUNE = True
LEARNING_RATE = 1e-4

# The dead run's own output_dir -- pinned explicitly (not left to train()'s
# default make_run_key-based naming) so this resumes into the SAME run
# directory instead of creating a new timestamped one next to it.
RUN_OUTPUT_DIR = Path("/content/drive/MyDrive/autosub_runs/checkpoints/whisper-full-v8_20261113-1812")
RESUME_CHECKPOINT = RUN_OUTPUT_DIR / "checkpoint-668"  # last completed epoch (4/8) before the SSH connection dropped

result = train(
    manifest_path=MANIFEST_PATH,
    output_dir=RUN_OUTPUT_DIR,
    num_epochs=NUM_EPOCHS,
    full_finetune=FULL_FINETUNE,
    learning_rate=LEARNING_RATE,
    run_name=RUN_NAME,
    resume_from_checkpoint=str(RESUME_CHECKPOINT),
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
        "Scales both levers past v7: much larger dataset (v3 manifest, "
        "248 songs pulled from 100 weeks of Billboard history vs. v7's "
        "102-song/20-week set) and more epochs (8 vs. v7's 5), testing "
        "whether a bigger/more diverse training set tolerates more "
        "training before memorizing. This run's SSH session dropped at "
        "step 668/1336 (end of epoch 4/8) because colab_run.sh launched "
        "training in the foreground at the time; resumed from the "
        "epoch-4 checkpoint for epochs 5-8 rather than restarting. "
        "colab_run.sh now launches training detached (nohup) so this "
        "shouldn't recur."
    ),
})
