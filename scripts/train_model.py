from pathlib import Path
from finetune.train import train

result = train(
    manifest_path=Path("data/finetune_manifest.json"),
    output_dir=Path("data/checkpoints/whisper-full-v4"),
    num_epochs=5,
    full_finetune=True,
    run_name="whisper-full-v4",
    report_to_wandb=True,
    # resume_from_checkpoint="data/checkpoints/whisper-full-v4/checkpoint-N",
)
print("Checkpoint saved to:", result)
