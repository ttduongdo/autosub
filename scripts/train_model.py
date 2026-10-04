from pathlib import Path
from finetune.train import train

result = train(
    manifest_path=Path("data/finetune_manifest.json"),
    output_dir=Path("data/checkpoints/whisper-lora-v3"),
    num_epochs=3,
)
print("Checkpoint saved to:", result)
