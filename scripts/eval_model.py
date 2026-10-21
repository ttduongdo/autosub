from pathlib import Path
from finetune.evaluate import compare_baseline_vs_finetuned

result = compare_baseline_vs_finetuned(
    manifest_path=Path("data/finetune_manifest.json"),
    checkpoint_dir=Path("data/checkpoints/whisper-full-v4"),
)
