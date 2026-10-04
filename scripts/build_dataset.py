from pathlib import Path
from finetune.dataset import build_manifest

output_path = Path("data/finetune_manifest.json")
result = build_manifest(num_songs=100, output_path=output_path)
print("Manifest written to:", result)
