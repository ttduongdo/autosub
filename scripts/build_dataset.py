from pathlib import Path
from finetune.dataset import build_manifest

# Writes to a new file, not data/finetune_manifest.json directly -- that
# file holds the current cleaned 44-song dataset (post data-fixup-1) that
# v5 and the LoRA ablation's results are tied to. Review/merge manually
# once this finishes instead of overwriting it outright.
output_path = Path("data/finetune_manifest_v2.json")
# A single week's Hot 100 is always exactly 100 songs, and only ~58% of
# raw candidates survive lyrics-lookup + audio-duration-validation
# (confirmed empirically: 91 candidates -> 53 usable). billboard_weeks=20
# pulls ~20 weeks of charts deduped by (artist, title) -- comfortably
# more than enough distinct candidates to reach num_songs=200 even with
# heavy week-to-week chart overlap.
result = build_manifest(num_songs=200, output_path=output_path, billboard_weeks=20)
print("Manifest written to:", result)
