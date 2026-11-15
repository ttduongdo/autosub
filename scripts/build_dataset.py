from pathlib import Path
from finetune.dataset import build_manifest

# Writes to a new file, not data/finetune_manifest.json directly -- that
# file holds the current cleaned 44-song dataset (post data-fixup-1) that
# v5 and the LoRA ablation's results are tied to. Review/merge manually
# once this finishes instead of overwriting it outright.
output_path = Path("data/finetune_manifest_v3.json")
# v2's 20-week/200-song-target run landed at 305 deduped candidates ->
# 119 usable (net ~5.95% of raw Billboard rows survive dedup + validation,
# since heavy week-to-week chart overlap collapses most repeat rows before
# validation even runs). Scaling that rate to a 400-song target needs on
# the order of 6700+ raw rows, i.e. 65-70+ weeks -- but dedup overlap gets
# worse (not linear) as more weeks are pulled, so billboard_weeks=100 is
# intentionally well above the naive estimate; _fetch_billboard_charts
# just caps at whatever's actually available; no cost to over-asking
# since this only reads small JSON files from GitHub, not audio.
result = build_manifest(num_songs=400, output_path=output_path, billboard_weeks=100)
print("Manifest written to:", result)
