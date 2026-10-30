"""Per-chunk error breakdown for a fine-tuned checkpoint's eval run.

compare_baseline_vs_finetuned (finetune/evaluate.py) already returns
per-chunk predictions/references, but only the aggregate WER gets looked
at. This script re-uses that same comparison, then for every chunk:

- computes substitution/deletion/insertion counts (jiwer.process_words),
  so a regression shows up as a *direction* (e.g. "mostly insertions")
  rather than just a worse number
- flags repetition-loop artifacts: the same word (or short phrase)
  repeated 3+ times in a row in the prediction, which earlier runs (v3)
  showed as a fine-tuning-induced failure mode distinct from ordinary
  mistranscription
- sorts chunks by how much WORSE the fine-tuned model did vs. baseline,
  so the worst regressions are easy to read first

This is the ASR analog of a confusion matrix: aggregate WER alone can't
tell you whether a small net improvement is broad-but-shallow or masking
a few bad regressions elsewhere, this can.
"""
import argparse
import json
import re
from pathlib import Path

import jiwer

from finetune.evaluate import (
    _WER_NORMALIZATION,
    compare_baseline_vs_finetuned,
)

# 3+ repeats of the same 1-4 word phrase, back to back.
_REPETITION_RE = re.compile(r"\b(\w+(?:\s+\w+){0,3})\b(?:\s+\1\b){2,}", re.IGNORECASE)


def _error_counts(reference: str, prediction: str) -> dict:
    ref_norm = _WER_NORMALIZATION(reference)
    pred_norm = _WER_NORMALIZATION(prediction)
    if not ref_norm.strip():
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    result = jiwer.process_words([ref_norm], [pred_norm])
    return {
        "wer": result.wer,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "hits": result.hits,
    }


def _has_repetition_loop(text: str) -> bool:
    return _REPETITION_RE.search(_WER_NORMALIZATION(text)) is not None


def analyze(manifest_path: Path, checkpoint_dir: Path, top_n: int = 15) -> dict:
    result = compare_baseline_vs_finetuned(manifest_path, checkpoint_dir)
    baseline = result["baseline"]
    finetuned = result["finetuned"]

    rows = []
    for i, reference in enumerate(baseline["references"]):
        baseline_pred = baseline["predictions"][i]
        finetuned_pred = finetuned["predictions"][i]

        baseline_errors = _error_counts(reference, baseline_pred)
        finetuned_errors = _error_counts(reference, finetuned_pred)

        rows.append({
            "index": i,
            "reference": reference,
            "baseline_prediction": baseline_pred,
            "finetuned_prediction": finetuned_pred,
            "baseline_wer": baseline_errors["wer"],
            "finetuned_wer": finetuned_errors["wer"],
            "wer_delta": finetuned_errors["wer"] - baseline_errors["wer"],  # positive = fine-tuning made it worse
            "finetuned_substitutions": finetuned_errors["substitutions"],
            "finetuned_deletions": finetuned_errors["deletions"],
            "finetuned_insertions": finetuned_errors["insertions"],
            "finetuned_repetition_loop": _has_repetition_loop(finetuned_pred),
            "baseline_repetition_loop": _has_repetition_loop(baseline_pred),
        })

    total_sub = sum(r["finetuned_substitutions"] for r in rows)
    total_del = sum(r["finetuned_deletions"] for r in rows)
    total_ins = sum(r["finetuned_insertions"] for r in rows)
    total_errors = total_sub + total_del + total_ins
    repetition_chunks = [r for r in rows if r["finetuned_repetition_loop"] and not r["baseline_repetition_loop"]]
    worst_regressions = sorted(rows, key=lambda r: r["wer_delta"], reverse=True)[:top_n]
    best_improvements = sorted(rows, key=lambda r: r["wer_delta"])[:top_n]

    print(f"Aggregate: baseline WER {baseline['wer']:.1%}, fine-tuned WER {finetuned['wer']:.1%}")
    print(f"\nError breakdown on fine-tuned predictions ({total_errors} total word errors):")
    if total_errors:
        print(f"  substitutions: {total_sub} ({total_sub / total_errors:.0%})")
        print(f"  deletions:     {total_del} ({total_del / total_errors:.0%})")
        print(f"  insertions:    {total_ins} ({total_ins / total_errors:.0%})")
    print(f"\nRepetition-loop artifacts introduced by fine-tuning (not present in baseline): {len(repetition_chunks)} / {len(rows)} chunks")

    print(f"\n--- Top {top_n} chunks where fine-tuning made things WORSE ---")
    for r in worst_regressions:
        if r["wer_delta"] <= 0:
            break
        print(f"\n[{r['index']}] baseline WER {r['baseline_wer']:.0%} -> fine-tuned WER {r['finetuned_wer']:.0%} (Δ{r['wer_delta']:+.0%})")
        print(f"  ref:       {r['reference']}")
        print(f"  baseline:  {r['baseline_prediction']}")
        print(f"  finetuned: {r['finetuned_prediction']}")
        if r["finetuned_repetition_loop"]:
            print("  flag: repetition loop")

    print(f"\n--- Top {top_n} chunks where fine-tuning helped MOST ---")
    for r in best_improvements:
        if r["wer_delta"] >= 0:
            break
        print(f"\n[{r['index']}] baseline WER {r['baseline_wer']:.0%} -> fine-tuned WER {r['finetuned_wer']:.0%} (Δ{r['wer_delta']:+.0%})")
        print(f"  ref:       {r['reference']}")
        print(f"  baseline:  {r['baseline_prediction']}")
        print(f"  finetuned: {r['finetuned_prediction']}")

    return {"rows": rows, "baseline_wer": baseline["wer"], "finetuned_wer": finetuned["wer"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/finetune_manifest.json")
    parser.add_argument("--checkpoint", default="data/checkpoints/whisper-full-v4")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--out", default=None, help="Optional path to dump full per-chunk JSON")
    args = parser.parse_args()

    analysis = analyze(Path(args.manifest), Path(args.checkpoint), top_n=args.top_n)

    if args.out:
        Path(args.out).write_text(json.dumps(analysis, indent=2))
        print(f"\nFull per-chunk data written to {args.out}")
