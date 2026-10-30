"""Pull a random sample of (audio clip, LRCLIB text) pairs to manually
spot-check alignment quality.

The manifest's lyrics_segments ARE the LRCLIB data (line text + start/end
timestamps), paired against vocals_path. This script just slices each
sampled segment's audio out to its own small WAV file so you can listen to
it next to the text LRCLIB claims is sung there, without writing any
throwaway slicing code by hand each time.

Uses the same CHUNK_PADDING as finetune/train.py so what you hear here
matches what the model actually trains on.

Usage: python -m scripts.sample_alignment_check --n 15
Then open the printed output dir and listen to each clip against its
printed text.
"""
import argparse
import json
import random
from pathlib import Path

import torchaudio

from finetune.train import CHUNK_PADDING

SAMPLE_RATE = 16000


def main(manifest_path: Path, n: int, seed: int, out_dir: Path) -> None:
    records = json.loads(manifest_path.read_text())
    all_segments = []
    for record in records:
        for seg in record["lyrics_segments"]:
            all_segments.append({
                "song_id": record["song_id"],
                "artist": record["artist"],
                "vocals_path": record["vocals_path"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })

    rng = random.Random(seed)
    sample = rng.sample(all_segments, min(n, len(all_segments)))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = []

    for i, seg in enumerate(sample):
        waveform, sample_rate = torchaudio.load(seg["vocals_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        start_sample = max(int((seg["start"] - CHUNK_PADDING) * SAMPLE_RATE), 0)
        end_sample = min(int((seg["end"] + CHUNK_PADDING) * SAMPLE_RATE), waveform.shape[1])
        clip = waveform[:, start_sample:end_sample]

        clip_path = out_dir / f"{i:02d}_{seg['song_id']}.wav"
        torchaudio.save(str(clip_path), clip, SAMPLE_RATE)

        print(f"[{i:02d}] {seg['artist']} ({seg['song_id']}) {seg['start']:.1f}s-{seg['end']:.1f}s")
        print(f"     text: {seg['text']}")
        print(f"     clip: {clip_path}")

        manifest_out.append({**seg, "clip_path": str(clip_path)})

    (out_dir / "sample_manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"\n{len(sample)} clips written to {out_dir}/ — listen to each against its printed text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/finetune_manifest.json")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/alignment_check")
    args = parser.parse_args()

    main(Path(args.manifest), args.n, args.seed, Path(args.out))
