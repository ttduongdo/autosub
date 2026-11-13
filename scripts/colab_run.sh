#!/usr/bin/env bash
# One-shot setup + run for a Colab CLI GPU session: provisions a fresh
# session, uploads the code + manifest this run needs, mounts Drive for
# the (too-large-to-upload-individually) audio, then runs
# scripts/train_model.py.
#
# Colab's free-tier runtimes get recycled (idle disconnect, time limit)
# with no warning -- when that happens, `colab ssh`/`colab upload` start
# failing with "this runtime does not expose the /colab/ssh endpoint"
# and `colab ls` shows no active sessions. There's no way to prevent
# this on the free tier, so this script exists to make recovering from
# it a single command instead of retyping ~10 commands by hand.
#
# Usage:
#   scripts/colab_run.sh [session-name] [gpu-type]
# Defaults: session-name=autosub-trainer, gpu-type=T4
set -euo pipefail

SESSION="${1:-autosub-trainer}"
GPU="${2:-T4}"

echo "== Creating session '$SESSION' (--gpu $GPU) =="
colab new -s "$SESSION" --gpu "$GPU"

echo "== Mounting Drive =="
colab drivemount -s "$SESSION"

echo "== Creating remote directories =="
colab ssh -s "$SESSION" -- mkdir -p /content/finetune /content/scripts /content/data

echo "== Installing training dependencies =="
# torch/transformers are usually preinstalled on Colab GPU runtimes, but
# datasets/jiwer/peft/torchaudio are not -- install explicitly rather than
# assume. Also upgrades torchao: Colab's preinstalled version (0.10.0) is
# older than peft's LoRA path requires (0.16.0+) -- only hit by LoRA
# training (get_peft_model), not full-finetune runs, which is why this
# wasn't caught until the Phase 2 LoRA ablation in the notebook.
colab ssh -s "$SESSION" -- pip install -q datasets jiwer peft torchaudio wandb
colab ssh -s "$SESSION" -- pip install -q -U torchao

echo "== Uploading code + manifest =="
colab upload -s "$SESSION" finetune/train.py /content/finetune/train.py
colab upload -s "$SESSION" finetune/evaluate.py /content/finetune/evaluate.py
colab upload -s "$SESSION" finetune/dataset.py /content/finetune/dataset.py
colab upload -s "$SESSION" finetune/__init__.py /content/finetune/__init__.py
colab upload -s "$SESSION" scripts/train_model.py /content/scripts/train_model.py
colab upload -s "$SESSION" data/finetune_manifest_v2.json /content/data/finetune_manifest_v2.json

echo "== Running training =="
# -m scripts.train_model, not a bare file path -- running it as a path
# only adds /content/scripts to sys.path (the script's own directory),
# so the sibling finetune/ package isn't importable. -m resolves
# scripts.train_model against the cwd instead, which IS /content, so
# finetune/ (also directly under /content) is found.
#
# HF_TOKEN: optional. whisper-base is a public model, so this only
# silences "unauthenticated requests" rate-limit warnings -- not required
# for this script to work. Forwarded from your local shell env if set
# (export HF_TOKEN=... once locally); omitted entirely if unset, so this
# doesn't fail on machines without it.
HF_TOKEN_ENV=""
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ENV="HF_TOKEN=$HF_TOKEN "
fi
colab ssh -s "$SESSION" -- "cd /content && ${HF_TOKEN_ENV}AUTOSUB_DRIVE_ROOT=/content/drive/MyDrive python -m scripts.train_model"
