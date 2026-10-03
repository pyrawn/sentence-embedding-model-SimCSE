#!/usr/bin/env bash
# Assemble the flat Kaggle Dataset folder for the Part 5 unsupervised ablation (same dropout mask).
# Upload the resulting folder (or the .zip) as Kaggle Dataset "simcse-ablation-unsup-inputs".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/kaggle/dist/simcse-ablation-unsup-inputs"
rm -rf "$OUT" && mkdir -p "$OUT"
cp "$ROOT/src/train_simcse.py" "$ROOT/src/evaluate.py" \
   "$ROOT/configs/ablation_unsup_same_mask.json" \
   "$ROOT/data/processed/unsup_sentences.txt" \
   "$ROOT/requirements.txt" "$OUT/"
cp "$ROOT/data/stsb/dev.jsonl" "$OUT/stsb_dev.jsonl"   # dev only; test stays local
(cd "$OUT" && sha256sum * > SHA256SUMS)
(cd "$ROOT/kaggle/dist" && rm -f simcse-ablation-unsup-inputs.zip && zip -qj simcse-ablation-unsup-inputs.zip simcse-ablation-unsup-inputs/*)
ls -la "$OUT"
