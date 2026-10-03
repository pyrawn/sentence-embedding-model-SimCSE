#!/usr/bin/env bash
# Assemble the flat Kaggle Dataset folder for the Part 5 supervised ablation (no hard negatives).
# Upload the resulting folder (or the .zip) as Kaggle Dataset "simcse-ablation-sup-inputs".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/kaggle/dist/simcse-ablation-sup-inputs"
rm -rf "$OUT" && mkdir -p "$OUT"
cp "$ROOT/src/train_simcse.py" "$ROOT/src/evaluate.py" \
   "$ROOT/configs/ablation_sup_no_hardneg.json" \
   "$ROOT/data/processed/sup_pairs.jsonl" \
   "$ROOT/requirements.txt" "$OUT/"
cp "$ROOT/data/stsb/dev.jsonl" "$OUT/stsb_dev.jsonl"   # dev only; test stays local
(cd "$OUT" && sha256sum * > SHA256SUMS)
(cd "$ROOT/kaggle/dist" && rm -f simcse-ablation-sup-inputs.zip && zip -qj simcse-ablation-sup-inputs.zip simcse-ablation-sup-inputs/*)
ls -la "$OUT"
