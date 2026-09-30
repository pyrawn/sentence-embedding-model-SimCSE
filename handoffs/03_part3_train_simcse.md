# Part 3 — Train both SimCSE modes

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

**Hard prerequisite**: Part 4 Phase A must be complete and passing (sanity
check against raw BERT and SBERT-2019 reference numbers) before you run any
training here. Import `evaluate_sts` from `src/evaluate.py` (Part 4) for
dev-set checkpoint selection — do not reimplement evaluation.

**Input**: `data/processed/unsup_sentences.txt` and
`data/processed/sup_pairs.jsonl` (Part 2), `data/stsb/dev.jsonl` (Part 4).

The two modes below can be built and run independently (e.g. by two different
workers) — they share `src/train_simcse.py` with a `--mode` flag but do not
depend on each other's runs.

## Shared script: `src/train_simcse.py`

Both modes start from `bert-base-uncased` and share: optimizer (AdamW),
in-batch contrastive loss with temperature τ, dropout as the only
augmentation mechanism, checkpointing on best dev Spearman, and the run
logging contract in `00_INDEX.md`. Every hyperparameter (pooling, learning
rate, batch size, τ, epochs, dropout rate, seed) is your choice — record
whatever you pick in the run's `config.json`; nothing here is prescribed
beyond what's stated.

### Mode A — Unsupervised (paper §3, Eq. 1)
- Input: `data/processed/unsup_sentences.txt`.
- Each sentence is encoded **twice** in the same forward batch with
  independent dropout masks (standard PyTorch training-mode dropout is
  sufficient — no explicit augmentation). The two encodings of the same
  sentence are the positive pair; every other sentence's two encodings in the
  batch are negatives.
- Loss: Eq. 1, `-log( exp(sim(h_i,h_i+)/τ) / Σ_j exp(sim(h_i,h_j+)/τ) )` over
  cosine similarity of normalized embeddings.
- Config file: `configs/unsupervised.json`.
- Run id prefix: `unsup_`.

### Mode B — Supervised (paper §4, Eq. 5, with hard negatives)
- Input: `data/processed/sup_pairs.jsonl` (premise, entailment,
  contradiction|null).
- Positive pair: (premise, entailment). Hard negative: (premise,
  contradiction) when present (~28% of rows per Part 2's manifest) — this is
  the default "hard negatives on" configuration (Part 5's ablation turns this
  off).
- Loss: Eq. 5 — denominator sums over other in-batch entailment embeddings
  **and** hard-negative embeddings when present.
- Config file: `configs/supervised.json`.
- Run id prefix: `sup_`.

## During training (both modes)
- Periodically call `evaluate_sts(..., split_path="data/stsb/dev.jsonl")` from
  Part 4's harness; keep the checkpoint with the best dev Spearman, discard
  the rest (don't fill disk with every epoch's weights).
- If loss doesn't move or collapses to (near-)zero, that's a signal to debug
  (common causes: wrong pooling/normalization before cosine sim, τ too
  low/high, learning rate too high, accidentally identical dropout masks for
  both views). Record what happened and the fix in the run's `notes` field —
  this feeds directly into the report (Part 6/7 reference it).
- On completion, append one record to `runs/run_log.jsonl` per the schema in
  `00_INDEX.md`, with `results.dev_spearman` filled in and
  `results.test_spearman` left `null` (test is scored once, later, in Part 6).
- Save `runs/<run_id>/config.json`, `runs/<run_id>/results.json`, and
  `runs/<run_id>/checkpoint/` (best checkpoint only).

## Definition of done
- `configs/unsupervised.json` and `configs/supervised.json` exist with every
  hyperparameter used.
- One completed run per mode in `runs/run_log.jsonl`, each with a saved best
  checkpoint and a non-null `dev_spearman`.
- Any training instability encountered is documented in the run's `notes`.
