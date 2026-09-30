# Part 7 — Publish to the Hugging Face Hub

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

**Depends on**: Part 6 (final benchmark table — you need the test Spearman
number to verify against after reloading).

## Task 1 — Pick and export the best model
Choose the better of your two SimCSE checkpoints (unsupervised vs. supervised)
by dev Spearman from `reports/benchmark_table.md` — if the assignment or team
wants both published, repeat this whole part per model; publishing at least
one is the required minimum.

Write `src/publish.py` that:
- Loads the chosen checkpoint from `runs/<run_id>/checkpoint/`.
- Wraps it as a `sentence-transformers` `SentenceTransformer` model: a
  `Transformer` module (the trained BERT weights) + a `Pooling` module
  matching the pooling strategy actually used in that run's
  `config.json` (cls/mean) + normalization if cosine similarity was used
  without it built into the pooling step.
- Saves locally to `model_cards/<model_name>/` and pushes to the Hub with
  `model.push_to_hub("<hf-username-or-org>/<model_name>")`.

Choose `<model_name>` descriptively, e.g. `simcse-bert-base-snli-unsup` or
`simcse-bert-base-snli-sup`.

## Task 2 — Verify the published model
This is mandatory, not optional:
- Reload with `SentenceTransformer("<hf-username-or-org>/<model_name>")` (from
  the Hub, not the local cache — confirm this, e.g. by clearing/renaming the
  local cache dir or running in a fresh environment).
- Re-run `evaluate_sts` (Part 4) on `data/stsb/test.jsonl` against the
  reloaded model.
- Compare the resulting test Spearman to the number already recorded for this
  run in `reports/benchmark_table.md` / `runs/run_log.jsonl`. It must match
  (small floating-point differences are fine; a meaningfully different number
  means the export/reload pipeline is broken — fix it, don't just report the
  mismatch).

## Task 3 — Model card
Write `model_cards/<model_name>/README.md` (this becomes the Hub model card)
covering:
- Training data: SNLI subset used (link back to
  `data/processed/MANIFEST.json` counts), not the paper's original data —
  state this explicitly.
- Recipe: mode (unsupervised/supervised), base model, pooling, all
  hyperparameters from the run's `config.json`.
- Metrics: dev and test Spearman from `reports/benchmark_table.md`, plus
  alignment/uniformity.
- Limitations: the gap-to-paper discussion from `reports/gap_analysis.md`,
  summarized in a few sentences, and any known failure modes from the Part 4
  Phase B retrieval failure case.

## Task 4 — Assemble the final report
The PDF's deliverables require **one** report, not scattered files. Write
`reports/report.md` that stitches together everything produced by earlier
parts, in this order:
1. Team member names (required — put this at the top):
   Julio Cesar De Aquino Castellanos, Gustavo Alexander Fuentes Marin,
   Valeria Nicol Hernandez Leon, Ricardo Daniel Horta Sanchez, Jose Angel
   Pech Xool, Lorena Danae Perez Lopez.
2. Part 1 answers — include `reports/part1_objective.md` verbatim or by
   reference.
3. Configuration table — one row per run (unsup, sup, both ablations) listing
   pooling/LR/batch size/τ/epochs/dropout/seed, pulled from `configs/*.json`.
4. Benchmark table + analysis — `reports/benchmark_table.md`, the
   alignment/uniformity plot and similarity-distribution figures from
   `reports/figures/`, and the retrievals from Part 4 Phase B.
5. The two ablations — `reports/ablations.md`.
6. The gap discussion — `reports/gap_analysis.md`.
7. (If applicable) the Part 8 optional-extensions section.
8. Links to the published Hub model(s) and their model cards.

## Definition of done
- Model is visible on the Hub at the chosen repo id.
- A fresh reload from the Hub reproduces the recorded test Spearman.
- `model_cards/<model_name>/README.md` covers data, recipe, metrics, and
  limitations, and matches what actually got pushed.
- `runs/run_log.jsonl`'s entry for this run is updated with the Hub repo id
  (add a `"hub_repo_id"` field).
- `reports/report.md` exists, opens with the team member names, and includes
  every section listed in Task 4 above.
