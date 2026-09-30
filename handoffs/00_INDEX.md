# Orchestration index — SimCSE assignment (U2T02)

This directory is the handoff contract between an orchestrator agent and a set of
worker agents, each of which implements one part of `docs/instructions.pdf`. Every
other file in `handoffs/` is self-contained: a worker should be able to read **one**
part file plus this index and start working without reading the PDF or any prior
conversation.

Team (for the final report / model cards, not for task assignment):
Julio Cesar De Aquino Castellanos, Gustavo Alexander Fuentes Marin, Valeria Nicol
Hernandez Leon, Ricardo Daniel Horta Sanchez, Jose Angel Pech Xool, Lorena Danae
Perez Lopez.

## Real execution order (not the same as the PDF's part numbering)

The PDF numbers parts 1–8 in reading order, but Part 4 contains a hard
prerequisite ("check your evaluation code is correct before your first training
run") that must happen before Part 3's training runs. Follow this DAG:

1. **Part 2** — data prep (produces the two training sets + STS-B dev/test files).
2. **Part 4, Phase A only** — build `src/evaluate.py` and sanity-check it against
   raw `bert-base-uncased` and `SBERT-2019` reference numbers. Do **not** proceed
   to training until this passes.
3. **Part 3** — train unsupervised and supervised models (can run as two parallel
   workers once step 2 passes; both import `src/evaluate.py` from step 2).
4. **Part 5** — ablations (reuses the Part 3 training script with one flag
   changed; can start as soon as Part 3's base runs are done).
5. **Part 4, Phase B** — full evaluation analysis (alignment/uniformity,
   similarity-distribution plot, retrievals) on the checkpoints from step 3.
6. **Part 6** — benchmark table + gap analysis (needs results from 3, 4, 5, plus
   the raw-BERT/SBERT-2019 numbers from step 2).
7. **Part 7** — publish best checkpoint to the Hub.
8. **Part 1** — conceptual write-up. Can be drafted any time, but the "how many
   negatives" sub-answer needs the final batch sizes from Part 3's configs, so
   finalize it last.
9. **Part 8** — optional extensions, only after 1–7 are done.

## Repository layout (every part writes/reads exactly these paths)

```
data/
  snli_train_100k.jsonl          # given, do not modify
  processed/
    unsup_sentences.txt          # Part 2 output, one sentence per line
    sup_pairs.jsonl              # Part 2 output
    MANIFEST.json                # Part 2 output: counts + checksums
  stsb/
    dev.jsonl                    # Part 2 output
    test.jsonl                   # Part 2 output
src/
  data_prep.py                   # Part 2
  evaluate.py                    # Part 4 (shared import for Part 3, 5, 6, 7)
  train_simcse.py                # Part 3 (one script, --mode unsup|sup)
  ablations.py                   # Part 5
  benchmark.py                   # Part 6
  publish.py                     # Part 7
configs/
  unsupervised.json              # Part 3
  supervised.json                # Part 3
  ablation_unsup_same_mask.json  # Part 5
  ablation_sup_no_hardneg.json   # Part 5
runs/
  run_log.jsonl                  # append-only, one JSON object per completed run
  <run_id>/
    config.json
    results.json
    checkpoint/                  # best checkpoint only (not every epoch)
reports/
  part1_objective.md             # Part 1
  benchmark_table.md             # Part 6
  gap_analysis.md                # Part 6
  ablations.md                   # Part 5
  report.md                      # Part 7 assembles the final report from the above
  figures/
model_cards/
  <model_name>/README.md         # Part 7
requirements.txt                 # pinned deps, every part that adds a dependency updates this
```

## Shared conventions every part must follow

- **Base model**: `bert-base-uncased` for both modes. Do not substitute.
- **Pinned dependencies**: whichever part first introduces a new package
  (`transformers`, `datasets`, `sentence-transformers`, `scipy`, etc.) adds it to
  `requirements.txt` with an exact version (`==`), not a range.
- **Seeds**: every run must set and record a seed (Python, NumPy, torch, and
  `transformers` seed). Record it in the run's `config.json`.
- **Run record schema** — one line appended to `runs/run_log.jsonl` per completed
  run:
  ```json
  {
    "run_id": "unsup_seed42_bs64",
    "mode": "unsupervised",
    "timestamp": "2026-09-30T12:00:00Z",
    "config": { "...": "full hyperparameter dict, same as configs/*.json" },
    "seed": 42,
    "hardware": { "device": "...", "num_gpus": 1, "precision": "fp16" },
    "results": { "dev_spearman": 0.0, "test_spearman": null, "best_step": 0 },
    "checkpoint_path": "runs/unsup_seed42_bs64/checkpoint/",
    "notes": "anything that broke and how it was fixed, or empty string"
  }
  ```
  `test_spearman` stays `null` until the test set is scored (test is scored
  **exactly once**, only for the final model chosen for Part 6/7 — never during
  development).
- **Reference values** (STS-B, Spearman ×100) used to sanity-check the eval
  harness and to build the Part 6 table:
  - raw `bert-base-uncased`, mean pooling: **59.31 dev / 47.29 test**
  - `SBERT-2019` (`bert-base-nli-mean-tokens`): **80.77 / 76.98**
- **Papers**:
  - SimCSE — Gao, Yao & Chen, EMNLP 2021, https://arxiv.org/abs/2104.08821
    (Eq. 1 = unsupervised loss, §3; Eq. 5 = supervised loss w/ hard negatives, §4)
  - Sentence-BERT — Reimers & Gurevych, 2019, https://arxiv.org/abs/1908.10084
  - Alignment & Uniformity — Wang & Isola, ICML 2020,
    https://arxiv.org/abs/2005.10242
- **STS-B source**: https://huggingface.co/datasets/sentence-transformers/stsb
  (dev split = 1,500 pairs, used repeatedly during development; test split =
  1,379 pairs, used exactly once for the final number).
- **Do not touch** `data/snli_train_100k.jsonl` or `docs/instructions.pdf`.

## Definition of done (whole project)

Matches the PDF's "minimum complete delivery": both models trained, own dev
**and** test Spearman numbers, the Part 6 benchmark table, a report covering
Parts 1/5/6, and at least one model on the Hub verified by reloading it.
Everything in Part 8 is extra credit, not required.
