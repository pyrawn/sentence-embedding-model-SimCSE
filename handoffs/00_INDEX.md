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
   workers once step 2 passes; both import `src/evaluate.py` from step 2). **No
   local GPU is available — these runs execute on Kaggle Notebooks**, see
   "Running training on Kaggle" below.
4. **Part 5** — ablations (reuses the Part 3 training script with one flag
   changed; can start as soon as Part 3's base runs are done). **Also runs on
   Kaggle**, same workflow as Part 3.
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
notebooks/
  <run_id>.ipynb                 # Part 3/5: the executed Kaggle notebook (with
                                  # cell outputs/logs) for each GPU run, kept as
                                  # evidence for the report — versioned in git
kaggle/
  <mode>.ipynb                   # Part 3/5: source notebook uploaded to Kaggle
  build_*_dataset.sh             # Part 3/5: scripts that assemble Kaggle Dataset folders
  dist/                          # Part 3/5: built Kaggle Dataset zips (gitignored)
  downloads/                     # Part 3/5: raw Kaggle output downloads (gitignored,
                                  # staging only — ingest into runs/ then discard)
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

## Running training on Kaggle (no local GPU)

The development machine has no GPU. **Only Part 3 and Part 5 need one**
(model training) — Parts 2, 4, 6, 7 are data processing / CPU-feasible
inference (embedding a few thousand STS-B sentences on CPU is a couple of
minutes, not hours) and run locally as normal. Do not move those to Kaggle;
it just adds sync overhead for no benefit.

For Part 3 / Part 5, the worker agent cannot execute inside Kaggle directly
(no shell access to Kaggle's remote runtime) — it prepares everything needed,
the human runs it on Kaggle, then the agent (or the human) imports the
results back. Workflow:

1. **Package for Kaggle.** Create/update a Kaggle Dataset (e.g.
   `simcse-snli-inputs`) containing: the relevant file from
   `data/processed/` (`unsup_sentences.txt` or `sup_pairs.jsonl`), the config
   JSON being run, `src/train_simcse.py` (and anything it imports, e.g.
   `src/evaluate.py`), and `requirements.txt`. Re-upload a new dataset
   version whenever the code or config changes.
2. **Create the Kaggle Notebook.** Attach that dataset as input. Settings:
   Accelerator = GPU (T4 x2 or P100), Internet = On (needed for `pip install`
   and to pull `bert-base-uncased` from the Hub). Kaggle requires phone
   verification on the account to enable GPU + Internet together — flag this
   to the user if it hasn't been done yet, don't assume it's set up.
3. **Notebook cells**, in order:
   - `!pip install -q -r /kaggle/input/<dataset-slug>/requirements.txt`
   - Copy the input files into the same relative layout the script expects
     (e.g. `data/processed/unsup_sentences.txt`) under `/kaggle/working/`, so
     paths in `src/train_simcse.py` resolve unmodified.
   - Run training, writing outputs under `/kaggle/working/runs/<run_id>/`:
     `!python train_simcse.py --mode unsup --config unsupervised.json --output_dir runs/<run_id>`
   - Final cell: write `runs/<run_id>/run_record.json` — **one** JSON object
     matching the run-record schema below (not the whole `run_log.jsonl`;
     that only exists locally), with `hardware` set to the actual Kaggle
     accelerator (e.g. `"Kaggle T4 x2"`).
   - "Save Version" → "Save & Run All", wait for it to finish.
4. **Bring results back.** Download the notebook's output (Output tab, or
   `kaggle kernels output <user>/<slug> -p ./kaggle_output` if the Kaggle CLI
   + API token are set up locally). Copy
   `runs/<run_id>/checkpoint/` and `runs/<run_id>/config.json` into the local
   repo at the identical path, then **append** the downloaded
   `run_record.json`'s content as one line to `runs/run_log.jsonl` (never
   overwrite that file — always append). Proceed with the rest of the part's
   "Definition of done" locally as written.
5. **Keep the executed notebook as evidence.** Also download the notebook
   itself (with its cell outputs/logs — download it from the already-run
   version, not a fresh editor draft) and save it as `notebooks/<run_id>.ipynb`.
   This is versioned in git (unlike `kaggle/downloads/`) — it's the log
   backing any training-instability discussion in the report.

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
