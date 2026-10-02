# Part 5 — Ablations

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

**Depends on**: Part 3's base unsupervised and supervised runs (for the
baseline configs to diff against) and Part 4 Phase A (eval harness).

**No local GPU**: same constraint as Part 3. Write the two ablation configs
and confirm `src/train_simcse.py` accepts them locally (config validation
only, no real training), then package and run these two on Kaggle following
"Running training on Kaggle" in `00_INDEX.md` — same dataset/notebook setup
as Part 3, just swap the config file. You cannot execute the runs yourself;
stop after packaging and resume the "Analysis" section once the checkpoints
and `run_record.json` files are brought back locally.

## Task
Run exactly one ablation per mode. In each, change **one thing only** versus
the corresponding Part 3 baseline config — everything else (seed, batch size,
LR, epochs, etc.) stays identical so the delta is attributable to that one
change.

### Ablation 1 — Unsupervised: shared dropout mask
- Copy `configs/unsupervised.json` → `configs/ablation_unsup_same_mask.json`,
  changing only the setting that makes both "views" of a sentence reuse the
  **same** dropout mask (i.e. the two forward passes become identical,
  removing the only source of augmentation in unsupervised SimCSE).
- Run with `src/train_simcse.py --mode unsup --config configs/ablation_unsup_same_mask.json`.
- Run id: `ablation_unsup_same_mask`.
- Expectation to verify, not assume: this should collapse alignment (positives
  become trivially identical) and hurt the dev/test Spearman relative to the
  baseline — confirm or refute this with your actual numbers.

### Ablation 2 — Supervised: hard negatives on vs. off
- Copy `configs/supervised.json` → `configs/ablation_sup_no_hardneg.json`,
  changing only the flag that disables using `contradiction` as a hard
  negative (train with (premise, entailment) positive pairs and **only**
  in-batch entailment negatives, i.e. Eq. 1-style loss on the supervised
  pairs, no Eq. 5 hard-negative term).
- Run with `src/train_simcse.py --mode sup --config configs/ablation_sup_no_hardneg.json`.
- Run id: `ablation_sup_no_hardneg`.
- The Part 3 supervised baseline run (`sup_*`, hard negatives on) is the other
  half of this comparison — don't retrain it.

## Analysis
For each ablation pair (baseline vs. ablated), using Part 4's `evaluate_sts`
(dev **and**, once available, the single final test pass from Part 6 — do not
score test here if Part 6 hasn't run yet):
- Report both configs side by side (what changed, one line).
- Report dev Spearman delta (and test delta once it exists).
- Give a rough estimate of run-to-run noise (e.g. rerun the baseline with a
  different seed if time allows, or reason from typical variance for this
  dataset size/batch size) and state whether the observed delta exceeds it.
- Explain *why* the change produced the effect you measured, in terms of the
  loss/embedding geometry (e.g. removing augmentation removes the alignment
  signal; removing hard negatives makes the model less able to separate
  semantically-close-but-different sentences).

Write this to `reports/ablations.md`.

## Definition of done
- `configs/ablation_unsup_same_mask.json` and
  `configs/ablation_sup_no_hardneg.json` exist and differ from their baseline
  by exactly one field.
- Both ablation runs are logged in `runs/run_log.jsonl` with checkpoints
  downloaded from Kaggle and present locally under `runs/<run_id>/`.
- `reports/ablations.md` contains both comparisons with dev deltas, a noise
  estimate, and a mechanistic explanation for each.
