# Part 6 — Benchmark table and the gap discussion

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

**Depends on**: Part 3 (both trained models), Part 4 Phase A (reference runs
for raw BERT / SBERT-2019 already logged), Part 4 Phase B (alignment/
uniformity for your models), Part 5 (ablation results, referenced but not
required to build the table itself).

This is the **only place** the STS-B test split gets scored for your two
models, raw BERT, and SBERT-2019 (if not already done in Part 4 Phase A) —
score it once here and reuse the number everywhere else (Part 1, Part 5, Part
7 all reference this table instead of re-running test evaluation).

## Task 1 — Build the table
Write `src/benchmark.py` that reads `runs/run_log.jsonl` and
`reports/eval_*.json` (Part 4 Phase B) and produces `reports/benchmark_table.md`
with one row per:

| Model | Dev Spearman | Test Spearman | Alignment | Uniformity |
|---|---|---|---|---|
| Your unsupervised SimCSE | | | | |
| Your supervised SimCSE | | | | |
| raw bert-base-uncased (mean pooling) | 59.31 | 47.29 | | |
| SBERT-2019 (bert-base-nli-mean-tokens) | 80.77 | 76.98 | | |
| SimCSE paper (unsup, reported) | 82.5 | 76.85 | — | — |
| SimCSE paper (sup, reported) | 86.2 | 84.25 | — | — |

Use `evaluate_sts` (Part 4) for the reproduced rows; use the reference values
from `00_INDEX.md` for raw BERT / SBERT-2019 dev and test; the paper rows
above are verified against the paper itself (unsup: Table 1, dev=82.5 /
Table 5, test=76.85; sup: Table 7, dev=86.2 / Table 5, test=84.25) — double
check them yourself against the PDF before publishing the final table, but
they are not placeholders. Compare **dev with dev and test with test** —
never mix them in one comparison (this is exactly why the two numbers per
paper-row matter: comparing your dev Spearman against the paper's 76.85
would understate your gap, because 76.85 is the paper's *test* number).

## Task 2 — The gap paragraph
Write `reports/gap_analysis.md` accounting for the distance between your
numbers and the paper's reported numbers. At minimum, identify and discuss
these known differences between your setup and the paper's, and put a rough
number on each one's likely contribution where you can:
- **Training data size/source**: for unsupervised, the paper trains on 10⁶
  sentences sampled from English Wikipedia (§3/Appendix A) — unrelated to
  NLI entirely — while you train on 165,529 sentences pooled from the SNLI
  subset; both the source and the size differ. For supervised, the paper
  trains on 314k SNLI+MNLI entailment pairs (Table 4/Appendix A), all with a
  contradiction hard negative available; you train on 33,351 SNLI-only pairs
  with hard negatives for only ~28% of them — smaller, single-dataset, and
  with partial hard-negative coverage.
- **Compute/training budget**: fewer steps/epochs, smaller batch size, or
  fewer seeds than the paper's tuned setup, if applicable to your actual
  config — compare your `configs/*.json` values to the paper's reported
  hyperparameters.
- **Hyperparameter search**: the paper tuned pooling/LR/τ/batch size on a dev
  set across many configurations; you tried far fewer, if any, combinations —
  state how many you actually tried.
- **Any other deliberate deviation** you made (document from
  `configs/*.json` and run `notes` fields, including anything that broke
  during Part 3 training).

State explicitly how much of the total gap each factor plausibly explains
(even a rough percentage split is fine, e.g. "data ~X points, compute ~Y
points, unexplained ~Z points") rather than leaving it purely qualitative —
the assignment asks for numbers, not just prose.

## Definition of done
- `reports/benchmark_table.md` has all six rows filled in, dev vs. dev and
  test vs. test, with alignment/uniformity for the four evaluable rows.
- Test-set Spearman for your two models (and raw BERT/SBERT-2019 if not
  already logged) is scored exactly once and recorded in
  `runs/run_log.jsonl` (update the relevant run's `results.test_spearman`).
- `reports/gap_analysis.md` names the concrete data/setup differences and
  gives a numeric attribution of the gap, not just a qualitative discussion.
