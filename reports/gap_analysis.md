# Part 6 — Gap analysis: our SimCSE vs. the paper

All numbers are STS-B Spearman ×100, BERT-base. We compare dev with dev and
test with test (see `reports/benchmark_table.md`). Paper numbers: Gao, Yao &
Chen (EMNLP 2021). Unsup dev is from Table 1 and unsup test from Table 5. Sup
dev is from Table 7 and sup test from Table 5. Paper setup details below come
from §3, §4, Table 4, Table 6 and Appendix A (Table A.1), checked against the
arXiv PDF.

## The gap

| Mode | Ours dev | Paper dev | **Δ dev** | Ours test | Paper test | **Δ test** | Our dev→test drop | Paper dev→test drop |
|---|---|---|---|---|---|---|---|---|
| Unsupervised (`unsup_seed42_bs64`) | 76.92 | 82.5 | **−5.58** | 68.33 | 76.85 | **−8.52** | 8.59 | 5.65 |
| Supervised (`sup_seed42_bs128`) | 82.05 | 86.2 | **−4.15** | 78.75 | 84.25 | **−5.50** | 3.30 | 1.95 |

For comparison, the dev→test drop is 12.02 for raw `bert-base-uncased` (59.31
→ 47.29) and 3.79 for SBERT-2019 (80.77 → 76.98). STS-B test is harder than
dev for every model. It is hardest for models that stay close to raw BERT.

The test gap is larger than the dev gap in both modes: by 2.94 points for
unsupervised and 1.35 for supervised. Comparing our dev 76.92 with the paper's
test 76.85 would hide all of this, which is why the two splits are kept
apart.

## Setup differences

| | Paper (BERT-base) | Ours | Source |
|---|---|---|---|
| Unsup training data | 10⁶ sentences sampled from **English Wikipedia** | **165,529** unique SNLI sentences (premises and hypotheses of a 100k-record subset) | §3 fn. 2, §4; `data/processed/MANIFEST.json` |
| Sup training data | SNLI + MNLI, **314k** (premise, entailment, contradiction) triplets | **33,351** SNLI (premise, entailment) pairs from the 100k-record subset (no-consensus records dropped, then sampled). Only **9,488 (28.4%)** have a contradiction | §4, Table 4; MANIFEST |
| Hard negatives | One per triplet (every anchor) | One for 28.4% of anchors (first contradiction for the same premise); 71.6% have none | §4 fn. 7; MANIFEST `hard_negative_rule` |
| Unsup bs / LR / epochs | 64 / 3e-5 / 1 | 64 / 3e-5 / 1 (same) | Table A.1; `configs/unsupervised.json` |
| Sup bs / LR / epochs | **512** / 5e-5 / 3 | **128** / 5e-5 / 3 (LR not re-tuned for the smaller batch) | Table A.1; `configs/supervised.json` |
| Optimizer steps (unsup) | 10⁶ / 64 ≈ 15,600 | 2,586 (best checkpoint at **step 125** = 8,000 sentences seen) | `runs/unsup_seed42_bs64/results.json` |
| Optimizer steps (sup) | 314k / 512 × 3 ≈ 1,840 (942k anchor-views) | 780 (100k anchor-views); best at step 350 | `runs/sup_seed42_bs128/results.json` |
| τ, max length, dropout | 0.05, 32, 0.1 | same | configs |
| Pooling | [CLS] + MLP. Unsup drops the MLP at test time; the paper's final sup model keeps it | Unsup: same. Sup: no MLP at all (`mlp_only_train: false` builds none) | Table 6; `src/train_simcse.py` |
| Grad clipping | HF default (1.0) | unsup 1.0; **sup none** (`max_grad_norm: null`) | configs |
| Dev eval frequency / selection | every 250 steps, keep best | every 125 (unsup) / 50 (sup) steps, keep best | App. A; configs |
| Hyperparameter search | grid of batch ∈ {64,128,256,512} × LR ∈ {1e-5,3e-5,5e-5} = **12 configs per model**, plus ablations of τ, p, pooling and hard-negative α on dev | **1 config per mode, no search.** The unsup config copies Table A.1. The sup config uses the paper's LR with a 4× smaller batch. The two Part 5 runs are ablations (one method flag each), not a search | App. A; `runs/run_log.jsonl` |
| Seeds | best-checkpoint numbers for one setting | 1 seed (42) per config | run log |
| Hardware / precision | — | 1× Kaggle Tesla T4, fp16 AMP; 618 s (unsup), 434 s (sup) | run log `hardware` |
| Things that broke | — | Unsup: "loss < 1e-3, possible collapse" alert. Investigated and found to be a **false positive** (positive cosine ≈ 0.83, not 1; the low loss comes from τ = 0.05). Sup: nothing (`notes` empty) | run log `notes` |

## Attribution of the gap

These are estimates. Every config has one seed, and we did not re-run with
the paper's data. Where the paper has a controlled comparison, we extrapolate
from it; otherwise the share is reasoned from our training curves. The
single-seed noise floor estimated in `reports/ablations.md` is about ±0.5
(sup) and ±1 (unsup). A residual smaller than that cannot be told apart from
zero.

### Supervised: −4.15 dev / −5.50 test

| Factor | Dev (pts) | Test (pts) | How the number was obtained |
|---|---|---|---|
| **Training data size and source** (33k SNLI pairs vs. 314k SNLI+MNLI) | **~2.1** (1.5–3) | **~3.1** | Paper Table 4, entailment-only positives: 134k → 314k pairs (2.3×) gives 84.1 → 84.9, i.e. +0.8, or ~0.65 per doubling. We have 9.4× fewer pairs (3.2 doublings), so ~2.1 dev. On test we add ~1.0 of the extra dev→test drop (3.30 vs. 1.95): a single-genre SNLI caption corpus transfers worse to the test genres than SNLI+MNLI does. |
| **Hard-negative coverage** (28.4% of anchors vs. 100%) | **~1.0** | **~1.0** | The paper gains +1.3 from hard negatives (Table 7, 84.9 → 86.2). We measured +0.23 (best) / +0.36 (matched steps). The missing ~1.0 is the gain we did not get. The next section explains why. |
| **No hyperparameter search; sup batch 128 with an LR tuned for 512** | **~0.3** (0–0.5) | **~0.3** | The paper reports that SimCSE is "not sensitive to batch sizes as long as tuning the learning rates accordingly". We did not tune the LR, so we assume a small loss. We tried 1 config against the paper's 12. |
| **Dev-selection optimism** (best of 16 dev evals) | (inflates our dev; shrinks the dev gap) | **~0.4** | Our sup dev curve spans 81.30–82.05. The selected 82.05 sits at the top of that band, so part of it does not carry over to test. |
| Pooling without MLP, no grad clipping, fp16, eval frequency | **~0** | **~0** | Table 6: sup [CLS] w/o MLP = 86.2 = w/ MLP. Clipping, fp16 and a denser eval schedule have no documented effect of this size. |
| **Unexplained / seed noise** | **~0.75** | **~0.7** | Residual; comparable to the ±0.5 single-seed floor. |
| **Total** | **4.15** | **5.50** | |

Roughly: on dev, data ≈ 50%, hard negatives ≈ 25%, hyperparameters ≈ 7%,
unexplained ≈ 18%. On test, data ≈ 56%, hard negatives ≈ 18%, hyperparameters
≈ 5%, selection ≈ 7%, unexplained ≈ 13%.

Data size and hard-negative coverage are not independent. Both come from
using one 100k-record SNLI subset, so the whole ~3.1 dev points (~4.1 test)
traces back to Part 2's data preparation, not to the training code.

### Unsupervised: −5.58 dev / −8.52 test

| Factor | Dev (pts) | Test (pts) | How the number was obtained |
|---|---|---|---|
| **Training data source and size** (165k short SNLI sentences vs. 10⁶ Wikipedia sentences) | **~4.5** (3.5–5) | **~6.5** | Our hyperparameters match Table A.1, so most of the gap has to come from the data. Our curve shows that data is the binding constraint: dev peaks at the **first** eval (step 125, after only 8,000 sentences). It then drifts **down** to 74.6 over the remaining 2,461 steps. More SNLI training hurts rather than helps, so the problem is not too few steps but the narrow caption-style domain. On test, the checkpoint (8k sentences into training) is still close to raw BERT, whose dev→test drop is 12.0. This accounts for ~2 of the 2.94 extra dev→test drop. |
| **Compute / steps** (2,586 vs. ~15,600) | **~0** directly | **~0** | Not binding. The best checkpoint is at step 125 and later steps lower dev. More steps on the same data would not close the gap; more steps on Wikipedia-scale data would, but that effect is counted under data. |
| **No hyperparameter search** | **~0.5** (0–1) | **~0.5** | We copied the paper's grid winner (bs 64, LR 3e-5) instead of re-tuning it for a different, much smaller corpus. The early peak suggests a lower LR or a shorter schedule might have helped, but we tried 1 config against the paper's 12. |
| **Dev-selection optimism** (best of 21 evals; the best is 1.06 above the next) | (inflates our dev; shrinks the dev gap) | **~1.0** | The selected step-125 score (76.92) is an isolated maximum; every later eval falls in 74.3–75.9. A single high reading on 1,500 dev pairs does not carry fully to test. |
| eval every 125 vs. 250 steps, fp16, pooling | **~0** | **~0** | Same pooling as the paper (MLP in training only, Table 6: 82.5). The denser eval schedule is, if anything, an advantage. |
| **Unexplained / seed noise** | **~0.6** | **~0.5** | Within the ±1 single-seed floor for unsup SimCSE. |
| **Total** | **5.58** | **8.52** | |

Roughly: on dev, data ≈ 80%, hyperparameters ≈ 9%, unexplained ≈ 11%. On
test, data ≈ 76%, selection ≈ 12%, hyperparameters ≈ 6%, unexplained ≈ 6%.

The unsupervised data shift is a change of **source**, not only of size. The
paper never trains unsup SimCSE on NLI sentences. Note that the Part 6 handoff
calls the paper's unsup corpus "NLI-derived", which is wrong (§3 fn. 2:
"We randomly sample 10⁶ sentences from English Wikipedia"). Because the
paper has no Wikipedia-subsample ablation, the ~4.5 is the residual left
after the other factors, not a direct extrapolation. Its range is wider than
the supervised data estimate.

## Part 5 finding: why our hard-negative effect is ~5× smaller than the paper's

The paper gets +1.3 dev from contradiction hard negatives (Table 7: 84.9 with
none, 86.2 with α = 1; Table 4 shows the same 84.9 → 86.2). Our Part 5
ablation (`ablation_sup_no_hardneg` vs. `sup_seed42_bs128`) gets +0.23 at the
best checkpoint and +0.36 averaged over 16 matched eval steps. That is
**3.6–5.7× smaller**, about 5×.

The main cause is **coverage**, which `reports/ablations.md` does not list.
In the paper, every anchor has a contradiction hypothesis (§4: "for each
premise and its entailment hypothesis, there is an accompanying contradiction
hypothesis"). In our data only 9,488 of 33,351 anchors (28.4%) have one,
because the 100k-record subset breaks up most premise triplets.
`src/train_simcse.py` adds only the non-null contradictions to the
denominator. A batch of 128 therefore has ~36 hard negatives, against 512 in
a paper batch of 512, and 72% of anchors never see a hard negative of their
own. In absolute terms we train on ~33× fewer hard negatives (9.5k vs. 314k).

If the paper's gain scaled linearly with coverage, we would expect
1.3 × 0.284 ≈ **0.37**. We measured **0.36** (matched-step mean). Coverage
alone accounts for nearly the whole 5× shortfall. The other reasons in
`ablations.md` (less data overall, early best checkpoints, the large in-batch
negative pool) may add to it, but they are not needed to explain the
magnitude. Small fixes for `ablations.md`: the paper's supervised set is
314k triplets, not ~275k, and the coverage effect should be listed first.

A direct fix would be to rebuild `sup_pairs.jsonl` from the full SNLI (or
SNLI+MNLI) so that every anchor has a contradiction, then re-run the ablation
pair with ≥ 3 seeds. We expect that to recover most of the ~1.0-point
hard-negative share of the supervised gap.

## Bottom line

- **Supervised (−4.15 dev / −5.50 test):** about three quarters of the gap is
  explained by the data subset: 9.4× fewer pairs (~2.1 / ~3.1) and 28%
  hard-negative coverage (~1.0). The rest is untuned hyperparameters (~0.3),
  dev-selection optimism on test (~0.4) and noise (~0.7).
- **Unsupervised (−5.58 dev / −8.52 test):** about four fifths of the gap is
  the training corpus (SNLI captions instead of 10⁶ Wikipedia sentences).
  Dev peaking after 8k sentences and then declining shows that more
  training on this corpus would not help. On test, an early, dev-optimistic
  checkpoint adds ~1.0 more.
- Neither gap points to a bug in the objective or the evaluation. The eval
  harness reproduces raw BERT and SBERT-2019 to ±0.00 (Part 4 Phase A). Our
  unsup model is 17.6 dev points above raw BERT. Our sup model beats
  SBERT-2019 by 1.3 dev and 1.8 test while training on 33k SNLI pairs
  against SBERT's ~1M SNLI+MNLI pairs (~1/30).
