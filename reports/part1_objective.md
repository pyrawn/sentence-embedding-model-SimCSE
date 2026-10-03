# Part 1 — Understanding the objective

Paper references: Gao, Yao & Chen, *SimCSE*, EMNLP 2021
(arXiv 2104.08821). Every equation, table and quote below was checked
against the arXiv PDF. In the paper, **Eq. 1** is the generic in-batch
contrastive loss for a pair (xᵢ, xᵢ⁺). **Eq. 4** instantiates it for
unsupervised SimCSE, with xᵢ⁺ = xᵢ and two dropout masks z, z′. **Eq. 5**
is the supervised loss with contradiction hard negatives. Our numbers come
from `configs/*.json`, `data/processed/MANIFEST.json` and
`reports/benchmark_table.md`.

## 1. Mechanics of Eq. 1 (and Eq. 5): how many negatives per step?

For anchor *i* in a mini-batch of N pairs, Eq. 1 is

  ℓᵢ = −log [ exp(sim(hᵢ, hᵢ⁺)/τ) / Σⱼ₌₁ᴺ exp(sim(hᵢ, hⱼ⁺)/τ) ],  sim = cosine.

- **Numerator:** the exponentiated, temperature-scaled cosine between the
  anchor and its *own* positive. In unsupervised SimCSE (Eq. 4) the positive
  is the same sentence encoded a second time with an independent dropout mask.
- **Denominator:** a sum over the positives of **all N** examples in the
  batch. That includes the anchor's own positive (j = i), so the loss is an
  N-way softmax classification: "which of the N candidates is my positive?"
  The other N − 1 terms are **in-batch negatives**. They are the other
  sentences' second views, not extra mined examples. In our code this is
  `cross_entropy(z1 @ z2.T / τ, arange(N))` (`src/train_simcse.py`).

**Unsupervised, our batch N = 64** (`configs/unsupervised.json`): each anchor
is pushed away from **N − 1 = 63 negatives** in every step. This is the same
batch as the paper's unsup BERT-base setting (Table A.1: 64).

**Supervised, Eq. 5.** Each example is a triplet (premise xᵢ, entailment
xᵢ⁺, contradiction xᵢ⁻), and the denominator sums over **both**
exp(sim(hᵢ, hⱼ⁺)/τ) and exp(sim(hᵢ, hⱼ⁻)/τ) for j = 1..N. With full coverage
an anchor would have (N − 1) entailment negatives plus N hard negatives,
i.e. 2N − 1. Our implementation adds every **non-null** contradiction in the
batch as an extra column. Only **28.4%** of our supervised pairs have one
(9,488 of 33,351, `sup_pairs_with_hard_negative_frac = 0.284`). With **our
batch N = 128** (`configs/supervised.json`):

| | Entailment (in-batch) negatives | Hard negatives (contradictions) | Total per anchor |
|---|---|---|---|
| Ours, `use_hard_negatives=true` | 127 | ≈ 0.284 × 128 ≈ **36** (expected value; varies per batch) | **≈ 163** |
| Same batch, 100% coverage (hypothetical) | 127 | 128 | 255 |
| Ours, Part 5 ablation `use_hard_negatives=false` | 127 | 0 | **127** |
| Paper (N = 512, every premise has a contradiction) | 511 | 512 | 1,023 |

Every anchor sees the ~36 contradictions present in its batch, but only
28.4% of anchors have **their own** contradiction, the one that is lexically
closest to them and therefore the most informative hard negative. Turning
this term on and off is exactly what the Part 5 supervised ablation
(`ablation_sup_no_hardneg`) does. Its small effect (−0.23 dev, −1.45 test,
see `reports/ablations.md`) is consistent with this low coverage
(`reports/gap_analysis.md`).

## 2. Temperature τ

τ divides every cosine before the softmax. Cosines lie in [−1, 1]. With
τ = 0.05 (our value and the paper's) they become logits in [−20, 20], so
small differences in cosine turn into large differences in probability.

- **Small τ → sharp softmax.** The loss is dominated by the negatives that
  are most similar to the anchor (the hardest ones), and their gradients get
  the largest weight. The objective strongly penalizes near-duplicate
  negatives and pushes them apart.
- **Large τ → flat softmax.** All negatives are weighted almost equally. The
  objective barely distinguishes a close negative from a far one, and the
  pressure to spread the embeddings out (uniformity) weakens.

Our own logs show how strong this effect is at τ = 0.05. In the unsupervised
baseline, positives have cosine ≈ 0.83 and random pairs ≈ 0.28. The
0.55-point cosine gap becomes a logit gap of 0.55 / 0.05 ≈ 11. With 63
negatives, the loss is ≈ 63·e⁻¹¹ ≈ 10⁻³. That is why the training loss
reached ~10⁻⁴ without any collapse (`runs/run_log.jsonl`, notes of
`unsup_seed42_bs64`).

**What the paper found** (Appendix D, Table D.1, STS-B **dev** Spearman,
cosine similarity with different τ, plus a dot-product variant "N/A"):

| τ | N/A (dot product) | 0.001 | 0.01 | **0.05** | 0.1 | 1 |
|---|---|---|---|---|---|---|
| STS-B dev | 85.9 | 84.9 | 85.4 | **86.2** | 82.0 | 64.0 |

The effect is **not monotonic**. Performance peaks at τ = 0.05. It falls
sharply when τ is too large (82.0 at 0.1 and 64.0 at 1, a 22-point loss). It
falls only mildly when τ is too small (84.9 at 0.001). The paper's
conclusion: "with a carefully tuned temperature τ = 0.05, cosine similarity
is better than dot product". *Our inference, not stated in the paper:* the
table does not name the model, but its τ = 0.05 value (86.2) matches the
paper's final **supervised** BERT-base dev score (Table 7). The ablation was
therefore most likely run in the supervised setting. The paper does not
report a τ sweep for unsupervised SimCSE.

## 3. SimCSE vs. Sentence-BERT (2019): why the objective matters for STS-B

SBERT-2019 (`bert-base-nli-mean-tokens`, our reference row) trains a
siamese BERT with a **classification objective** on NLI. The two sentence
embeddings u, v are combined as (u, v, |u − v|), and a softmax layer predicts
entailment / neutral / contradiction (Reimers & Gurevych, 2019). The loss
only requires that **some linear classifier on top of (u, v, |u − v|)** can
separate the three labels. Nothing in it says that cos(u, v) has to be high
for paraphrases and low for unrelated sentences. Cosine geometry is, at
best, a side effect. The classifier is thrown away at inference.

SimCSE's loss is **written in terms of the cosine** (sim = cosine in
Eq. 1/4/5). It directly raises cos(hᵢ, hᵢ⁺) and lowers cos(hᵢ, hⱼ) for every
other sentence in the batch.

This difference matters for STS-B because of how we (and the paper)
evaluate (`src/evaluate.py`, Part 4). We embed each sentence, compute the
**raw cosine** between the two embeddings, and report its Spearman
correlation with the gold score. No regressor is trained on STS-B (the
paper follows the same "no additional regressor" setting, §6.1). The
evaluation reads exactly the quantity SimCSE optimizes, and that SBERT's
objective only constrains indirectly. Two consequences show up in our
numbers:

- With the same base model and NLI supervision, our supervised SimCSE beats
  SBERT-2019 (82.05 vs. 80.77 dev, 78.75 vs. 76.98 test). It does so while
  training on ~1/30 of SBERT's NLI pairs (33k SNLI pairs vs. ~1M SNLI+MNLI).
- The in-batch negatives also add a term that SBERT does not have: pushing
  apart *all* unrelated sentences, not only the labeled pairs. This shows in
  uniformity, −3.36 for our sup model vs. −3.05 for SBERT-2019 (next
  section). Spearman is a *ranking* metric over all 1,379 test pairs, so
  spreading out unrelated sentences, and not only separating labeled pairs,
  is what makes the cosine ranking reliable.

## 4. Alignment and uniformity

Wang & Isola (ICML 2020) define two properties of normalized embeddings
f(x), both lower = better:

- **Alignment** = E ‖f(x) − f(x⁺)‖² over positive pairs: positives should be
  close.
- **Uniformity** = log E exp(−2‖f(x) − f(y)‖²) over random pairs: the
  embeddings should spread over the hypersphere.

**Why one loss does both.** Split −log of the softmax in Eq. 1 into two
terms:

  ℓᵢ = −sim(hᵢ, hᵢ⁺)/τ  +  log Σⱼ exp(sim(hᵢ, hⱼ⁺)/τ).

The first term only depends on the positive pair. Minimizing it raises the
positive cosine, which on the unit sphere is the same as reducing
‖hᵢ − hᵢ⁺‖² (= 2 − 2cos). That is **alignment**. The second term is a
log-sum-exp over the anchor's similarity to everything in the batch.
Minimizing it pushes the anchor away from all other sentences, most of all
the closest ones. This is a soft version of the uniformity potential. The
paper makes this precise (§5, Eq. 6): as the number of negatives → ∞, the
loss tends to −(1/τ)·E[f(x)ᵀf(x⁺)] + E_x log E_{x⁻} exp(f(x)ᵀf(x⁻)/τ). The
first term keeps positives similar ("alignment"), and the second "pushes
negative pairs apart" ("uniformity"). The paper also shows the second term
flattens the singular-value spectrum of the embedding matrix, which reduces
BERT's anisotropy (§5, Eq. 7).

**What our numbers show** (STS-B dev, `reports/benchmark_table.md` and
`reports/eval_<run_id>.json`; alignment uses the 264 dev pairs with gold ≥ 4,
uniformity uses the 2,910 unique dev sentences):

| Model | Alignment ↓ | Uniformity ↓ | Dev | Test |
|---|---|---|---|---|
| raw `bert-base-uncased` (mean) | 0.195 | −1.650 | 59.31 | 47.29 |
| Our unsup SimCSE (`unsup_seed42_bs64`) | **0.305** | **−2.963** | 76.92 | 68.33 |
| Our sup SimCSE (`sup_seed42_bs128`) | **0.220** | **−3.362** | 82.05 | 78.75 |
| SBERT-2019 | 0.193 | −3.049 | 80.77 | 76.98 |
| Ablation: unsup, same dropout mask | 0.561 | −3.081 | 61.47 | 52.33 |
| Ablation: sup, no hard negatives | 0.190 | −3.086 | 81.82 | 77.30 |

- **Raw BERT has "good" alignment only because everything is close.** Its
  uniformity (−1.65) is by far the worst: the space is an anisotropic cone in
  which every pair, related or not, has a high cosine. This matches the
  paper's observation (§7: pre-trained embeddings "have good alignment, [but]
  their uniformity is poor").
- **Unsupervised SimCSE mainly buys uniformity.** Uniformity improves from
  −1.65 to −2.96, while alignment gets somewhat worse (0.195 → 0.305).
  Dropout views are a very weak positive signal, so the second term of the
  loss dominates. The net result is +17.6 dev points over raw BERT. The paper
  reports that unsup SimCSE improves uniformity "whereas keeping a good
  alignment". Ours gives up more alignment than that, which is consistent
  with training on a narrow SNLI-caption corpus (see `gap_analysis.md`).
- **Supervision brings alignment back.** Entailment pairs are real
  paraphrases, so the first term now carries semantic signal: alignment
  improves 0.305 → 0.220. The hard-negative and in-batch terms push
  uniformity further, to −3.36. This is the model that is best on both axes
  relative to the unsup one, and it is our best STS-B model, again as the
  paper reports (§7: supervised data "further amends alignment").
- **The two ablations separate the two terms.** With a *shared* dropout mask
  (`ablation_unsup_same_mask`), the positive is identical to the anchor, so
  the first term is constant and has zero gradient. Only the uniformity term
  is trained. The result is exactly that: uniformity of −3.08, *better* than
  the unsup baseline, but alignment collapses to 0.561, and STS-B drops to
  61.47 dev / 52.33 test. Uniformity without alignment is not enough.
  Removing **hard negatives** (`ablation_sup_no_hardneg`) moves the balance
  the other way. Alignment is slightly better (0.190), but uniformity is
  clearly worse (−3.09 vs. −3.36), because the closest negatives (the
  contradictions) are gone from the second term. Test drops by 1.45.
