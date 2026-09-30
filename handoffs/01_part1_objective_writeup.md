# Part 1 — Understand the objective (write-up, no training code)

Read `handoffs/00_INDEX.md` first for shared paths/conventions. This part is
conceptual: produce a written answer, not a trained model. It can be drafted
early, but the final numeric sub-answer needs `configs/unsupervised.json` and
`configs/supervised.json` from Part 3 (batch sizes) — finalize this file last,
after Part 3's configs exist.

## Output

Write `reports/part1_objective.md` answering exactly these four questions,
grounded in Eq. 1 (unsupervised) and Eq. 5 (supervised, hard negatives) of the
SimCSE paper (https://arxiv.org/abs/2104.08821):

### 1. Eq. 1 mechanics
- What is in the numerator, and what does the denominator sum over?
- With your chosen batch size N (pull the actual value from
  `configs/unsupervised.json`), how many negatives does one optimization step
  push the positive away from?
  - Unsupervised (Eq. 1): for a batch of N sentences, each encoded twice under
    independent dropout, the loss for anchor i is
    `-log( exp(sim(h_i, h_i+)/τ) / Σ_{j=1..N} exp(sim(h_i, h_j+)/τ) )`.
    The denominator sums over the N "positive" embeddings in the batch
    (including the anchor's own), so each anchor has **N − 1 in-batch
    negatives**. Plug in the real N and state the number.
  - Supervised (Eq. 5): the denominator sums over both `h_j+` (entailment) and
    `h_j-` (hard negative, when present) for j = 1..N. Each anchor gets
    `N − 1` negatives from other examples' entailment embeddings, plus up to
    `N` hard negatives — but only ~28% of premises in the supervised set have
    one (see `data/processed/MANIFEST.json` from Part 2 for the exact
    fraction). State the resulting approximate negative count for your actual
    batch size, and note this is exactly what the Part 5 ablation
    (hard negatives on/off) toggles.

### 2. Temperature τ
- What does τ control in Eq. 1 (sharpness of the softmax over similarities /
  how hard the objective penalizes near-duplicate negatives)?
- What did the paper find when it ablated τ (cite the direction of the effect
  reported in the paper, not just "it matters")?

### 3. SimCSE vs. Sentence-BERT (2019) objective
- Sentence-BERT is trained with a classification/regression objective over
  labeled pairs (e.g., softmax over NLI labels or regression to a similarity
  score), not an in-batch contrastive objective.
- Explain why this difference matters specifically for STS-B evaluation
  (cosine similarity on raw embeddings, no fine-tuned regressor on top — see
  Part 4): a contrastive objective directly optimizes the geometry that
  cosine similarity reads off, while a classification objective does not
  guarantee that geometry.

### 4. Alignment and uniformity
- Explain, in your own words, how minimizing the SimCSE loss simultaneously
  pulls positive pairs together (alignment) and pushes the full embedding
  distribution toward uniformity on the hypersphere (Wang & Isola, ICML 2020,
  https://arxiv.org/abs/2005.10242).
- This connects directly to the metrics computed in Part 4 Phase B and used in
  the Part 6 table — reference those actual numbers once they exist, don't
  leave this purely abstract.

## Definition of done
`reports/part1_objective.md` exists with all four numbered answers, the
negative-count sub-answers use the real batch sizes from the Part 3 configs
(not placeholders), and section 4 references the actual alignment/uniformity
numbers computed in Part 4/6 once available.
