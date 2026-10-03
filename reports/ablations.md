# Part 5 — Ablations

One ablation per training mode. Each one changes **exactly one field** of the
Part 3 baseline config. Seed (42), batch size, LR, epochs, temperature,
pooling and the rest stay the same, so the delta comes from that one change.
`run_id` also differs, but it is only an identifier. All four runs trained on
Kaggle (Tesla T4, fp16 AMP, 1 GPU used). Every score is Spearman ×100 on the
STS-B **dev** split (1,500 pairs), computed by `src/evaluate.py`. "Best" means
the best of the periodic dev evaluations, which is also the checkpoint kept
under `runs/<run_id>/checkpoint/`.

Test deltas are **not reported yet**. The test split is scored exactly once,
in Part 6, for the final model only. If Part 6 also scores these checkpoints,
add the test column then.

## Results

| Pair | Baseline run | Ablated run | What changed (one field) | Dev baseline | Dev ablated | Δ dev |
|---|---|---|---|---|---|---|
| Unsupervised | `unsup_seed42_bs64` | `ablation_unsup_same_mask` | `same_dropout_mask: false → true` (both views reuse one forward pass and one dropout mask) | 76.92 (step 125) | 61.47 (step 1000) | **−15.4** |
| Supervised | `sup_seed42_bs128` | `ablation_sup_no_hardneg` | `use_hard_negatives: true → false` (Eq. 5 → Eq. 1-style loss on (premise, entailment) pairs, only in-batch negatives) | 82.05 (step 350) | 81.82 (step 150) | **−0.23** |

Reference: raw `bert-base-uncased` with mean pooling scores 59.31 dev.

Training curves (from each run's `results.json`):

| Run | Dev range over all evals | Same-step Δ vs baseline (mean ± sd, n) | Training loss, last 250 steps |
|---|---|---|---|
| `unsup_seed42_bs64` | 74.29 – 76.92 | — | ~1.3e-4 |
| `ablation_unsup_same_mask` | 61.17 – 61.47 | −13.8 ± 0.6 (n = 21, every step between −12.9 and −15.6) | ~7.6e-7 |
| `sup_seed42_bs128` | 81.30 – 82.05 | — | ~0.46 |
| `ablation_sup_no_hardneg` | 81.06 – 81.82 | −0.36 ± 0.17 (n = 16, ablation lower at 16/16 steps) | ~0.40 |

## Noise estimate

Each config has only one seed (42). No baseline was rerun with another seed,
so the noise level below is a reasoned estimate, not a measurement. It
combines three sources:

1. **Sampling noise in the dev set.** On n = 1,500 pairs at ρ ≈ 0.82, the
   Fisher-z standard error of a correlation is
   (1 − ρ²)/√(n − 3) ≈ 0.0085, or **about 0.85–0.9 points** for Spearman.
   Both models are scored on the same pairs, so the paired SE of the
   difference is smaller. It is still likely a few tenths of a point. A paired
   bootstrap on the two checkpoints would measure it.
2. **Variation between checkpoints in one run.** The supervised baseline's dev
   score moves within a 0.75-point band (81.30–82.05) across its 16 evals. The
   ablation moves within 0.76 points. "Best dev" is the maximum of 16 noisy
   readings, so it is biased upward by an amount close to that band's spread.
3. **Variation between seeds.** We did not measure it. For BERT-base
   contrastive fine-tuning on STS-B it is usually a few tenths of a point in
   supervised mode and around 1 point in unsupervised mode, where SimCSE is
   known to be more sensitive to seed.

Taking these together, a reasonable noise floor for comparing two runs that
each have one seed is **about ±0.5 points (supervised) and about ±1 point
(unsupervised)**.

- **Unsupervised, −15.4.** This is more than 15× the noise floor. The two dev
  curves never overlap: the ablation's best (61.47) is 12.8 points below the
  baseline's worst (74.29). The effect is real.
- **Supervised, −0.23.** This is **below** the single-seed noise floor. It is
  smaller than the sampling SE, smaller than either run's own variation
  between checkpoints, and smaller than typical seed variance. On its own it
  is **not a significant difference**. One observation points in the
  expected direction: the ablation is lower at all 16 matched eval steps
  (mean −0.36). Both runs share a seed and data order, so this comparison is
  partly paired. That consistency suggests a small real effect. It does not
  prove one, because the two runs' dropout RNG streams diverge after step 1:
  the ablated forward pass has fewer rows. Confirming it would take at least
  2 more seeds per config, comparing means.

## Mechanisms

### Unsupervised: a shared dropout mask removes the alignment signal (real collapse)

Unsupervised SimCSE (Eq. 1) has only one source of augmentation: two dropout
masks. In the baseline, each sentence appears twice in one forward pass, so
rows *i* and *B+i* get independent masks and become two slightly different
views. With `same_dropout_mask=true`, `src/train_simcse.py` runs one forward
pass and uses `z1 = z2 = z`, so the two views are literally the same tensor.
This has three consequences:

- After L2 normalisation the positive cosine is exactly 1. The positive logit
  is the constant 1/τ = 20, and its gradient is zero. The **alignment** term
  of the objective (Wang & Isola) is gone.
- The only way left to lower the loss is to push the 63 in-batch negatives
  apart. The objective becomes **uniformity only**. The model can reach almost
  zero loss without learning anything about which sentences mean similar
  things.
- On the STS-B dev split, that matches what we see. The run gains a little
  early, from spreading out BERT's anisotropic [CLS] space; it lands slightly
  above raw-BERT mean pooling (59.31). Then it stops improving: dev stays at
  61.2–61.5 for the whole epoch. The baseline, which does get an alignment
  gradient, reaches 76.92.

The paper's own ablation shows the same thing. In SimCSE §3, Table 3
("Effects of different dropout probabilities"), "Fixed 0.1" means the same
dropout mask for both views, and it causes a large drop on STS-B dev: 43.6,
down from 82.5 for the paper's default (different dropout masks). The
authors attribute it to representation collapse, because the positive pair
no longer carries any information. Our drop (76.92 → 61.47) is smaller in
absolute terms — likely because our setup (CLS pooling, `mlp_only_train`,
smaller/easier SNLI-derived training set) doesn't collapse as completely in
one epoch — but the direction and the mechanism are the same.

### The same log symptom had two different causes

Both unsupervised runs triggered the same automatic alert in
`run_record.json`: *"final training loss < 1e-3: possible collapse (identical
views?)"*. For the baseline it was a **false positive**. For this ablation it
is a **real collapse**. The symptom (near-zero InfoNCE loss) is the same, but
the cause differs:

| | `unsup_seed42_bs64` (baseline) | `ablation_unsup_same_mask` |
|---|---|---|
| Views | independent dropout masks | the same tensor (`z1 = z2`) |
| Positive cosine | ~0.83 on average (min 0.78), measured locally | exactly 1.0 by construction |
| Why the loss is small | τ = 0.05 turns the gap between positive and negatives (0.83 vs ~0.28) into a logit gap of ~11, so loss ≈ 63·e⁻¹¹ ≈ 1e-3. The task is easy, but the positive term still has a gradient. | The positive logit is a constant 20. The loss falls only through negative repulsion, a task that is trivially solvable. |
| Loss plateau | ~1e-4 | ~7.6e-7 (~170× lower), reached by step 50 |
| Dev Spearman | 76.92, well above raw BERT | 61.47, flat for all 21 evals |
| Verdict | not collapsed | collapsed alignment |

**Lesson:** a near-zero contrastive loss does not, on its own, diagnose
collapse. With a small τ, a model that has learned the task correctly can
also drive the loss toward zero. Two checks tell the cases apart. First,
check whether the positive pair is trivially identical (positive cosine equal
to 1, a single forward pass reused). Second, check whether dev Spearman moves
during training. The alert threshold is a useful prompt to look closer, but
it cannot decide the question.

### Supervised: hard negatives sharpen fine-grained separation, but the effect is small here

With `use_hard_negatives=true` (Eq. 5), each premise's softmax denominator
also contains the contradiction hypotheses of the batch. SNLI contradictions
usually overlap heavily with the premise in words and topic ("A man is
playing a guitar" / "A man is not playing an instrument"), so they are
negatives that sit close to the anchor. Pushing them away forces the encoder
to encode meaning differences that go beyond topic and lexical overlap. That
is what STS-B rewards in its middle range of similarity scores. Without hard
negatives (Eq. 1 on the supervised pairs), the 127 in-batch negatives are
mostly unrelated sentences. They are easy to separate, so the task gets
easier. Its final loss is ~0.40, against ~0.46 for the baseline, and it
pushes less on the semantic boundary.

The measured effect (−0.23 best, −0.36 at matched steps) has the expected
sign but is far smaller than in the paper. SimCSE Table 7 ("STS-B
development results with different hard negative policies") reports +1.3
points on STS-B dev from adding contradiction hard negatives (no hard
negative: 84.9 → with weighted contradiction: 86.2; contradiction alone
already reaches 86.1, so most of the gain comes from adding the hard
negative at all, not from the weighting). Likely reasons:

- **Less data.** Our 100k-row SNLI subset gives 33,351 triplets, against
  ~275k SNLI+MNLI triplets in the paper. That means fewer hard negatives in
  total.
- **Early best checkpoints.** Both runs peak early (steps 150 and 350 of
  780), so most of the training where hard negatives would matter does not
  affect the selected checkpoint.
- **A large batch.** With batch size 128, the in-batch entailment negatives
  already supply much of the uniformity signal.

With one seed per config, we can only say that hard negatives help by at most
a few tenths of a point in this setup. We cannot claim a significant gain.
