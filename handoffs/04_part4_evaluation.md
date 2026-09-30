# Part 4 — Evaluation harness (two phases — Phase A blocks Part 3)

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

This part is split into two phases because the PDF requires the eval code to
be verified *before* any training run starts. Do Phase A immediately after
Part 2 finishes; Phase B only after Part 3 (and Part 5) produce checkpoints.

## Phase A — build and sanity-check `src/evaluate.py` (blocks Part 3)

### Task
Implement `src/evaluate.py` exposing at minimum:

```python
def embed_sentences(model, tokenizer, sentences, pooling="cls", batch_size=64, device=...) -> np.ndarray:
    """Return L2-normalized embeddings, shape (N, hidden_size)."""

def evaluate_sts(model, tokenizer, split_path, pooling="cls") -> dict:
    """
    split_path points at data/stsb/dev.jsonl or data/stsb/test.jsonl (Part 2 output).
    Embeds sentence1/sentence2 independently, normalizes, takes cosine similarity
    (no regressor, no fine-tuning on STS-B), and returns:
      {"spearman": float, "similarities": list[float], "human_scores": list[float]}
    """
```
- Cosine similarity on **normalized** embeddings only — no linear/MLP head on
  top, per the assignment (Appendix B of the SimCSE paper explains why).
- Correlate `similarities` vs. `human_scores` with Spearman's rho
  (`scipy.stats.spearmanr`).
- Must work for: (a) a raw `transformers` `AutoModel` checkpoint (pooling is a
  parameter — support at least `"cls"` and `"mean"`), and (b) a
  `sentence-transformers` `SentenceTransformer` model (used for SBERT-2019 and
  for reloading the published model in Part 7). Keep the two code paths behind
  one function signature so Part 3/5/6/7 don't need to know which kind of
  model they're passing.

### Sanity check (mandatory gate before Part 3 starts)
Run `evaluate_sts` against:
- raw `bert-base-uncased` with mean pooling on `data/stsb/dev.jsonl` and
  `data/stsb/test.jsonl` — must land close to **59.31 dev / 47.29 test**
  (Spearman ×100).
- `sentence-transformers/bert-base-nli-mean-tokens` (SBERT-2019) on the same
  splits — must land close to **80.77 / 76.98**.

If either number is far off, the bug is in `evaluate.py` (pooling, missing
normalization, wrong similarity, off-by-one in alignment of
sentence1/sentence2/score, or scoring the wrong file) — fix it here, do not
let Part 3 start training against unverified eval code. Save these two
reference runs to `runs/run_log.jsonl` (mode `"reference"`) so Part 6 can pull
the numbers from there instead of recomputing them.

## Phase B — full evaluation analysis (after Part 3/5 checkpoints exist)

For every trained checkpoint (unsupervised, supervised, and each ablation
variant), in addition to `evaluate_sts`, compute and save under
`reports/figures/` and `reports/eval_<run_id>.json`:

1. **Alignment & uniformity** (Wang & Isola, ICML 2020). Using STS-B dev pairs
   (normalized embeddings):
   - `alignment = mean( ||f(x) - f(x+)||^2 )` over pairs whose gold score ≥ 4
     (treated as positive pairs — this is the paper's own convention for
     reporting these numbers on STS-B).
   - `uniformity = log( mean( exp(-2 * ||f(x_i) - f(x_j)||^2) ) )` over all
     unique sentences in STS-B dev, i ≠ j pairs.
   - Lower alignment is better (closer positives); more negative uniformity is
     better (more spread out).
2. **Similarity distribution by human rating**: a plot (e.g. box plot or
   violin plot) of predicted cosine similarity, grouped by integer-rounded
   gold STS-B score (0–5). Save as
   `reports/figures/sim_distribution_<run_id>.png`.
3. **Nearest-neighbor retrievals**: pick a handful of query sentences (from
   STS-B dev or the unsupervised sentence pool), retrieve top-k by cosine
   similarity among the model's embeddings, and include at least one
   **failure case** (a clearly wrong top-1 neighbor) with a short discussion
   of why it likely failed. Save as `reports/retrievals_<run_id>.md`.

## Definition of done
- Phase A: `src/evaluate.py` exists, sanity-check numbers are within a small
  tolerance of the reference values, both reference runs are logged in
  `runs/run_log.jsonl`. Part 3 is blocked until this is true.
- Phase B: for every run in `runs/run_log.jsonl` with `mode` in
  `{"unsupervised", "supervised", "ablation_*"}`, there is a matching
  `reports/eval_<run_id>.json` (with alignment/uniformity), a similarity
  distribution figure, and a retrievals file with ≥1 discussed failure case.
