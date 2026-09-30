# Part 2 — Build the training and evaluation data

Read `handoffs/00_INDEX.md` first for shared paths/conventions. This part has
no dependencies — start here.

## Input
`data/snli_train_100k.jsonl` — 100,000 lines, each a JSON object
`{"premise": str, "hypothesis": str, "label": 0|1|2}` where 0 = entailment,
1 = neutral, 2 = contradiction. Do not modify this file.

## Task

Write `src/data_prep.py` that produces four artifacts:

### 1. Unsupervised training set → `data/processed/unsup_sentences.txt`
The set of unique sentences across **both** the `premise` and `hypothesis`
fields of every record, one sentence per line, deduplicated. Expected count:
**165,529 unique sentences**. If your count differs, treat it as a bug in your
dedup logic (e.g. not stripping whitespace consistently) before proceeding —
don't silently accept a different number.

### 2. Supervised training set → `data/processed/sup_pairs.jsonl`
One JSON object per line: `{"premise": str, "entailment": str, "contradiction": str|null}`.
- Build from (premise, hypothesis) pairs where `label == 0` (entailment).
  Expected count: **33,351 pairs**.
- For each such premise, if the *same premise* also has a `label == 2`
  (contradiction) hypothesis elsewhere in the file, attach it as
  `"contradiction"`; otherwise `"contradiction": null`. Expected coverage:
  **~28%** of rows have a non-null contradiction — these are the paper's hard
  negatives (§4, Eq. 5). If a premise has multiple contradiction candidates,
  pick one deterministically (e.g. first occurrence) and note the rule used.

### 3. STS-B evaluation data → `data/stsb/dev.jsonl` and `data/stsb/test.jsonl`
Download from https://huggingface.co/datasets/sentence-transformers/stsb
(`load_dataset("sentence-transformers/stsb")`). Write each split as one JSON
object per line: `{"sentence1": str, "sentence2": str, "score": float}`
(raw score scale as provided by the dataset — do not rescale here, rescaling
if needed is the eval harness's job in Part 4). Expected counts: **dev = 1,500
pairs, test = 1,379 pairs**. Keep test completely untouched and never peek at
it outside Part 4/6's single final evaluation pass.

### 4. Manifest → `data/processed/MANIFEST.json`
```json
{
  "unsup_sentence_count": 165529,
  "sup_pair_count": 33351,
  "sup_pairs_with_hard_negative": 0,
  "sup_pairs_with_hard_negative_frac": 0.0,
  "stsb_dev_count": 1500,
  "stsb_test_count": 1379
}
```
Fill in the real numbers your script produced (the `sup_pairs_with_hard_negative*`
fields especially — Part 1 and Part 5 both read this file instead of
recomputing the fraction).

## Dependencies to add
`datasets` (for STS-B download) — add the exact installed version to
`requirements.txt`.

## Definition of done
- `data/processed/unsup_sentences.txt`, `data/processed/sup_pairs.jsonl`,
  `data/stsb/dev.jsonl`, `data/stsb/test.jsonl`, and
  `data/processed/MANIFEST.json` all exist.
- Counts in the manifest match the expected values above (165,529 / 33,351 /
  ~28% / 1,500 / 1,379); any deviation is explained in a comment in
  `src/data_prep.py`, not silently ignored.
- `src/data_prep.py` is re-runnable from scratch and deterministic.
