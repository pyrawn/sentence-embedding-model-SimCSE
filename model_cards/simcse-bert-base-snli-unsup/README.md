---
language: en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: google-bert/bert-base-uncased
datasets:
- stanfordnlp/snli
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- simcse
- contrastive-learning
model-index:
- name: simcse-bert-base-snli-unsup
  results:
  - task:
      type: semantic-similarity
      name: Semantic Textual Similarity
    dataset:
      type: sentence-transformers/stsb
      name: STS-B (test)
      split: test
    metrics:
    - type: spearman_cosine
      value: 68.33
      name: Spearman (cosine) x100
---

# simcse-bert-base-snli-unsup

Unsupervised SimCSE ([Gao, Yao & Chen, EMNLP 2021](https://arxiv.org/abs/2104.08821), Eq. 1) on top of `bert-base-uncased`, trained on **sentences from a 100k-record SNLI subset** as a course assignment (U2T02, Trends in Data Science). It maps a sentence to a 768-dimensional, L2-normalized vector; compare vectors with cosine similarity.

**Team:** Julio Cesar De Aquino Castellanos, Gustavo Alexander Fuentes Marin, Valeria Nicol Hernandez Leon, Ricardo Daniel Horta Sanchez, Jose Angel Pech Xool, Lorena Danae Perez Lopez.

This is a reproduction under a reduced budget, **not** the paper's released checkpoint: it is 8.5 Spearman points below the paper on STS-B test (see [Limitations](#limitations)). The supervised companion model is [`pyrawn/simcse-bert-base-snli-sup`](https://huggingface.co/pyrawn/simcse-bert-base-snli-sup) (78.75 test) and is better on every metric below.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pyrawn/simcse-bert-base-snli-unsup")
emb = model.encode(["A man is playing a guitar.", "Someone plays an instrument."])
print(model.similarity(emb[0:1], emb[1:2]))  # cosine similarity
```

Architecture: `Transformer` (BERT-base, max 512 tokens) → `Pooling(cls)` (last-layer `[CLS]` hidden state, **without** the MLP used during training and without BERT's pretrained pooler) → `Normalize`.

## Training data

Not the paper's data. The paper trains unsupervised SimCSE on 10⁶ sentences sampled from English Wikipedia; this model was trained on **165,529 unique sentences** (premises and hypotheses, exact-string deduplication) extracted from a fixed 100k-record subset of the SNLI training set (`data/snli_train_100k.jsonl`, SHA-256 `0927a0b1…7f88`). Counts and checksums are in the project's `data/processed/MANIFEST.json`. SNLI sentences are short English image captions, a much narrower domain than Wikipedia. Labels are not used.

## Training recipe

From `runs/unsup_seed42_bs64/config.json`:

| Hyperparameter | Value |
|---|---|
| Mode | unsupervised (dropout noise as the only augmentation; in-batch negatives) |
| Base model | `bert-base-uncased` |
| Training pooling | `[CLS]` + MLP (`pooling: cls`, `mlp_only_train: true`) |
| Inference / eval pooling | `cls_before_pooler` (`[CLS]` without the MLP) |
| Batch size | 64 |
| Learning rate | 3e-5, linear decay, warmup ratio 0.0 |
| Optimizer | AdamW, weight decay 0.0, max grad norm 1.0 |
| Epochs | 1 (2,586 steps) |
| Temperature τ | 0.05 |
| Max sequence length (training) | 32 |
| Dropout (hidden / attention) | 0.1 / 0.1 |
| Same dropout mask for both views | false |
| Precision | fp16 (AMP) |
| Seed | 42 |
| Dev evaluation / checkpoint selection | every 125 steps on STS-B dev; best kept (**step 125**) |
| Hardware | 1× Tesla T4 (Kaggle), ~10 min |
| Libraries | torch 2.10.0, transformers 5.18.0, Python 3.12 |

## Evaluation

STS-B (`sentence-transformers/stsb`), protocol of the SimCSE paper (Appendix B): embed each sentence independently, cosine similarity, Spearman ρ against gold scores, no STS-B fine-tuning. Spearman ×100:

| Model | Dev | Test | Alignment ↓ | Uniformity ↓ |
|---|---|---|---|---|
| **This model** | **76.92** | **68.33** | **0.305** | **−2.963** |
| Supervised companion (`simcse-bert-base-snli-sup`) | 82.05 | 78.75 | 0.220 | −3.362 |
| raw `bert-base-uncased` (mean pooling) | 59.31 | 47.29 | 0.195 | −1.650 |
| SBERT-2019 (`bert-base-nli-mean-tokens`) | 80.77 | 76.98 | 0.193 | −3.049 |
| SimCSE paper, unsup BERT-base (reported) | 82.50 | 76.85 | — | — |

- Dev is the best periodic dev score during training (used for checkpoint selection, so optimistically biased). Test was scored once on the kept checkpoint and **reproduced after publishing** by reloading this repo from the Hub into an empty cache (same value to floating-point precision).
- Alignment / uniformity (Wang & Isola, 2020) on STS-B dev, L2-normalized: alignment over the 264 dev pairs with gold ≥ 4; uniformity over the 2,910 unique dev sentences.
- Paraphrase retrieval over STS-B dev (264 gold ≥ 4 pairs, 2,910-sentence corpus): Recall@1 0.830, Recall@5 0.958, MRR 0.885.

## Limitations

- **Gap to the paper: −5.58 dev / −8.52 test.** About four fifths of it is attributed to the training corpus (165k short SNLI captions instead of 10⁶ Wikipedia sentences). Dev peaked at the very first evaluation (step 125, 8,000 sentences seen) and then drifted down to ~74.6, so more training on this corpus would not help. The test gap is larger than the dev gap partly because the selected checkpoint is an isolated dev maximum (~1 point of dev-selection optimism). One configuration and one seed were run (no hyperparameter search); the single-seed noise floor is about ±1 point.
- **Lexical-template failures.** Without paraphrase supervision the model keeps sentences with the same syntactic frame and topic close even when the action/object differ: "The man is stirring the rice." retrieves "The man is buttering the bread." (cosine 0.80, gold 0.4/5) above "A person is mixing a pot of rice."; "cracking eggs into a bowl" ranks the true paraphrase "broke raw eggs into a bowl" only 6th.
- **Same topic, different event.** News headlines sharing a template ("6.4-magnitude quake strikes off Indonesia" vs. "6.9-magnitude quake strikes off Russia's Kuril Islands", cosine 0.84, gold 1/5) are treated as near-paraphrases. The model encodes topic and form more than numbers or named entities.
- **Domain.** Trained only on English image-caption sentences of ≤ 32 tokens. Expect weaker results on long text, other languages, and specialized domains. Not evaluated on anything other than STS-B.
- Inherits the biases of BERT's pretraining data and of SNLI.

## Citation

```bibtex
@inproceedings{gao2021simcse,
  title={SimCSE: Simple Contrastive Learning of Sentence Embeddings},
  author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle={EMNLP},
  year={2021}
}
```
