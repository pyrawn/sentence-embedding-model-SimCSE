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
- name: simcse-bert-base-snli-sup
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
      value: 78.75
      name: Spearman (cosine) x100
---

# simcse-bert-base-snli-sup

Supervised SimCSE ([Gao, Yao & Chen, EMNLP 2021](https://arxiv.org/abs/2104.08821), Eq. 5: entailment pairs as positives, contradictions as hard negatives) on top of `bert-base-uncased`, trained on **a 100k-record SNLI subset** as a course assignment (U2T02, Trends in Data Science). It maps a sentence to a 768-dimensional, L2-normalized vector; compare vectors with cosine similarity.

**Team:** Julio Cesar De Aquino Castellanos, Gustavo Alexander Fuentes Marin, Valeria Nicol Hernandez Leon, Ricardo Daniel Horta Sanchez, Jose Angel Pech Xool, Lorena Danae Perez Lopez.

This is a reproduction under a reduced budget, **not** the paper's released checkpoint: it is 5.5 Spearman points below the paper on STS-B test (see [Limitations](#limitations)). It is our best model, and it beats SBERT-2019 on STS-B dev and test while training on about 1/30 of its data. The unsupervised companion model is [`pyrawn/simcse-bert-base-snli-unsup`](https://huggingface.co/pyrawn/simcse-bert-base-snli-unsup) (68.33 test).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pyrawn/simcse-bert-base-snli-sup")
emb = model.encode(["A man is playing a guitar.", "Someone plays an instrument."])
print(model.similarity(emb[0:1], emb[1:2]))  # cosine similarity
```

Architecture: `Transformer` (BERT-base, max 512 tokens) → `Pooling(cls)` (last-layer `[CLS]` hidden state; this model was trained without an MLP head, and BERT's pretrained pooler is not used) → `Normalize`.

## Training data

Not the paper's data. The paper trains supervised SimCSE on 314k SNLI + MNLI (premise, entailment, contradiction) triplets; this model was trained on **33,351 (premise, entailment) pairs** from a fixed 100k-record subset of the SNLI training set (`data/snli_train_100k.jsonl`, SHA-256 `0927a0b1…7f88`; records without annotator consensus dropped). Only **9,488 pairs (28.4%)** have a hard negative: the first contradiction hypothesis in file order for the exact same premise. The other 71.6% are trained with in-batch negatives only. Counts, the hard-negative rule and checksums are in the project's `data/processed/MANIFEST.json`. No MNLI data is used.

## Training recipe

From `runs/sup_seed42_bs128/config.json`:

| Hyperparameter | Value |
|---|---|
| Mode | supervised (entailment positives + contradiction hard negatives, in-batch negatives) |
| Base model | `bert-base-uncased` |
| Training pooling | `[CLS]`, no MLP (`pooling: cls`, `mlp_only_train: false`) |
| Inference / eval pooling | `cls` |
| Use hard negatives | true |
| Batch size | 128 (paper: 512) |
| Learning rate | 5e-5, linear decay, warmup ratio 0.0 |
| Optimizer | AdamW, weight decay 0.0, no gradient clipping (`max_grad_norm: null`) |
| Epochs | 3 (780 steps) |
| Temperature τ | 0.05 |
| Max sequence length (training) | 32 |
| Dropout (hidden / attention) | 0.1 / 0.1 |
| Precision | fp16 (AMP) |
| Seed | 42 |
| Dev evaluation / checkpoint selection | every 50 steps on STS-B dev; best kept (**step 350**) |
| Hardware | 1× Tesla T4 (Kaggle), ~7 min |
| Libraries | torch 2.10.0, transformers 5.18.0, Python 3.12 |

## Evaluation

STS-B (`sentence-transformers/stsb`), protocol of the SimCSE paper (Appendix B): embed each sentence independently, cosine similarity, Spearman ρ against gold scores, no STS-B fine-tuning. Spearman ×100:

| Model | Dev | Test | Alignment ↓ | Uniformity ↓ |
|---|---|---|---|---|
| **This model** | **82.05** | **78.75** | **0.220** | **−3.362** |
| Unsupervised companion (`simcse-bert-base-snli-unsup`) | 76.92 | 68.33 | 0.305 | −2.963 |
| raw `bert-base-uncased` (mean pooling) | 59.31 | 47.29 | 0.195 | −1.650 |
| SBERT-2019 (`bert-base-nli-mean-tokens`) | 80.77 | 76.98 | 0.193 | −3.049 |
| SimCSE paper, sup BERT-base (reported) | 86.20 | 84.25 | — | — |

- Dev is the best periodic dev score during training (used for checkpoint selection, so optimistically biased). Test was scored once on the kept checkpoint and **reproduced after publishing** by reloading this repo from the Hub into an empty cache (same value to floating-point precision).
- Alignment / uniformity (Wang & Isola, 2020) on STS-B dev, L2-normalized: alignment over the 264 dev pairs with gold ≥ 4; uniformity over the 2,910 unique dev sentences.
- Paraphrase retrieval over STS-B dev (264 gold ≥ 4 pairs, 2,910-sentence corpus): Recall@1 0.890, Recall@5 0.985, MRR 0.929.
- Ablation: the same recipe without hard negatives reaches 81.82 dev (+0.23 from hard negatives, vs. +1.3 in the paper; the difference is explained by the 28.4% hard-negative coverage).

## Limitations

- **Gap to the paper: −4.15 dev / −5.50 test.** About three quarters of it is attributed to the data subset: 9.4× fewer training pairs from a single genre (SNLI only, no MNLI; ~2.1 dev / ~3.1 test) and hard negatives for only 28.4% of anchors (~1.0). The rest is untuned hyperparameters (batch 128 with the LR the paper tuned for 512; ~0.3), dev-selection optimism on test (~0.4) and single-seed noise (~0.7; the noise floor is about ±0.5). One configuration and one seed were run.
- **Same topic, different event.** Headlines that share a template but report different events are treated as near-paraphrases: "6.4-magnitude quake strikes off Indonesia" ↔ "6.9-magnitude quake strikes off Russia's Kuril Islands" (cosine 0.84, gold 1/5); "Military plane crashes in south France" → "…southeastern Turkey, 1 dead" (0.78, gold 1/5). SNLI contradictions are about scene content, so nothing teaches the model that changing a place, number or named entity changes the meaning.
- **Domain.** Trained only on English image-caption sentences of ≤ 32 tokens; news-style text is where most STS-B errors occur. Expect weaker results on long text, other languages, and specialized domains. Not evaluated on anything other than STS-B.
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
