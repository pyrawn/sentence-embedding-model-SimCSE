"""Part 4 (Phase A) — STS-B evaluation harness shared by Parts 3/5/6/7.

Protocol (SimCSE paper, Appendix B): embed sentence1 and sentence2
independently, L2-normalize, score each pair by cosine similarity (no regressor,
no fine-tuning on STS-B), and report Spearman's rho between the cosine
similarities and the gold scores.

Works with two kinds of model behind the same signature:
  * a raw `transformers` model (AutoModel) + tokenizer, pooling in
    {"cls", "cls_before_pooler", "pooler", "mean"};
  * a `sentence_transformers.SentenceTransformer` (tokenizer ignored; the model's
    own pooling module is used and the `pooling` argument is ignored).

CLI:
  # sanity check the harness against the reference numbers and log both runs
  python src/evaluate.py --reference
  # score an arbitrary model on dev (test is scored only for the final model)
  python src/evaluate.py --model path/or/hub-id --pooling cls --split dev
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
STSB_DIR = ROOT / "data" / "stsb"
RUN_LOG = ROOT / "runs" / "run_log.jsonl"

POOLINGS = ("cls", "cls_before_pooler", "pooler", "mean")

# Spearman x100 reference values (handoffs/00_INDEX.md) and the tolerance used
# for the Phase A gate.
REFERENCES = [
    {
        "run_id": "reference_bert-base-uncased_mean",
        "model": "bert-base-uncased",
        "kind": "transformers",
        "pooling": "mean",
        "expected": {"dev": 59.31, "test": 47.29},
    },
    {
        "run_id": "reference_sbert-2019_bert-base-nli-mean-tokens",
        "model": "sentence-transformers/bert-base-nli-mean-tokens",
        "kind": "sentence-transformers",
        "pooling": "mean",
        "expected": {"dev": 80.77, "test": 76.98},
    },
]
TOLERANCE = 1.0


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def _is_sentence_transformer(model):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    return isinstance(model, SentenceTransformer)


def _pool(outputs, attention_mask, pooling):
    hidden = outputs.last_hidden_state
    if pooling in ("cls", "cls_before_pooler"):
        # [CLS] token of the last layer, without BERT's pretrained pooler/MLP
        return hidden[:, 0]
    if pooling == "pooler":
        return outputs.pooler_output
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLINGS}")


@torch.no_grad()
def embed_sentences(model, tokenizer, sentences, pooling="cls", batch_size=64,
                    device=None, max_length=512):
    """Return L2-normalized embeddings, shape (N, hidden_size)."""
    device = device or default_device()
    sentences = list(sentences)

    if _is_sentence_transformer(model):
        emb = model.encode(sentences, batch_size=batch_size, device=device,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        return emb.astype(np.float32)

    if tokenizer is None:
        raise ValueError("a tokenizer is required for raw transformers models")
    was_training = model.training
    model.eval()
    model.to(device)
    chunks = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], padding=True,
                          truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        outputs = model(**batch)
        pooled = _pool(outputs, batch["attention_mask"], pooling)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        chunks.append(pooled.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0)


def load_sts(split_path):
    s1, s2, scores = [], [], []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s1.append(r["sentence1"])
            s2.append(r["sentence2"])
            scores.append(float(r["score"]))
    return s1, s2, scores


def evaluate_sts(model, tokenizer, split_path, pooling="cls", batch_size=64,
                 device=None):
    """Cosine similarity of normalized embeddings vs. gold scores (Spearman).

    Returns {"spearman": float, "similarities": list[float],
             "human_scores": list[float]}; spearman is rho in [-1, 1].
    """
    s1, s2, scores = load_sts(split_path)
    e1 = embed_sentences(model, tokenizer, s1, pooling, batch_size, device)
    e2 = embed_sentences(model, tokenizer, s2, pooling, batch_size, device)
    sims = (e1 * e2).sum(axis=1)
    rho = spearmanr(sims, scores).correlation
    return {"spearman": float(rho), "similarities": sims.tolist(),
            "human_scores": scores}


def load_model(name_or_path, kind="transformers", device=None):
    """Return (model, tokenizer); tokenizer is None for SentenceTransformers."""
    device = device or default_device()
    if kind == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name_or_path, device=device), None
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModel.from_pretrained(name_or_path).to(device).eval()
    return model, tokenizer


def hardware_info(device):
    info = {"device": device, "num_gpus": torch.cuda.device_count(),
            "precision": "fp32"}
    if device.startswith("cuda"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def append_run_log(record):
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_reference_check(seed=42, batch_size=64, device=None, log=True):
    """Phase A gate: score both reference models on dev and test."""
    device = device or default_device()
    set_seed(seed)
    all_ok = True
    for ref in REFERENCES:
        t0 = time.time()
        model, tokenizer = load_model(ref["model"], ref["kind"], device)
        got = {}
        for split in ("dev", "test"):
            res = evaluate_sts(model, tokenizer, STSB_DIR / f"{split}.jsonl",
                               ref["pooling"], batch_size, device)
            got[split] = res["spearman"]
        deltas = {s: round(got[s] * 100 - ref["expected"][s], 2) for s in got}
        ok = all(abs(d) <= TOLERANCE for d in deltas.values())
        all_ok &= ok
        print(f"{ref['run_id']}: dev {got['dev'] * 100:.2f} "
              f"(ref {ref['expected']['dev']}, {deltas['dev']:+.2f}) | "
              f"test {got['test'] * 100:.2f} "
              f"(ref {ref['expected']['test']}, {deltas['test']:+.2f}) "
              f"-> {'OK' if ok else 'FAIL'} [{time.time() - t0:.0f}s]")
        if log:
            append_run_log({
                "run_id": ref["run_id"],
                "mode": "reference",
                "timestamp": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "config": {"model_name_or_path": ref["model"],
                           "model_kind": ref["kind"],
                           "pooling": ref["pooling"],
                           "similarity": "cosine (L2-normalized)",
                           "batch_size": batch_size, "max_length": 512,
                           "eval_files": ["data/stsb/dev.jsonl",
                                          "data/stsb/test.jsonl"]},
                "seed": seed,
                "hardware": hardware_info(device),
                "results": {"dev_spearman": got["dev"],
                            "test_spearman": got["test"], "best_step": None,
                            "expected_dev_x100": ref["expected"]["dev"],
                            "expected_test_x100": ref["expected"]["test"],
                            "delta_x100": deltas,
                            "within_tolerance": ok},
                "checkpoint_path": None,
                "notes": "Phase A eval-harness sanity check; no training. "
                         f"Tolerance +/-{TOLERANCE} Spearman x100.",
            })
        del model
    return all_ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--reference", action="store_true",
                   help="run the Phase A sanity check and log it")
    p.add_argument("--no-log", action="store_true",
                   help="with --reference: do not append to runs/run_log.jsonl")
    p.add_argument("--model", help="hub id or local path")
    p.add_argument("--kind", default="transformers",
                   choices=("transformers", "sentence-transformers"))
    p.add_argument("--pooling", default="cls", choices=POOLINGS)
    p.add_argument("--split", default="dev", choices=("dev", "test"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.reference:
        ok = run_reference_check(args.seed, args.batch_size, args.device,
                                 log=not args.no_log)
        print("SANITY CHECK", "PASSED" if ok else "FAILED")
        raise SystemExit(0 if ok else 1)

    if not args.model:
        p.error("--model is required unless --reference is given")
    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.kind, args.device)
    res = evaluate_sts(model, tokenizer, STSB_DIR / f"{args.split}.jsonl",
                       args.pooling, args.batch_size, args.device)
    print(f"{args.model} [{args.pooling}] STS-B {args.split}: "
          f"Spearman x100 = {res['spearman'] * 100:.2f}")


if __name__ == "__main__":
    main()
