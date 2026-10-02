"""Part 3 — SimCSE training on bert-base-uncased.

Modes:
  * unsup (paper §3, Eq. 1): every sentence is encoded twice in the same
    forward batch; the two dropout-noised encodings are the positive pair and
    the other sentences' second views are the in-batch negatives.
  * sup   (paper §4, Eq. 5): (premise, entailment) is the positive pair; the
    other rows' entailments plus every in-batch contradiction (hard negative,
    present for ~28% of rows) are the negatives.

Every hyperparameter lives in the JSON config (configs/{unsupervised,supervised}.json);
CLI flags only override paths and debug limits. The best checkpoint by STS-B
dev Spearman (src/evaluate.py) is kept under <output_dir>/checkpoint/, and on
completion <output_dir>/ holds config.json, results.json and run_record.json
(one record in the runs/run_log.jsonl schema of handoffs/00_INDEX.md; it is
NOT appended to run_log.jsonl here — that happens locally after import).

  python src/train_simcse.py --mode unsup --config configs/unsupervised.json \
      --output_dir runs/unsup_seed42_bs64
  python src/train_simcse.py --mode sup --config configs/supervised.json \
      --output_dir runs/sup_seed42_bs128
"""

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # local evaluate.py, not HF `evaluate`
from evaluate import evaluate_sts, set_seed  # noqa: E402

MODES = {"unsup": "unsupervised", "sup": "supervised"}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SimCSEModel(nn.Module):
    """BERT encoder + optional MLP over [CLS] used only during training.

    Paper (§3, Appendix A): unsup SimCSE trains with an MLP on top of [CLS]
    and evaluates with the plain [CLS] representation ("cls_before_pooler").
    The MLP is discarded at save time; only the encoder is checkpointed.
    """

    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        hf_cfg = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            hidden_dropout_prob=cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=cfg["attention_probs_dropout_prob"])
        self.encoder = AutoModel.from_pretrained(cfg["model_name_or_path"],
                                                 config=hf_cfg)
        self.pooling = cfg["pooling"]
        hidden = hf_cfg.hidden_size
        self.mlp = (nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
                    if cfg["mlp_only_train"] else None)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            raise ValueError(f"unsupported training pooling {self.pooling!r}")
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        return pooled


def unsup_contrastive_loss(z1, z2, temperature):
    """Eq. 1: -log exp(sim(h_i,h_i+)/t) / sum_j exp(sim(h_i,h_j+)/t).

    z1, z2: (B, H) — two dropout views of the same B sentences. Cosine
    similarity of L2-normalized embeddings; the positive for row i is
    column i, every other column is an in-batch negative.
    """
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def load_unsup_sentences(path, limit=None):
    with open(path, encoding="utf-8") as f:
        sents = [line.rstrip("\n") for line in f if line.strip()]
    return sents[:limit] if limit else sents


def unsup_step(model, tokenizer, batch_sents, cfg, device):
    enc = tokenizer(batch_sents, padding=True, truncation=True,
                    max_length=cfg["max_seq_length"],
                    return_tensors="pt").to(device)
    if cfg["same_dropout_mask"]:
        # Part 5 ablation: both "views" share one dropout mask, i.e. a single
        # forward pass reused as its own positive.
        z = model(**enc)
        z1, z2 = z, z
    else:
        # Duplicate each sentence in one forward pass: training-mode dropout
        # samples an independent mask for every row, so rows i and B+i are
        # two different noisy views of the same sentence.
        bsz = enc["input_ids"].size(0)
        doubled = {k: torch.cat([v, v], dim=0) for k, v in enc.items()}
        z = model(**doubled)
        z1, z2 = z[:bsz], z[bsz:]
    return unsup_contrastive_loss(z1, z2, cfg["temperature"])


def sup_contrastive_loss(z1, z2, z3, temperature):
    """Eq. 5: -log exp(sim(h_i,h_i+)/t) /
                    sum_j [exp(sim(h_i,h_j+)/t) + exp(sim(h_i,h_j-)/t)].

    z1: (B, H) premises, z2: (B, H) entailments, z3: (M, H) contradictions for
    the M <= B rows that have one (M may be 0). The positive for row i is
    column i of z1 @ z2.T; every other entailment and every hard negative in
    the batch (columns B..B+M-1) is a negative.
    """
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    logits = z1 @ z2.T
    if z3 is not None and z3.size(0) > 0:
        z3 = F.normalize(z3.float(), dim=-1)
        logits = torch.cat([logits, z1 @ z3.T], dim=1)
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits / temperature, labels)


def load_sup_pairs(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def sup_step(model, tokenizer, batch_rows, cfg, device):
    premises = [r["premise"] for r in batch_rows]
    entailments = [r["entailment"] for r in batch_rows]
    # Part 5 ablation sets use_hard_negatives=false -> denominator is only the
    # in-batch entailments.
    hard_negs = ([r["contradiction"] for r in batch_rows
                  if r.get("contradiction")]
                 if cfg["use_hard_negatives"] else [])
    # One forward pass over premises + entailments + hard negatives.
    enc = tokenizer(premises + entailments + hard_negs, padding=True,
                    truncation=True, max_length=cfg["max_seq_length"],
                    return_tensors="pt").to(device)
    z = model(**enc)
    bsz = len(batch_rows)
    z1, z2, z3 = z[:bsz], z[bsz:2 * bsz], z[2 * bsz:]
    return sup_contrastive_loss(z1, z2, z3, cfg["temperature"])


STEPS = {"unsup": (load_unsup_sentences, unsup_step),
         "sup": (load_sup_pairs, sup_step)}


def evaluate_dev(model, tokenizer, cfg, dev_path, device):
    res = evaluate_sts(model.encoder, tokenizer, dev_path,
                       pooling=cfg["eval_pooling"],
                       batch_size=cfg["eval_batch_size"], device=device)
    model.train()
    return res["spearman"]


def save_checkpoint(model, tokenizer, ckpt_dir):
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True)
    model.encoder.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def hardware_record(device, use_fp16, label):
    info = {"device": label or device,
            "num_gpus": 1 if device.startswith("cuda") else 0,
            "num_gpus_available": torch.cuda.device_count(),
            "precision": "fp16 (amp)" if use_fp16 else "fp32"}
    if device.startswith("cuda"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def library_versions():
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__}


def train(mode, cfg, args, out_dir, device):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    load_data, step_fn = STEPS[mode]
    sentences = load_data(resolve(cfg["train_file"]),
                          args.max_train_sentences)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    model = SimCSEModel(cfg).to(device)
    model.train()

    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(sentences, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True, generator=generator,
                        collate_fn=list)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg["num_epochs"]
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if total_steps == 0:
        raise SystemExit("no training steps: fewer sentences than batch_size")

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    use_fp16 = cfg["fp16"] and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    dev_path = resolve(args.dev_file or cfg["dev_file"])
    eval_steps = args.eval_steps or cfg["eval_steps"]
    ckpt_dir = out_dir / "checkpoint"
    history, loss_log = [], []
    best = {"dev_spearman": -math.inf, "step": None}
    notes = []

    def maybe_eval(step):
        rho = evaluate_dev(model, tokenizer, cfg, dev_path, device)
        history.append({"step": step, "dev_spearman": rho})
        improved = rho > best["dev_spearman"]
        if improved:
            best.update(dev_spearman=rho, step=step)
            save_checkpoint(model, tokenizer, ckpt_dir)
        print(f"[eval] step {step}/{total_steps} dev spearman x100 = "
              f"{rho * 100:.2f}{'  (best, saved)' if improved else ''}",
              flush=True)

    print(f"{len(sentences)} sentences, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps, device={device}, fp16={use_fp16}",
          flush=True)
    step, t0, running = 0, time.time(), []
    done = False
    for epoch in range(cfg["num_epochs"]):
        for batch_sents in loader:
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=use_fp16):
                loss = step_fn(model, tokenizer, batch_sents, cfg, device)
            if not torch.isfinite(loss):
                notes.append(f"non-finite loss at step {step + 1}; stopped")
                done = True
                break
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["max_grad_norm"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running.append(loss.item())

            if step % cfg["log_steps"] == 0 or step == total_steps:
                avg = sum(running) / len(running)
                loss_log.append({"step": step, "epoch": epoch, "loss": avg,
                                 "lr": scheduler.get_last_lr()[0]})
                running = []
                print(f"step {step}/{total_steps} loss {avg:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            if step % eval_steps == 0 or step == total_steps:
                maybe_eval(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if history and history[-1]["step"] != step and step > 0:
        maybe_eval(step)
    if loss_log and loss_log[-1]["loss"] < 1e-3:
        notes.append("final training loss < 1e-3: possible collapse "
                     "(identical views?)")
    return {"best": best, "history": history, "loss_log": loss_log,
            "final_step": step, "train_sentences": len(sentences),
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "train_seconds": round(time.time() - t0, 1), "fp16": use_fp16,
            "notes": notes}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", required=True, choices=tuple(MODES))
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None,
                   help="default: runs/<run_id from config>")
    p.add_argument("--run_id", default=None, help="override config run_id")
    p.add_argument("--device", default=None)
    p.add_argument("--hardware", default=None,
                   help='label recorded in run_record, e.g. "Kaggle T4 x2"')
    p.add_argument("--notes", default="", help="free text for run_record")
    # debug / smoke-test limits (recorded in config.json when used)
    p.add_argument("--max_train_sentences", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--dev_file", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()

    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = json.load(f)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    run_id = args.run_id or cfg["run_id"]
    cfg["run_id"] = run_id
    debug = {k: getattr(args, k) for k in
             ("max_train_sentences", "max_steps", "eval_steps", "dev_file",
              "batch_size") if getattr(args, k)}
    if debug:
        cfg["debug_overrides"] = debug

    out_dir = resolve(args.output_dir or f"runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    started = utc_now()
    result = train(args.mode, cfg, args, out_dir, device)

    best = result["best"]
    cfg_out = dict(cfg, library_versions=library_versions(),
                   started=started)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)
    results = {"dev_spearman": best["dev_spearman"], "test_spearman": None,
               "best_step": best["step"], **{k: v for k, v in result.items()
                                             if k not in ("best", "notes")}}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    notes = "; ".join(filter(None, [args.notes] + result["notes"]))
    record = {
        "run_id": run_id,
        "mode": MODES[args.mode],
        "timestamp": utc_now(),
        "config": cfg,
        "seed": cfg["seed"],
        "hardware": hardware_record(device, result["fp16"], args.hardware),
        "results": {"dev_spearman": best["dev_spearman"],
                    "test_spearman": None, "best_step": best["step"]},
        "checkpoint_path": f"runs/{run_id}/checkpoint/",
        "notes": notes,
    }
    with open(out_dir / "run_record.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    print(f"done: best dev spearman x100 = {best['dev_spearman'] * 100:.2f} "
          f"at step {best['step']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
