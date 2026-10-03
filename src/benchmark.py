"""Part 6 — STS-B benchmark table (dev vs. dev, test vs. test).

This is the only place the STS-B test split is scored for the two trained
models. Two steps:

  1. python src/benchmark.py --score-test
       Scores STS-B test exactly once for unsup_seed42_bs64 and
       sup_seed42_bs128 with evaluate_sts (src/evaluate.py) and writes the
       result into results.test_spearman of their existing lines in
       runs/run_log.jsonl (no new lines). Refuses to re-score a run whose
       test_spearman is already set. Raw BERT / SBERT-2019 test numbers were
       already logged by Part 4 Phase A and are not re-scored.

  2. python src/benchmark.py
       Builds reports/benchmark_table.md from runs/run_log.jsonl,
       reports/eval_<run_id>.json (Part 4 Phase B) and the paper numbers.
       Alignment/uniformity for raw BERT and SBERT-2019 are computed on STS-B
       dev with the Phase B definitions (eval_analysis.alignment_uniformity)
       and cached in reports/eval_<reference run_id>.json; test is not used.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # local evaluate.py, not HF `evaluate`
from evaluate import (REFERENCES, default_device, embed_sentences,  # noqa: E402
                      evaluate_sts, load_model, load_sts, set_seed)
from eval_analysis import POS_THRESHOLD, alignment_uniformity  # noqa: E402

RUN_LOG = ROOT / "runs" / "run_log.jsonl"
STSB_DIR = ROOT / "data" / "stsb"
REPORTS = ROOT / "reports"
TABLE = REPORTS / "benchmark_table.md"

FINAL_RUNS = [("unsup_seed42_bs64", "Our unsupervised SimCSE"),
              ("sup_seed42_bs128", "Our supervised SimCSE")]
REFERENCE_ROWS = [
    ("reference_bert-base-uncased_mean",
     "raw bert-base-uncased (mean pooling)"),
    ("reference_sbert-2019_bert-base-nli-mean-tokens",
     "SBERT-2019 (bert-base-nli-mean-tokens)"),
]
# Spearman x100. unsup: dev Table 1, test Table 5; sup: dev Table 7, test
# Table 5 (Gao, Yao & Chen, EMNLP 2021).
PAPER_ROWS = [
    ("SimCSE paper (unsup, reported)", 82.5, 76.85, "Table 1 / Table 5"),
    ("SimCSE paper (sup, reported)", 86.2, 84.25, "Table 7 / Table 5"),
]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_log():
    with open(RUN_LOG, encoding="utf-8") as f:
        return f.read().splitlines()


def runs_by_id(lines):
    out = {}
    for line in lines:
        if line.strip():
            r = json.loads(line)
            out[r["run_id"]] = r
    return out


def score_test(device, batch_size, seed):
    lines = read_log()
    runs = runs_by_id(lines)
    scores = {}
    for run_id, _ in FINAL_RUNS:
        run = runs[run_id]
        if run["results"].get("test_spearman") is not None:
            raise SystemExit(f"{run_id}: test_spearman already logged "
                             f"({run['results']['test_spearman']}); the test "
                             "split is scored exactly once, refusing.")
        cfg = json.load(open(ROOT / "runs" / run_id / "config.json",
                             encoding="utf-8"))
        set_seed(seed)
        model, tok = load_model(str(ROOT / run["checkpoint_path"]),
                                "transformers", device)
        res = evaluate_sts(model, tok, STSB_DIR / "test.jsonl",
                           cfg["eval_pooling"], batch_size, device)
        scores[run_id] = res["spearman"]
        print(f"{run_id} [{cfg['eval_pooling']}] STS-B test: "
              f"Spearman x100 = {res['spearman'] * 100:.2f}", flush=True)
        del model

    # Rewrite only the two target lines; every other line stays byte-identical.
    new_lines = []
    for line in lines:
        if line.strip():
            r = json.loads(line)
            if r["run_id"] in scores:
                r["results"]["test_spearman"] = scores[r["run_id"]]
                r["results"]["test_scored_at"] = utc_now()
                line = json.dumps(r)
        new_lines.append(line)
    RUN_LOG.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def reference_alignment_uniformity(run_id, device, batch_size, seed):
    """Dev-split alignment/uniformity for a Phase A reference model (cached)."""
    path = REPORTS / f"eval_{run_id}.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    ref = next(r for r in REFERENCES if r["run_id"] == run_id)
    set_seed(seed)
    model, tok = load_model(ref["model"], ref["kind"], device)
    dev = STSB_DIR / "dev.jsonl"
    s1, s2, gold = load_sts(dev)
    uniq = sorted(set(s1) | set(s2))
    uniq_emb = embed_sentences(model, tok, uniq, ref["pooling"], batch_size,
                               device)
    idx = {s: i for i, s in enumerate(uniq)}
    e1 = uniq_emb[[idx[s] for s in s1]]
    e2 = uniq_emb[[idx[s] for s in s2]]
    au = alignment_uniformity(e1, e2, np.asarray(gold) * 5.0, uniq_emb)
    out = {
        "run_id": run_id, "mode": "reference", "model": ref["model"],
        "eval_pooling": ref["pooling"], "split": "dev", **au,
        "alignment_definition": "mean ||f(x)-f(x+)||^2, STS-B dev pairs "
                                f"with gold >= {POS_THRESHOLD:g}/5",
        "uniformity_definition": "log mean exp(-2||f(x_i)-f(x_j)||^2), "
                                 "unique STS-B dev sentences, i<j",
        "seed": seed, "device": device, "timestamp": utc_now(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{run_id}: align {au['alignment']:.4f} | unif "
          f"{au['uniformity']:.4f}", flush=True)
    del model
    return out


def fmt(x, scale=100.0, nd=2):
    return "—" if x is None else f"{x * scale:.{nd}f}"


def build_table(device, batch_size, seed):
    runs = runs_by_id(read_log())
    rows = []
    for run_id, label in FINAL_RUNS:
        r = runs[run_id]["results"]
        if r.get("test_spearman") is None:
            raise SystemExit(f"{run_id}: test_spearman not logged yet; run "
                             "`python src/benchmark.py --score-test` first.")
        ev = json.load(open(REPORTS / f"eval_{run_id}.json",
                            encoding="utf-8"))
        rows.append((f"{label} (`{run_id}`)", fmt(r["dev_spearman"]),
                     fmt(r["test_spearman"]), fmt(ev["alignment"], 1, 3),
                     fmt(ev["uniformity"], 1, 3)))
    for run_id, label in REFERENCE_ROWS:
        r = runs[run_id]["results"]
        au = reference_alignment_uniformity(run_id, device, batch_size, seed)
        rows.append((label, fmt(r["dev_spearman"]), fmt(r["test_spearman"]),
                     fmt(au["alignment"], 1, 3), fmt(au["uniformity"], 1, 3)))
    for label, dev, test, _ in PAPER_ROWS:
        rows.append((label, f"{dev:.2f}", f"{test:.2f}", "—", "—"))

    def r100(run_id, key):
        return runs[run_id]["results"][key] * 100

    gaps = []
    for (run_id, label), (plabel, pdev, ptest, _) in zip(FINAL_RUNS,
                                                         PAPER_ROWS):
        gaps.append((label, r100(run_id, "dev_spearman"), pdev,
                     r100(run_id, "test_spearman"), ptest))

    md = [
        "# Part 6 — STS-B benchmark",
        "",
        "Spearman ρ ×100 between cosine similarity and gold scores on STS-B "
        "(`sentence-transformers/stsb`; dev = 1,500 pairs, test = 1,379 "
        "pairs), protocol of `src/evaluate.py` (no STS-B fine-tuning). "
        "Generated by `src/benchmark.py`.",
        "",
        "| Model | Dev Spearman | Test Spearman | Alignment ↓ | Uniformity ↓ |",
        "|---|---|---|---|---|",
        *[f"| {' | '.join(row)} |" for row in rows],
        "",
        "## Gap to the paper (dev vs. dev, test vs. test)",
        "",
        "| Mode | Ours dev | Paper dev | Δ dev | Ours test | Paper test "
        "| Δ test |",
        "|---|---|---|---|---|---|---|",
        *[f"| {label} | {d:.2f} | {pd:.2f} | {d - pd:+.2f} | {t:.2f} | "
          f"{pt:.2f} | {t - pt:+.2f} |" for label, d, pd, t, pt in gaps],
        "",
        "## Notes",
        "",
        "- **Dev** for our models is the best periodic dev evaluation during "
        "training, i.e. the score of the checkpoint that was kept (dev was "
        "used for checkpoint selection, so it is optimistically biased; "
        "test is not).",
        "- **Test** for our models was scored exactly once, by "
        "`python src/benchmark.py --score-test`, on that kept checkpoint; "
        "raw BERT and SBERT-2019 test numbers come from the Part 4 Phase A "
        "run (`runs/run_log.jsonl`), not re-scored here.",
        "- **Pooling:** unsup = `[CLS]` without the training MLP "
        "(`cls_before_pooler`); sup = `[CLS]` (trained without an MLP); raw "
        "BERT and SBERT-2019 = mean pooling.",
        "- **Alignment / uniformity** (Wang & Isola, 2020; lower is better "
        "for both) are computed on STS-B **dev**, L2-normalized embeddings: "
        "alignment = mean ‖f(x) − f(x⁺)‖² over the "
        f"dev pairs with gold ≥ {POS_THRESHOLD:g}/5 (n = 264); uniformity = "
        "log mean exp(−2‖f(xᵢ) − f(xⱼ)‖²) over the 2,910 unique dev "
        "sentences. Trained models: `reports/eval_<run_id>.json` (Part 4 "
        "Phase B); references: `reports/eval_reference_*.json` (this script). "
        "The paper rows have no comparable alignment/uniformity number (the "
        "paper's Fig. 2 uses a different positive set), hence —. Read the "
        "two together: raw BERT's low alignment comes with very poor "
        "uniformity (an anisotropic space where *every* pair is close), "
        "which is why it scores lowest on STS-B.",
        "- **Paper rows** (Gao, Yao & Chen, EMNLP 2021, BERT-base): unsup dev "
        "82.5 = Table 1, unsup test 76.85 = Table 5; sup dev 86.2 = Table 7, "
        "sup test 84.25 = Table 5. Never compare our dev with the paper's "
        "test (e.g. 76.92 dev vs. 76.85 test would wrongly suggest the "
        "unsupervised gap is zero).",
        "",
    ]
    TABLE.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {TABLE.relative_to(ROOT)}")
    for row in rows:
        print("  " + " | ".join(row))
    for label, d, pd, t, pt in gaps:
        print(f"  gap {label}: dev {d - pd:+.2f}, test {t - pt:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--score-test", action="store_true",
                   help="score STS-B test once for the two final runs and "
                        "update their run_log.jsonl lines")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = args.device or default_device()
    if args.score_test:
        score_test(device, args.batch_size, args.seed)
    else:
        build_table(device, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
