"""Part 4 (Phase B) — full evaluation analysis of trained checkpoints.

For every trained run in runs/run_log.jsonl (mode "unsupervised",
"supervised" or "ablation_*"), using STS-B dev and the Phase A harness in
src/evaluate.py (reused, not reimplemented):

  1. Alignment & uniformity (Wang & Isola, 2020) on L2-normalized embeddings
       alignment  = mean ||f(x) - f(x+)||^2 over dev pairs with gold >= 4/5
       uniformity = log mean exp(-2 ||f(x_i) - f(x_j)||^2) over unique dev
                    sentences, i != j
     -> reports/eval_<run_id>.json
  2. Predicted cosine similarity grouped by integer-rounded gold score (0-5)
     -> reports/figures/sim_distribution_<run_id>.png
  3. Nearest-neighbor retrievals among the unique dev sentences, plus
     human-verified failure cases (top-1 neighbor that STS-B annotators rated
     as unrelated) -> reports/retrievals_<run_id>.md

Gold scores in data/stsb/*.jsonl are normalized to [0, 1]
(sentence-transformers/stsb), so they are multiplied by 5 here to get the
usual 0-5 scale. Test is never touched (it is scored once, in Part 6).

The hand-written "## Failure discussion" section of an existing retrievals
file is preserved when the file is regenerated.

  python src/eval_analysis.py                      # every trained run
  python src/eval_analysis.py --run_id sup_seed42_bs128
  python src/eval_analysis.py --check              # Phase B definition of done
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # local evaluate.py, not HF `evaluate`
from evaluate import (default_device, embed_sentences, evaluate_sts,  # noqa: E402
                      load_model, load_sts, set_seed)

RUN_LOG = ROOT / "runs" / "run_log.jsonl"
DEV = ROOT / "data" / "stsb" / "dev.jsonl"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

POS_THRESHOLD = 4.0       # gold (0-5 scale) for alignment positives
UNRELATED_THRESHOLD = 1.0  # gold (0-5 scale) for "clearly wrong" neighbors
N_QUERIES, TOP_K, N_FAILURES = 6, 5, 3
DISCUSSION_HEADER = "## Failure discussion"
DISCUSSION_TODO = "TODO: discuss at least one failure case above."


def trained_runs():
    runs = []
    with open(RUN_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["mode"] in ("unsupervised", "supervised") or \
                    r["mode"].startswith("ablation_"):
                runs.append(r)
    return runs


def alignment_uniformity(e1, e2, gold5, uniq_emb):
    pos = gold5 >= POS_THRESHOLD
    align = float(((e1[pos] - e2[pos]) ** 2).sum(1).mean())
    # ||a - b||^2 = 2 - 2 cos(a, b) for unit vectors
    sq = np.clip(2.0 - 2.0 * (uniq_emb @ uniq_emb.T).astype(np.float64), 0,
                 None)
    iu = np.triu_indices(len(uniq_emb), k=1)
    unif = float(np.log(np.exp(-2.0 * sq[iu]).mean()))
    return {"alignment": align, "uniformity": unif,
            "n_positive_pairs": int(pos.sum()),
            "n_unique_sentences": len(uniq_emb)}


def plot_distribution(sims, gold5, run_id, spearman, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = np.rint(gold5).astype(int)
    data = [sims[buckets == b] for b in range(6)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, positions=range(6), widths=0.55, showfliers=False)
    rng = np.random.default_rng(0)
    for b, d in enumerate(data):
        ax.scatter(b + rng.uniform(-0.18, 0.18, len(d)), d, s=4, alpha=0.25,
                   color="tab:blue", linewidths=0)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{b}\n(n={len(d)})" for b, d in enumerate(data)])
    ax.set_xlabel("STS-B dev gold score (rounded to integer)")
    ax.set_ylabel("predicted cosine similarity")
    ax.set_title(f"{run_id} — STS-B dev, Spearman x100 = {spearman * 100:.2f}")
    ax.set_ylim(min(-0.1, float(sims.min()) - 0.05), 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {f"bucket_{b}": {"n": int(len(d)),
                            "median": float(np.median(d)) if len(d) else None,
                            "mean": float(d.mean()) if len(d) else None}
            for b, d in enumerate(data)}


def retrievals(s1, s2, gold5, uniq, uniq_emb, seed):
    idx = {s: i for i, s in enumerate(uniq)}
    sim = uniq_emb @ uniq_emb.T
    np.fill_diagonal(sim, -np.inf)  # never retrieve the query itself
    order = np.argsort(-sim, axis=1)
    # gold for every (unordered) STS-B dev pair, to label neighbors
    gold_of = {}
    for a, b, g in zip(s1, s2, gold5):
        gold_of[(idx[a], idx[b])] = gold_of[(idx[b], idx[a])] = g

    # Recall@1 / MRR of the gold paraphrase over every positive pair
    ranks = []
    for a, b, g in zip(s1, s2, gold5):
        if g >= POS_THRESHOLD and a != b:
            ia, ib = idx[a], idx[b]
            ranks.append(int(np.where(order[ia] == ib)[0][0]) + 1)
    ranks = np.array(ranks)
    summary = {"n_queries": int(len(ranks)),
               "recall_at_1": float((ranks == 1).mean()),
               "recall_at_5": float((ranks <= 5).mean()),
               "mrr": float((1.0 / ranks).mean())}

    def neighbors(i):
        return [{"sentence": uniq[j], "cosine": float(sim[i, j]),
                 "gold_with_query": gold_of.get((i, j))}
                for j in order[i, :TOP_K]]

    # Fixed query set (same seed -> same queries for every run): sentences
    # that have a gold >= 4 paraphrase in dev, so the expected neighbor is known.
    cand = sorted({(a, b) for a, b, g in zip(s1, s2, gold5)
                   if g >= POS_THRESHOLD and a != b})
    pick = np.random.default_rng(seed).choice(len(cand), N_QUERIES,
                                              replace=False)
    queries = []
    for k in pick:
        a, b = cand[k]
        ia = idx[a]
        queries.append({"query": a, "gold_paraphrase": b,
                        "gold_paraphrase_rank":
                            int(np.where(order[ia] == idx[b])[0][0]) + 1,
                        "neighbors": neighbors(ia)})

    # Human-verified failures: the top-1 neighbor is the query's own STS-B
    # partner, but annotators rated that pair as (nearly) unrelated.
    failures = []
    for a, b, g in zip(s1, s2, gold5):
        if g > UNRELATED_THRESHOLD or a == b:
            continue
        for q, t in ((a, b), (b, a)):
            iq = idx[q]
            if order[iq, 0] == idx[t]:
                failures.append({"query": q, "gold_top1": float(g),
                                 "neighbors": neighbors(iq)})
    failures.sort(key=lambda f: -f["neighbors"][0]["cosine"])
    summary["n_wrong_top1_unrelated"] = len(failures)
    return summary, queries, failures[:N_FAILURES]


def fmt_gold(g):
    return "—" if g is None else f"{g:.2f}"


def write_retrievals_md(path, run_id, pooling, summary, queries, failures):
    old_discussion = None
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if DISCUSSION_HEADER in text:
            old_discussion = text.split(DISCUSSION_HEADER, 1)[1].strip()

    md = [f"# Nearest-neighbor retrievals — `{run_id}`", "",
          "Corpus: the unique sentences of STS-B dev "
          f"({summary['corpus_size']}); embeddings from "
          f"`runs/{run_id}/checkpoint/` with `{pooling}` pooling, "
          "L2-normalized, cosine similarity; the query itself is excluded. "
          "\"gold\" is the STS-B human score (0–5) of the (query, neighbor) "
          "pair when that pair exists in dev, otherwise —.", "",
          "## Paraphrase retrieval over all dev positives", "",
          f"For each of the {summary['n_queries']} dev pairs with gold ≥ "
          f"{POS_THRESHOLD:g}, rank of the gold paraphrase among all other "
          "dev sentences:", "",
          "| Recall@1 | Recall@5 | MRR | wrong top-1 rated unrelated (gold ≤ "
          f"{UNRELATED_THRESHOLD:g}) |", "|---|---|---|---|",
          f"| {summary['recall_at_1']:.3f} | {summary['recall_at_5']:.3f} | "
          f"{summary['mrr']:.3f} | {summary['n_wrong_top1_unrelated']} |", "",
          f"## Sample queries (top-{TOP_K})", "",
          "Same query set for every run (fixed seed), each with a known gold "
          "paraphrase in dev (**bold** = that paraphrase).", ""]
    for n, q in enumerate(queries, 1):
        md += [f"### Q{n}. \"{q['query']}\"", "",
               f"Gold paraphrase (rank {q['gold_paraphrase_rank']}): "
               f"\"{q['gold_paraphrase']}\"", "",
               "| # | cosine | gold | neighbor |", "|---|---|---|---|"]
        for r, nb in enumerate(q["neighbors"], 1):
            s = nb["sentence"].replace("|", "\\|")
            if nb["sentence"] == q["gold_paraphrase"]:
                s = f"**{s}**"
            md.append(f"| {r} | {nb['cosine']:.3f} | "
                      f"{fmt_gold(nb['gold_with_query'])} | {s} |")
        md.append("")

    md += ["## Failure cases (human-verified)", "",
           "Queries whose top-1 neighbor is a sentence that STS-B annotators "
           f"scored ≤ {UNRELATED_THRESHOLD:g}/5 against that very query, "
           "i.e. a clearly wrong top-1, sorted by the wrong cosine.", ""]
    if not failures:
        md += ["_No top-1 neighbor in dev was rated ≤ "
               f"{UNRELATED_THRESHOLD:g}/5 by annotators for this run._", ""]
    for n, f in enumerate(failures, 1):
        md += [f"### F{n}. \"{f['query']}\"", "",
               "| # | cosine | gold | neighbor |", "|---|---|---|---|"]
        for r, nb in enumerate(f["neighbors"], 1):
            s = nb["sentence"].replace("|", "\\|")
            md.append(f"| {r} | {nb['cosine']:.3f} | "
                      f"{fmt_gold(nb['gold_with_query'])} | {s} |")
        md.append("")

    md += [DISCUSSION_HEADER, "", old_discussion or DISCUSSION_TODO, ""]
    path.write_text("\n".join(md), encoding="utf-8")


def analyze(run, device, seed, batch_size):
    run_id = run["run_id"]
    t0 = time.time()
    set_seed(seed)
    cfg = json.load(open(ROOT / "runs" / run_id / "config.json",
                         encoding="utf-8"))
    pooling = cfg["eval_pooling"]
    ckpt = ROOT / run["checkpoint_path"]
    model, tok = load_model(str(ckpt), "transformers", device)

    sts = evaluate_sts(model, tok, DEV, pooling, batch_size, device)
    s1, s2, gold = load_sts(DEV)
    gold5 = np.asarray(gold) * 5.0
    sims = np.asarray(sts["similarities"])

    uniq = sorted(set(s1) | set(s2))
    uniq_emb = embed_sentences(model, tok, uniq, pooling, batch_size, device)
    idx = {s: i for i, s in enumerate(uniq)}
    e1 = uniq_emb[[idx[s] for s in s1]]
    e2 = uniq_emb[[idx[s] for s in s2]]

    au = alignment_uniformity(e1, e2, gold5, uniq_emb)
    fig_path = FIGURES / f"sim_distribution_{run_id}.png"
    buckets = plot_distribution(sims, gold5, run_id, sts["spearman"],
                                fig_path)
    summary, queries, failures = retrievals(s1, s2, gold5, uniq, uniq_emb,
                                            seed)
    summary["corpus_size"] = len(uniq)
    ret_path = REPORTS / f"retrievals_{run_id}.md"
    write_retrievals_md(ret_path, run_id, pooling, summary, queries, failures)

    out = {
        "run_id": run_id, "mode": run["mode"],
        "checkpoint_path": run["checkpoint_path"],
        "eval_pooling": pooling, "split": "dev",
        "dev_spearman": sts["spearman"],
        "dev_spearman_logged": run["results"]["dev_spearman"],
        **au,
        "alignment_definition": "mean ||f(x)-f(x+)||^2, STS-B dev pairs "
                                f"with gold >= {POS_THRESHOLD:g}/5",
        "uniformity_definition": "log mean exp(-2||f(x_i)-f(x_j)||^2), "
                                 "unique STS-B dev sentences, i<j",
        "similarity_by_gold_bucket": buckets,
        "retrieval": summary,
        "figure": str(fig_path.relative_to(ROOT)),
        "retrievals_file": str(ret_path.relative_to(ROOT)),
        "seed": seed, "device": device,
        "timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(REPORTS / f"eval_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{run_id}: dev x100 {sts['spearman'] * 100:.2f} (logged "
          f"{run['results']['dev_spearman'] * 100:.2f}) | align "
          f"{au['alignment']:.4f} | unif {au['uniformity']:.4f} | "
          f"R@1 {summary['recall_at_1']:.3f} | failures "
          f"{summary['n_wrong_top1_unrelated']} [{time.time() - t0:.0f}s]",
          flush=True)
    del model


def check():
    """Phase B definition of done; returns True when every run is covered."""
    ok = True
    for run in trained_runs():
        rid = run["run_id"]
        problems = []
        ev = REPORTS / f"eval_{rid}.json"
        if not ev.exists():
            problems.append(f"missing {ev.relative_to(ROOT)}")
        else:
            d = json.load(open(ev, encoding="utf-8"))
            for k in ("alignment", "uniformity"):
                if not isinstance(d.get(k), float) or not np.isfinite(d[k]):
                    problems.append(f"{k} missing/non-finite in eval json")
        if not (FIGURES / f"sim_distribution_{rid}.png").exists():
            problems.append("missing similarity-distribution figure")
        rp = REPORTS / f"retrievals_{rid}.md"
        if not rp.exists():
            problems.append(f"missing {rp.relative_to(ROOT)}")
        else:
            text = rp.read_text(encoding="utf-8")
            disc = text.split(DISCUSSION_HEADER, 1)[-1] \
                if DISCUSSION_HEADER in text else ""
            if not re.search(r"### F\d", text):
                problems.append("no failure case listed")
            if not disc.strip() or DISCUSSION_TODO in disc:
                problems.append("failure case not discussed")
        print(f"[{'OK' if not problems else 'FAIL'}] {rid}"
              + ("" if not problems else ": " + "; ".join(problems)))
        ok &= not problems
    print("PHASE B DONE" if ok else "PHASE B INCOMPLETE")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run_id", action="append",
                   help="limit to these runs (repeatable)")
    p.add_argument("--check", action="store_true",
                   help="only verify the Phase B definition of done")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    device = args.device or default_device()
    runs = [r for r in trained_runs()
            if not args.run_id or r["run_id"] in args.run_id]
    if args.run_id and len(runs) != len(set(args.run_id)):
        p.error(f"unknown run_id among {args.run_id}")
    for run in runs:
        analyze(run, device, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
