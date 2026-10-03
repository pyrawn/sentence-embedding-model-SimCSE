"""Part 7 — export a trained SimCSE checkpoint as a SentenceTransformer, push
it to the Hugging Face Hub, and verify the published copy.

Pipeline (run per model, in order):
  # 1. build the SentenceTransformer in model_cards/<name>/ and check that its
  #    embeddings equal the raw-checkpoint embeddings evaluate_sts scored
  python src/publish.py export --run-id unsup_seed42_bs64 --name simcse-bert-base-snli-unsup
  # 2. upload model_cards/<name>/ exactly as it is on disk
  python src/publish.py push --name simcse-bert-base-snli-unsup --repo pyrawn/simcse-bert-base-snli-unsup
  # 3. reload from the Hub into an empty, throwaway cache and re-score STS-B test
  python src/publish.py verify --run-id unsup_seed42_bs64 --repo pyrawn/simcse-bert-base-snli-unsup
  # 4. record the repo id on the run's line in runs/run_log.jsonl
  python src/publish.py record --run-id unsup_seed42_bs64 --repo pyrawn/simcse-bert-base-snli-unsup

Export matches the evaluate_sts protocol used for the recorded test number
(src/benchmark.py --score-test): last-layer [CLS] hidden state with no pooler /
MLP on top (both "cls" and "cls_before_pooler" in src/evaluate.py), L2
normalization, cosine similarity, max_length 512.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RUN_LOG = ROOT / "runs" / "run_log.jsonl"
STSB_DIR = ROOT / "data" / "stsb"
CARDS_DIR = ROOT / "model_cards"

# eval_pooling in a run's config.json -> sentence-transformers Pooling mode.
# evaluate.py pools both as hidden[:, 0] (the [CLS] token of the last layer).
POOLING_MODES = {"cls": "cls", "cls_before_pooler": "cls", "mean": "mean"}
# evaluate.py tokenizes with truncation at 512 tokens.
EVAL_MAX_LENGTH = 512
# Spearman (in [-1, 1]) tolerance for the reload check: 1e-4 = 0.01 x100.
TOLERANCE = 1e-4


def read_run(run_id):
    for line in RUN_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line)["run_id"] == run_id:
            return json.loads(line)
    raise SystemExit(f"run {run_id!r} not found in {RUN_LOG}")


def run_config(run_id):
    return json.loads((ROOT / "runs" / run_id / "config.json").read_text())


def build_sentence_transformer(checkpoint, eval_pooling):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import (
        Normalize, Pooling, Transformer)

    if eval_pooling not in POOLING_MODES:
        raise SystemExit(f"no Pooling equivalent for eval_pooling={eval_pooling!r}")
    transformer = Transformer(str(checkpoint), max_seq_length=EVAL_MAX_LENGTH)
    pooling = Pooling(transformer.get_embedding_dimension(),
                      pooling_mode=POOLING_MODES[eval_pooling])
    return SentenceTransformer(modules=[transformer, pooling, Normalize()],
                               device="cpu")


def cmd_export(args):
    import numpy as np
    from evaluate import embed_sentences, load_model, load_sts

    run = read_run(args.run_id)
    cfg = run_config(args.run_id)
    checkpoint = ROOT / run["checkpoint_path"]
    out = CARDS_DIR / args.name

    model = build_sentence_transformer(checkpoint, cfg["eval_pooling"])
    # Keep a README already written for this model; otherwise let
    # sentence-transformers write its placeholder card (replaced in Task 3).
    readme = out / "README.md"
    saved_readme = readme.read_text(encoding="utf-8") if readme.exists() else None
    if out.exists():
        for p in out.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    model.save(str(out), model_name=args.name)
    if saved_readme is not None:
        readme.write_text(saved_readme, encoding="utf-8")
    print(model)

    # The exported model must produce the same embeddings as the raw checkpoint
    # under evaluate.py's pooling (checked on dev sentences; test is scored only
    # in `verify`).
    s1, s2, _ = load_sts(STSB_DIR / "dev.jsonl")
    sentences = list(dict.fromkeys(s1 + s2))
    raw, tok = load_model(str(checkpoint), "transformers", "cpu")
    ref = embed_sentences(raw, tok, sentences, cfg["eval_pooling"], 64, "cpu")
    del raw
    from sentence_transformers import SentenceTransformer

    reloaded = SentenceTransformer(str(out), device="cpu")
    got = embed_sentences(reloaded, None, sentences, batch_size=64, device="cpu")
    diff = float(np.abs(ref - got).max())
    print(f"max |raw checkpoint - exported| over {len(sentences)} dev "
          f"sentences: {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("exported model does not match the raw checkpoint")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  {p.stat().st_size:,} B")


def cmd_push(args):
    from sentence_transformers import SentenceTransformer

    folder = CARDS_DIR / args.name
    model = SentenceTransformer(str(folder), device="cpu")
    url = model.push_to_hub(args.repo, local_model_path=str(folder),
                            exist_ok=True, commit_message=args.message)
    print(f"pushed {folder} -> https://huggingface.co/{args.repo}\n{url}")


def cmd_verify(args):
    # Point every Hugging Face cache at an empty temp dir *before* importing
    # anything that reads HF_HOME, so the model can only come from the Hub.
    cache = Path(tempfile.mkdtemp(prefix="hf_verify_"))
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache / "st")
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer
        from evaluate import evaluate_sts, set_seed

        assert not any(cache.rglob("*.safetensors")), "cache not empty"
        info = HfApi().model_info(args.repo)
        print(f"Hub repo {args.repo} @ {info.sha}")
        set_seed(42)
        model = SentenceTransformer(args.repo, device="cpu",
                                    cache_folder=str(cache / "hub"))
        downloaded = [p for p in cache.rglob("model.safetensors")]
        print(f"downloaded into fresh cache: {[str(p) for p in downloaded]}")
        if not downloaded:
            raise SystemExit("weights were not downloaded into the fresh cache")
        res = evaluate_sts(model, None, STSB_DIR / "test.jsonl",
                           batch_size=64, device="cpu")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    recorded = read_run(args.run_id)["results"]["test_spearman"]
    got = res["spearman"]
    delta = got - recorded
    ok = abs(delta) <= TOLERANCE
    print(f"{args.repo} STS-B test Spearman: reloaded {got:.10f} | recorded "
          f"{recorded:.10f} | delta {delta:+.2e} -> {'MATCH' if ok else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)


def cmd_record(args):
    lines = RUN_LOG.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = 0
    for i, line in enumerate(lines):
        if line.strip() and json.loads(line)["run_id"] == args.run_id:
            rec = json.loads(line)
            rec["hub_repo_id"] = args.repo
            lines[i] = json.dumps(rec) + "\n"
            hits += 1
    if hits != 1:
        raise SystemExit(f"expected exactly one line for {args.run_id}, got {hits}")
    RUN_LOG.write_text("".join(lines), encoding="utf-8")
    print(f"{args.run_id}: hub_repo_id = {args.repo}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run-id", required=True)
    e.add_argument("--name", required=True)
    u = sub.add_parser("push")
    u.add_argument("--name", required=True)
    u.add_argument("--repo", required=True)
    u.add_argument("--message", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--run-id", required=True)
    v.add_argument("--repo", required=True)
    r = sub.add_parser("record")
    r.add_argument("--run-id", required=True)
    r.add_argument("--repo", required=True)
    args = p.parse_args()
    {"export": cmd_export, "push": cmd_push, "verify": cmd_verify,
     "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
