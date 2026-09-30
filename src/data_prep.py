"""Part 2 — build SimCSE training data and STS-B evaluation files.

Outputs (see handoffs/02_part2_data_prep.md):
  data/processed/unsup_sentences.txt   unique sentences, one per line
  data/processed/sup_pairs.jsonl       {"premise", "entailment", "contradiction"|null}
  data/stsb/dev.jsonl, data/stsb/test.jsonl
  data/processed/MANIFEST.json         counts + sha256 checksums

Deterministic and re-runnable: all outputs are overwritten on every run and
ordering follows first occurrence in data/snli_train_100k.jsonl.

Notes on the rules used:
  * Unsupervised dedup is exact string match with NO whitespace normalization.
    This reproduces the expected 165,529 sentences. Stripping whitespace would
    give 165,528: the only collision is "A group of men playing soccer." vs.
    " A group of men playing soccer." (leading space). We keep the raw strings
    so the count matches the reference value; the effect on training is nil.
  * Supervised pairs: every label==0 (entailment) row, in file order. The hard
    negative is the FIRST label==2 (contradiction) hypothesis in file order for
    the exact same premise string; null if the premise has no contradiction.
  * STS-B scores are written exactly as the HF dataset provides them
    (sentence-transformers/stsb is already normalized to [0, 1]); no
    rescaling here. The test split is written to disk and never inspected.
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SNLI_PATH = ROOT / "data" / "snli_train_100k.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
STSB_DIR = ROOT / "data" / "stsb"

ENTAILMENT, CONTRADICTION = 0, 2

EXPECTED = {
    "unsup_sentence_count": 165529,
    "sup_pair_count": 33351,
    "stsb_dev_count": 1500,
    "stsb_test_count": 1379,
}


def load_snli(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_unsup(rows):
    # dict preserves insertion order -> deterministic first-occurrence ordering
    seen = {}
    for r in rows:
        for key in ("premise", "hypothesis"):
            s = r[key]
            if "\n" in s or "\r" in s:
                raise ValueError(f"sentence contains a newline: {s!r}")
            seen.setdefault(s, None)
    return list(seen)


def build_sup(rows):
    first_contradiction = {}
    for r in rows:
        if r["label"] == CONTRADICTION:
            first_contradiction.setdefault(r["premise"], r["hypothesis"])
    return [
        {
            "premise": r["premise"],
            "entailment": r["hypothesis"],
            "contradiction": first_contradiction.get(r["premise"]),
        }
        for r in rows
        if r["label"] == ENTAILMENT
    ]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = load_snli(SNLI_PATH)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    unsup = build_unsup(rows)
    unsup_path = PROCESSED_DIR / "unsup_sentences.txt"
    with open(unsup_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unsup) + "\n")

    sup = build_sup(rows)
    sup_path = PROCESSED_DIR / "sup_pairs.jsonl"
    write_jsonl(sup_path, sup)
    n_hard = sum(p["contradiction"] is not None for p in sup)

    stsb = load_dataset("sentence-transformers/stsb")
    stsb_paths = {"dev": STSB_DIR / "dev.jsonl", "test": STSB_DIR / "test.jsonl"}
    stsb_counts = {}
    for name, split in (("dev", "validation"), ("test", "test")):
        records = [
            {"sentence1": ex["sentence1"], "sentence2": ex["sentence2"], "score": float(ex["score"])}
            for ex in stsb[split]
        ]
        write_jsonl(stsb_paths[name], records)
        stsb_counts[name] = len(records)

    manifest = {
        "unsup_sentence_count": len(unsup),
        "sup_pair_count": len(sup),
        "sup_pairs_with_hard_negative": n_hard,
        "sup_pairs_with_hard_negative_frac": round(n_hard / len(sup), 6),
        "stsb_dev_count": stsb_counts["dev"],
        "stsb_test_count": stsb_counts["test"],
        "hard_negative_rule": "first label==2 hypothesis in file order for the exact same premise string",
        "unsup_dedup_rule": "exact string match, no whitespace normalization",
        "checksums_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (SNLI_PATH, unsup_path, sup_path, stsb_paths["dev"], stsb_paths["test"])
        },
    }
    with open(PROCESSED_DIR / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    mismatches = {k: (manifest[k], v) for k, v in EXPECTED.items() if manifest[k] != v}
    frac = manifest["sup_pairs_with_hard_negative_frac"]
    print(json.dumps({k: v for k, v in manifest.items() if k != "checksums_sha256"}, indent=2))
    if mismatches or not 0.27 <= frac <= 0.29:
        raise SystemExit(f"count mismatch vs expected: {mismatches}, hard-negative frac={frac}")
    print("All counts match expected values.")


if __name__ == "__main__":
    main()
