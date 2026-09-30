# Part 8 — Optional extensions

Read `handoffs/00_INDEX.md` first for shared paths/conventions.

**Only start this after Parts 1–7 are complete** — the minimum required
deliverables (both models trained, dev+test evaluation, benchmark table,
report, ≥1 verified model on the Hub) do not depend on anything here. This
part exists purely to improve the grade beyond the minimum.

## Task
Anything beyond the required scope: more training data (state the source and
how it was obtained), other evaluation datasets beyond STS-B, additional
ablations beyond the two required ones, hyperparameter sweeps, etc. — your
call, no fixed list from the assignment.

The one hard rule: **declare exactly what you did**. Add a "Part 8 — optional
extensions" section to `reports/report.md` (see Part 7's report assembly)
stating:
- What extra thing was done.
- What data/tools it used, and where it came from.
- What changed in the results because of it (numbers, not just "it helped").

Any extra runs still follow the same conventions as the rest of the project:
logged in `runs/run_log.jsonl`, config saved under `runs/<run_id>/`, checkpoint
saved if a model was trained, and reused `src/evaluate.py` for any new
evaluation numbers rather than a one-off script.

## Definition of done
- If nothing is attempted here, this file requires no action — do not
  fabricate an "optional" section in the report.
- If something is attempted, it's declared in `reports/report.md` with data
  source, method, and quantified impact, and follows the standard run-logging
  conventions.
