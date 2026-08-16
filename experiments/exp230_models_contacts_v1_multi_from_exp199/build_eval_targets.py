# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the 577-unit eval target file every mode scores against.

Joins two published artifacts, neither of which is sufficient alone:

* ``data/contacts-v1-model-eval-exp89/gt_universe.jsonl`` -- the fixed #89
  benchmark: 554 ``(dataset, stem)`` units, each with ``L``, the resolved
  residue indices and the ground-truth contacts.  It carries **no sequence**.
* #226's ``eval_queries_expanded.fasta`` -- the 776-query set, keyed
  ``{dataset}__{stem}``, verified here to be a strict superset of #225's 554
  with byte-identical sequences.

Emits the column set exp82's ``score_rollout_worker.py`` consumes
(``dataset``, ``stem``, ``L``, ``input_seq``) plus the ground-truth contacts, so
one file drives Gate A (plain-mode R-precision), Gate B (the plain-mode section
count) and the multi-mode report.

Count units on **(dataset, stem)**, never stem alone: ``7ur7_A`` / ``8ah9_A``
recur across datasets with *different* sequences (#226).

    uv run python build_eval_targets.py --work /data/exp230_multi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: #89's contact definition, and BOTH halves of it matter.  The universe file
#: stores every pair with its contact *degree*, so both filters have to be
#: applied here or the ground truth is wrong.
#:
#: ``MIN_DEG`` is the one that is easy to miss: **21.7 %** of the universe's
#: separation>=6 pairs have a degree below it (the minimum is 1.2e-12).  Keeping
#: them inflates ``n_gt`` by ~22 %, and since R-precision cuts the ranking at
#: R = n_true, an inflated R makes every number here incomparable with exp199's
#: published 0.611 / 0.545 / 0.337 and with #180's frontier table.  exp82's
#: ``true_matrix`` applies ``d >= MIN_DEG and (j - i) >= MIN_SEP``; so does this.
MIN_SEP = 6
MIN_DEG = 0.001

SCHEMA = pa.schema([
    ("target_id", pa.string()), ("dataset", pa.string()), ("stem", pa.string()),
    ("L", pa.int32()), ("input_seq", pa.string()),
    ("n_gt", pa.int32()), ("gt_contacts", pa.list_(pa.list_(pa.int32(), 2))),
    # Cut membership, so ONE scoring run yields every reported slice.
    ("in_legacy554", pa.bool_()), ("in_eval2", pa.bool_()),
    ("designed_any", pa.bool_()), ("passes_30", pa.bool_()),
    ("best_identity", pa.float32()),
])


def read_fasta(path: Path) -> dict[str, str]:
    out, key = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            key = line[1:]
            out[key] = ""
        elif key is not None:
            out[key] += line
    return out


def load_cuts(work: Path) -> tuple[set, dict]:
    """Legacy-554 membership, and #226's identity annotation for the eval2 307."""
    legacy = set()
    for line in (work / "gt_universe.jsonl").read_text().splitlines():
        r = json.loads(line)
        legacy.add((r["dataset"], r["stem"]))

    import csv
    ann = {}
    with (work / "eval2_manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            ann[(row["dataset"], row["stem"])] = {
                "designed_any": str(row.get("designed_any", "")).lower() in ("1", "true", "yes"),
                "passes_30": str(row.get("passes_30", "")).lower() in ("1", "true", "yes"),
                "best_identity": float(row["best_identity"]) if row.get("best_identity") else float("nan"),
            }
    return legacy, ann


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--universe", default="gt_universe_eval2.jsonl")
    ap.add_argument("--fasta", default="eval776.fasta")
    ap.add_argument("--expect", type=int, default=577)
    a = ap.parse_args()

    universe = [json.loads(line) for line in (a.work / a.universe).read_text().splitlines()]
    seqs = read_fasta(a.work / a.fasta)
    legacy, ann = load_cuts(a.work)
    out = a.out or (a.work / f"eval{a.expect}_targets.parquet")

    rows, missing, dropped_deg = [], [], 0
    for rec in universe:
        key = f"{rec['dataset']}__{rec['stem']}"
        seq = seqs.get(key)
        if seq is None:
            missing.append(key)
            continue
        if len(seq) != rec["L"]:
            raise SystemExit(f"{key}: fasta length {len(seq)} != universe L {rec['L']}")
        gt = set()
        for i, j, *rest in rec["contacts"]:
            i, j = int(i), int(j)
            if abs(i - j) < MIN_SEP:
                continue
            deg = float(rest[0]) if rest else 1.0
            if deg < MIN_DEG:                      # exp82/exp89's true_matrix filter
                dropped_deg += 1
                continue
            gt.add((min(i, j), max(i, j)))
        gt = sorted(gt)
        k = (rec["dataset"], rec["stem"])
        meta = ann.get(k)
        rows.append({
            "target_id": key, "dataset": rec["dataset"], "stem": rec["stem"],
            "L": int(rec["L"]), "input_seq": seq,
            "n_gt": len(gt), "gt_contacts": [[i, j] for i, j in gt],
            "in_legacy554": k in legacy,
            "in_eval2": meta is not None,
            "designed_any": bool(meta["designed_any"]) if meta else False,
            "passes_30": bool(meta["passes_30"]) if meta else False,
            "best_identity": float(meta["best_identity"]) if meta else float("nan"),
        })
    if missing:
        raise SystemExit(f"{len(missing)} eval units have no sequence, e.g. {missing[:5]}")
    if len(rows) != a.expect:
        raise SystemExit(f"expected {a.expect} eval units, built {len(rows)}")
    if len({r["target_id"] for r in rows}) != a.expect:
        raise SystemExit("duplicate (dataset, stem) keys")

    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
    n_stems = len({r["stem"] for r in rows})
    n_legacy = sum(r["in_legacy554"] for r in rows)
    n_e2 = sum(r["in_eval2"] for r in rows)
    n_nat = sum(r["in_eval2"] and not r["designed_any"] for r in rows)
    n_30 = sum(r["in_eval2"] and r["passes_30"] for r in rows)
    print(f"[eval] {len(rows)} units / {n_stems} unique stems -> {out}")
    print(f"[eval] dropped {dropped_deg:,} pairs below MIN_DEG {MIN_DEG} "
          f"({100 * dropped_deg / (dropped_deg + sum(r['n_gt'] for r in rows)):.1f}% of sep>=6 pairs)")
    print(f"[eval] cuts: legacy554 {n_legacy} | eval2 {n_e2} | eval2-natural {n_nat} | eval2 <30% {n_30}")
    print(f"[eval] L min/median/max = {min(r['L'] for r in rows)} / "
          f"{sorted(r['L'] for r in rows)[len(rows) // 2]} / {max(r['L'] for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
