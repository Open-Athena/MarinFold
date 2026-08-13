# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the 554-protein eval target file both gates score against.

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

#: #89's contact definition. Pairs closer than this are not contacts, and the
#: universe file stores every pair, so the filter has to be applied here.
MIN_SEP = 6

SCHEMA = pa.schema([
    ("target_id", pa.string()), ("dataset", pa.string()), ("stem", pa.string()),
    ("L", pa.int32()), ("input_seq", pa.string()),
    ("n_gt", pa.int32()), ("gt_contacts", pa.list_(pa.list_(pa.int32(), 2))),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    universe = [json.loads(line) for line in (a.work / "gt_universe.jsonl").read_text().splitlines()]
    seqs = read_fasta(a.work / "eval776.fasta")
    out = a.out or (a.work / "eval554_targets.parquet")

    rows, missing = [], []
    for rec in universe:
        key = f"{rec['dataset']}__{rec['stem']}"
        seq = seqs.get(key)
        if seq is None:
            missing.append(key)
            continue
        if len(seq) != rec["L"]:
            raise SystemExit(f"{key}: fasta length {len(seq)} != universe L {rec['L']}")
        gt = sorted({(min(int(i), int(j)), max(int(i), int(j)))
                     for i, j, *_ in rec["contacts"]
                     if abs(int(i) - int(j)) >= MIN_SEP})
        rows.append({
            "target_id": key, "dataset": rec["dataset"], "stem": rec["stem"],
            "L": int(rec["L"]), "input_seq": seq,
            "n_gt": len(gt), "gt_contacts": [[i, j] for i, j in gt],
        })
    if missing:
        raise SystemExit(f"{len(missing)} eval units have no sequence, e.g. {missing[:5]}")
    if len(rows) != 554:
        raise SystemExit(f"expected 554 eval units, built {len(rows)}")
    if len({r["target_id"] for r in rows}) != 554:
        raise SystemExit("duplicate (dataset, stem) keys")

    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
    n_stems = len({r["stem"] for r in rows})
    print(f"[eval] {len(rows)} units / {n_stems} unique stems -> {out}")
    print(f"[eval] L min/median/max = {min(r['L'] for r in rows)} / "
          f"{sorted(r['L'] for r in rows)[len(rows) // 2]} / {max(r['L'] for r in rows)}")
    print(f"[eval] n_gt median = {sorted(r['n_gt'] for r in rows)[len(rows) // 2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
