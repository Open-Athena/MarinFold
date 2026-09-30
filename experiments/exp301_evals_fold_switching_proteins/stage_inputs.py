#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn the Phase-0 universe into the worker's targets parquet, and optionally upload it.

Reads ``data/foldswitch_universe.jsonl`` + ``data/premise_gate.csv``, keeps the
pairs that pass the premise gate, and writes the flat table the scoring worker
reads. ``fs_lo``/``fs_hi`` are -1 when the fold-switching region was not located
(the gate rejects those, so it should not happen — the encoding exists so the
worker never has to handle a null).

Also folds in ``--calibration N`` eval-val proteins as ``role="calibration"``
rows, so the scoring run gates itself against exp277's published per-protein
numbers instead of asserting its own correctness.

**Run this with ``--with "huggingface_hub>=1.5"``** — HF *buckets* need hub 1.x,
which cannot go in the experiment venv (marinfold pins transformers<5, and every
transformers 4.x pins huggingface-hub<1.0). The conflict is real but harmless:
nothing needs marinfold and the bucket in one process.

    uv run --with "huggingface_hub>=1.5" python stage_inputs.py
    uv run --with "huggingface_hub>=1.5" python stage_inputs.py \\
        --upload s3://marin-us-east-02a/MarinFold/exp301/eval_targets.parquet
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: exp245's eval-val ground truth, and exp277's published per-protein scores on
#: it. A handful of those proteins ride along in the targets file as
#: ``role="calibration"`` rows — same worker, same recipe, an empty fold2 — so
#: the run gates itself against numbers that already exist instead of asserting
#: its own correctness. The comparison is paired per protein, which is both
#: cheaper and tighter than matching a pooled mean.
EXP245_BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245"
EXP277_PER_PROTEIN = (
    HERE.parent / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2/figure_per_protein.csv"
)
MIN_DEGREE = 0.001
MIN_SEP = 6

SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("role", pa.string()),
    ("fold1", pa.string()),
    ("fold2", pa.string()),
    ("tier", pa.string()),
    ("seq_class", pa.string()),
    ("sequence", pa.string()),
    ("L", pa.int32()),
    ("contacts_fold1", pa.list_(pa.list_(pa.int32(), 2))),
    ("contacts_fold2", pa.list_(pa.list_(pa.int32(), 2))),
    ("common_positions", pa.list_(pa.int32())),
    ("fs_lo", pa.int32()),
    ("fs_hi", pa.int32()),
])


def calibration_rows(n: int) -> list[dict]:
    """``n`` eval-val proteins, spread over the length range, as calibration rows.

    Ground truth comes from exp245's published universe under the same contact
    definition used everywhere else (degree >= 0.001, |i-j| >= 6, both endpoints
    resolved). ``contacts_fold2`` is empty, so the worker's ``recall_A`` on these
    rows is ordinary R-precision territory and can be compared, per protein,
    against exp277's published score.
    """
    import io

    import fsspec

    published = {
        row["stem"]
        for row in csv.DictReader(EXP277_PER_PROTEIN.open())
        if row["predictor"] == "MarinFold exp277" and row["subset"] == "eval-val"
    }
    with fsspec.open(f"{EXP245_BUCKET}/eval_targets_foldbench_monomers.parquet", "rb") as fh:
        targets = {r["stem"]: r for r in pq.read_table(fh).to_pylist()}
    with fsspec.open(f"{EXP245_BUCKET}/gt_universe_scored.jsonl", "rb") as fh:
        universe = {r["stem"]: r for r in map(json.loads, io.TextIOWrapper(fh))}

    usable = sorted(published & targets.keys() & universe.keys(),
                    key=lambda s: targets[s]["L"])
    if not usable:
        raise ValueError("no eval-val protein is present in all three sources")
    # Even spread over the length range rather than the short tail.
    picks = [usable[round(k * (len(usable) - 1) / max(n - 1, 1))] for k in range(min(n, len(usable)))]

    rows = []
    for stem in dict.fromkeys(picks):
        gt, target = universe[stem], targets[stem]
        resolved = {int(p) for p in gt["resolved"]}
        contacts = sorted(
            [int(i), int(j)]
            for i, j, degree in gt["contacts"]
            if degree >= MIN_DEGREE and abs(int(i) - int(j)) >= MIN_SEP
            and int(i) in resolved and int(j) in resolved
        )
        rows.append({
            "pair_id": f"cal_{stem}", "role": "calibration",
            "fold1": stem, "fold2": "", "tier": "cal", "seq_class": "calibration",
            "sequence": target["input_seq"], "L": int(target["L"]),
            "contacts_fold1": contacts, "contacts_fold2": [],
            "common_positions": sorted(resolved), "fs_lo": -1, "fs_hi": -1,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upload", default=None, metavar="URI", help="also write to this fsspec URI")
    ap.add_argument("--all", action="store_true", help="include pairs that fail the premise gate")
    ap.add_argument("--calibration", type=int, default=20,
                    help="eval-val proteins to ride along as a published-number gate (0 to omit)")
    args = ap.parse_args()

    records = [json.loads(line) for line in (DATA / "foldswitch_universe.jsonl").open()]
    passing = {
        row["pair_id"]
        for row in csv.DictReader((DATA / "premise_gate.csv").open())
        if row["passes_gate"] == "True"
    }
    keep = records if args.all else [r for r in records if r["pair_id"] in passing]
    keep.sort(key=lambda r: r["L"])

    rows = {name: [] for name in SCHEMA.names}
    for r in keep:
        region = r["fs_region"] or [-1, -1]
        rows["pair_id"].append(r["pair_id"])
        rows["role"].append("foldswitch")
        rows["fold1"].append(r["fold1"])
        rows["fold2"].append(r["fold2"])
        rows["tier"].append(r["tier"])
        rows["seq_class"].append(r["seq_class"])
        rows["sequence"].append(r["sequence"])
        rows["L"].append(r["L"])
        rows["contacts_fold1"].append(r["contacts_fold1"])
        rows["contacts_fold2"].append(r["contacts_fold2"])
        rows["common_positions"].append(r["common_positions"])
        rows["fs_lo"].append(region[0])
        rows["fs_hi"].append(region[1])

    n_foldswitch = len(rows["pair_id"])
    for extra in (calibration_rows(args.calibration) if args.calibration else []):
        for name in SCHEMA.names:
            rows[name].append(extra[name])

    table = pa.table(rows, schema=SCHEMA)
    local = DATA / "eval_targets.parquet"
    pq.write_table(table, local, compression="zstd")
    print(f"wrote {local} — {table.num_rows} rows "
          f"({n_foldswitch} fold-switch + {table.num_rows - n_foldswitch} calibration), "
          f"{local.stat().st_size / 2**10:.0f} KiB")
    print(f"  L: {min(rows['L'])}-{max(rows['L'])}")
    print(f"  tiers: {dict(sorted({t: rows['tier'].count(t) for t in set(rows['tier'])}.items()))}")

    if args.upload:
        import fsspec

        with fsspec.open(args.upload, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        print(f"uploaded to {args.upload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
