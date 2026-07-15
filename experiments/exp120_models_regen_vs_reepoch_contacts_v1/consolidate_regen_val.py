# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate the exp100-pipeline regen-val run into a ``document``-column corpus
for tokenization as the ``contacts-v1-val-regen`` validation component.

The constrained-decode worker writes one ``runs/val/documents/<entry_id>.json`` per
protein (the selected lowest-unmodified-NLL only-correct rollout). This reads them,
applies exp100's correctness gate (keep iff n_gt==0 OR selected prec==rec==1.0 —
i.e. no incorrect document), and writes sharded ``entry_id`` + ``document`` parquet
to ``…/data/regen_val/`` (the path ``contacts_v1_ft_common.VAL_REGEN_GLOB`` points
at). Mirrors exp100 ``publish_to_hf._read_document``.

    uv run python consolidate_regen_val.py \
        --run gs://.../exp120_.../regen_val_gen/runs/val \
        --dest gs://.../exp120_.../data/regen_val
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

PFX = ("gs://marin-us-east5/protein-structure/MarinFold/"
       "exp120_regen_vs_reepoch_contacts_v1")


def _read_document(fs, p):
    with fs.open(p, "r") as fh:
        d = json.load(fh)
    if d.get("skipped"):
        return None
    s = d["selected"]
    # correctness gate: only-correct by construction; drop any desynced rollout
    if d["n_gt"] > 0 and not (s["all_prec"] == 1.0 and s["all_rec"] == 1.0):
        return None
    return {"entry_id": d["entry_id"], "document": s["document"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{PFX}/regen_val_gen/runs/val")
    ap.add_argument("--dest", default=f"{PFX}/data/regen_val")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    dpaths = [p for p in fs.ls(f"{a.run.rstrip('/')}/documents", detail=False)
              if p.endswith(".json")]
    print(f"{len(dpaths)} document json files", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(lambda p: _read_document(fs, p), dpaths) if r is not None]
    n_dropped = len(dpaths) - len(rows)
    rows.sort(key=lambda r: r["entry_id"])
    print(f"kept {len(rows)} only-correct docs (dropped {n_dropped})", flush=True)

    dest = a.dest.rstrip("/")
    per = (len(rows) + a.shards - 1) // a.shards
    for si in range(a.shards):
        lo, hi = si * per, min((si + 1) * per, len(rows))
        if lo >= hi:
            break
        with fs.open(f"{dest}/regen_val-{si:05d}-of-{a.shards:05d}.parquet", "wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows[lo:hi]), fh)
    print(f"wrote {len(rows)} regen val docs to {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
