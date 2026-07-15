# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the exp100 regenerated (Arm B) training docs from the HF *bucket* to
region-local GCS, sharded, as a plain ``document``-column corpus.

The regenerated set is published only to the open-athena HF **bucket**
(``regenerated_documents.parquet``, 941,004 rows, 3.6 GB). HF buckets are NOT
levanter/fsspec-addressable — the tokenize step on the iris/TPU worker resolves
``open-athena/MarinFold`` as a dataset repo and 404s (the recurring exp53/85/108
gotcha). So we copy it down with the ``hf buckets`` CLI and re-write it as N
gcsfs-globbable parquet shards next to the TPU, keeping only ``entry_id`` +
``document`` (the two columns the tokenize step and Arm-A join need).

    uv run python mirror_regen_train.py            # default: 64 shards to the exp120 prefix

Needs an ``hf`` CLI with ``buckets`` support on PATH (system hf, huggingface_hub
>= 1.5; the venv hf may be too old — see exp100 publish_to_hf.find_hf).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import gcsfs
import pyarrow.parquet as pq

SRC = ("hf://buckets/open-athena/MarinFold/data/"
       "contacts-v1-train-only-correct-exp100/regenerated_documents.parquet")
DEST_DIR = ("gs://marin-us-east5/protein-structure/MarinFold/"
            "exp120_regen_vs_reepoch_contacts_v1/data/regen_train")


def find_hf() -> str:
    """An ``hf`` binary that supports ``buckets`` (skip venv-shadowed old ones)."""
    cands, seen = [], []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "hf")
        if os.path.exists(p) and p not in seen and ".venv" not in p and "/venv/" not in p:
            seen.append(p); cands.append(p)
    w = shutil.which("hf")
    if w and w not in cands:
        cands.append(w)
    for hf in cands:
        try:
            if subprocess.run([hf, "buckets", "--help"], capture_output=True).returncode == 0:
                return hf
        except OSError:
            continue
    raise RuntimeError("no `hf` with `buckets` support on PATH (need huggingface_hub >= 1.5)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST_DIR, help="GCS dir for the sharded corpus")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--local", default=None, help="local scratch path for the download")
    a = ap.parse_args()

    tmp = a.local or tempfile.mkdtemp(prefix="exp120_regen_")
    local_pq = os.path.join(tmp, "regenerated_documents.parquet")
    if not os.path.exists(local_pq):
        hf = find_hf()
        print(f"downloading {a.src} -> {local_pq}", flush=True)
        subprocess.run([hf, "buckets", "cp", a.src, local_pq], check=True)

    tbl = pq.read_table(local_pq, columns=["entry_id", "document"])
    n = tbl.num_rows
    print(f"read {n} rows; sharding to {a.dest} ({a.shards} shards)", flush=True)

    fs = gcsfs.GCSFileSystem()
    dest = a.dest.rstrip("/")
    rows_per = (n + a.shards - 1) // a.shards
    for si in range(a.shards):
        lo, hi = si * rows_per, min((si + 1) * rows_per, n)
        if lo >= hi:
            break
        shard = tbl.slice(lo, hi - lo)
        path = f"{dest}/regen_train-{si:05d}-of-{a.shards:05d}.parquet"
        with fs.open(path, "wb") as fh:
            pq.write_table(shard, fh)
        if si % 8 == 0 or si == a.shards - 1:
            print(f"  shard {si}: rows [{lo},{hi})", flush=True)
    print(f"done: {n} regenerated docs mirrored to {dest}", flush=True)
    print("NOTE: Arm B token count is (by construction) equal to Arm A's — "
          "use the total-tokens value printed by build_arm_a_aligned.py for "
          "EXP120_STEPS_PER_EPOCH.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
