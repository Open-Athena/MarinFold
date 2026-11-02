# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 prompt generation — S3-native port of exp98's ``gen_prompts.py``.

Materializes the resampled contacts-v1 prefixes the rollout worker generates
from. Per target we build ``-k`` independent document realizations
(``build_document(f"{entry_id}:r{r}", residues, [])`` — a fresh N-terminus start
and statement order each time, the format's nuisance symmetries; exp82's settled
*rollout + resample* recipe), so the worker itself stays marinfold-free.

Difference from exp98's version: **fsspec/S3, not gcsfs.** exp98 hard-imported
``gcsfs``; cw-rno2a pods have no GCS at all, so this uses plain fsspec with the
virtual-hosted addressing CoreWeave requires.

.. warning::
   Still **one object per target**, like exp98. That is fine here (554 eval
   proteins) but does NOT scale: a 1M-protein run would create ~1M S3 objects.
   Fixing it means aggregating prompts into per-shard parquets *and* teaching the
   rollout worker's ``{prompts}/{entry}.parquet`` reader the packed layout — both
   sides have to change together, so it is deliberately left for the 1M push
   rather than half-done here.

Needs ``marinfold`` on the path::

    PYTHONPATH=<repo>/marinfold uv run --no-sync python gen_prompts_exp163.py \\
        --targets targets_eval.parquet -k 24 \\
        --out s3://marin-us-east-02a/MarinFold/exp163/eval554/prompts
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

BEGIN = "<begin_statements>"
NUM_POS = 2000

# CoreWeave AI Object Storage rejects path-style S3 requests.
S3_CONFIG_KWARGS = {"s3": {"addressing_style": "virtual"}}


def _fs_for(url: str):
    if not url.startswith("s3://"):
        return fsspec.filesystem("file")
    return fsspec.filesystem(
        "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"), config_kwargs=S3_CONFIG_KWARGS
    )


def _strip(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


def build_prompts(entry_id: str, sequence: str, k: int) -> list[dict]:
    """``k`` independent contacts-v1 prefixes for one target."""
    residues = residues_from_sequence(sequence)
    rows = []
    for r in range(k):
        doc = build_document(f"{entry_id}:r{r}", residues, [], config=GenerationConfig())
        if doc is None:
            raise RuntimeError(f"build_document returned None for {entry_id} (L={len(sequence)})")
        L = doc.seq_len
        prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
        seq_positions = [(doc.n_term_index + i) % NUM_POS for i in range(L)]
        rows.append(dict(r=r, L=L, prefix=prefix, seq_positions=seq_positions))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("-k", "--n-rollouts", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, help="prompts dir (local or s3://)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    fs = _fs_for(a.out)
    out = _strip(a.out).rstrip("/")
    if not a.out.startswith("s3://"):
        os.makedirs(out, exist_ok=True)

    with fsspec.open(a.targets, "rb") as fh:
        targets = pq.read_table(fh).to_pylist()
    if a.limit:
        targets = targets[: a.limit]
    print(f"[exp163] {len(targets)} targets x {a.n_rollouts} prompts -> {a.out}", flush=True)

    def process(t):
        dest = f"{out}/{t['entry_id']}.parquet"
        if not a.overwrite and fs.exists(dest):
            return False
        rows = build_prompts(t["entry_id"], t["sequence"], a.n_rollouts)
        with fs.open(dest, "wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows), fh)
        return True

    written = skipped = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, t) for t in targets]
        for n, fut in enumerate(as_completed(futs)):
            if fut.result():
                written += 1
            else:
                skipped += 1
            if (n + 1) % 100 == 0 or n + 1 == len(targets):
                print(f"  {n+1}/{len(targets)} ({written} written, {skipped} skipped)", flush=True)
    print(f"[exp163] done: {written} written, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
