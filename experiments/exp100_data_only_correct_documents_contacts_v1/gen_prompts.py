# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B-prep (local, needs marinfold): materialize the resampled contacts-v1
prefixes the TPU rollout worker generates from.

Per target we build ``--k`` independent contacts-v1 document realizations
(``build_document(f"{entry_id}:r{r}", residues, [])`` — fresh N-terminus start +
statement order each, the format's nuisance symmetries; exp82's settled
*rollout + resample* recipe). We pre-build them here so the TPU worker can stay
**marinfold-free** and run in the marin checkout's vllm/tpu env (mirrors exp89's
``gen_ensemble_prompts.py`` + thin worker split).

Output: one ``prompts/<entry_id>.parquet`` per target (rows ``r, L, prefix,
seq_positions``), where ``prefix`` is the token string up to and including
``<begin_statements>`` and ``seq_positions[i]`` is the position index of sequence
index ``i`` (so the worker maps generated ``<pX>`` back to a residue).

    uv run python gen_prompts.py --targets data/targets.parquet -k 10 \
        --out gs://marin-us-east5/protein-structure/MarinFold/exp100_only_correct_contacts_v1_train/prompts
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

BEGIN = "<begin_statements>"
NUM_POS = 2000


def build_prompts(entry_id: str, sequence: str, k: int) -> list[dict]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="data/targets.parquet")
    ap.add_argument("-k", "--n-rollouts", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N targets (calibration)")
    ap.add_argument("--out", required=True, help="prompts dir (local or gs://)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=32,
                    help="thread pool size for build+write (GCS write is I/O-bound)")
    args = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    is_gcs = args.out.startswith("gs://")
    if not is_gcs:
        os.makedirs(args.out, exist_ok=True)

    with fsspec.open(args.targets, "rb") as fh:
        targets = pq.read_table(fh).to_pylist()

    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} targets x {args.n_rollouts} rollouts -> {args.out}", flush=True)

    out = args.out.rstrip("/")

    def process(t):
        dest = f"{out}/{t['entry_id']}.parquet"
        if not args.overwrite:
            exists = fs.exists(dest) if is_gcs else os.path.exists(dest)
            if exists:
                return False
        rows = build_prompts(t["entry_id"], t["sequence"], args.n_rollouts)
        table = pa.Table.from_pylist(rows)
        if is_gcs:
            with fs.open(dest, "wb") as fh:
                pq.write_table(table, fh)
        else:
            pq.write_table(table, dest)
        return True

    written = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, t): t for t in targets}
        for n, fut in enumerate(as_completed(futs)):
            if fut.result():
                written += 1
            else:
                skipped += 1
            if (n + 1) % 50 == 0 or n == 0:
                print(f"  {n+1}/{len(targets)} ({written} written, {skipped} skipped)", flush=True)
    print(f"done: {written} written, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
