# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B (local, needs marinfold): materialize the resampled contacts-v1
prefixes the GPU rollout worker generates from.

Copied from exp98 (local-only; no gcsfs). Per target we build ``-k`` independent
contacts-v1 document realizations (``build_document(f"{entry_id}:r{r}", residues,
[])`` — fresh N-terminus start + statement order each, the format's nuisance
symmetries; exp82's settled *rollout + resample* recipe). Pre-building here keeps
the worker able to run without re-deriving documents.

Output: one ``prompts/<entry_id>.parquet`` per target (rows ``r, L, prefix,
seq_positions``), where ``prefix`` is the token string up to and including
``<begin_statements>`` and ``seq_positions[i]`` is the position index of sequence
index ``i`` (so the worker maps generated ``<pX>`` back to a residue).

    uv run python gen_prompts.py --targets data/targets.parquet -k 1000 --out data/prompts
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ap.add_argument("-k", "--n-rollouts", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="only the first N targets")
    ap.add_argument("--out", default="data/prompts", help="prompts dir (local)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    targets = pq.read_table(args.targets).to_pylist()
    if args.limit:
        targets = targets[: args.limit]
    out = args.out.rstrip("/")
    print(f"{len(targets)} targets x {args.n_rollouts} rollouts -> {out}", flush=True)

    def process(t):
        dest = f"{out}/{t['entry_id']}.parquet"
        if not args.overwrite and os.path.exists(dest):
            return False
        rows = build_prompts(t["entry_id"], t["sequence"], args.n_rollouts)
        pq.write_table(pa.Table.from_pylist(rows), dest)
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
