# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Local generation / resume for the exp105 train split (off-cluster).

The at-scale iris grind is CPU-capacity-bound on marin (~22 worker slots). This
generates train shards on the workstation's cores instead — the AFDB `gcs_uri`
bucket (`public-datasets-deepmind-alphafold-v4`) is *publicly readable*, so
`marinfold.document_structures.io.read_object_bytes` fetches fine locally
(no requester-pays billing needed; force-setting a billing project is what 403s).

Each manifest shard `shard_{idx:05d}.parquet` → one output
`ccoord_v1-{idx:05d}-of-{total:05d}.parquet` written with its ORIGINAL index, so
the GCS `documents/train/` dir stays complete + round-descending regardless of
who (cluster or local) produced which shard. **Idempotent**: skips shards whose
output already exists on GCS, so it composes with the concurrent iris job and is
safely restartable.

Usage:
    RERUN_PROCS=32 uv run python local_generate_train.py            # all missing train shards
    uv run python local_generate_train.py --shards 5,17,42          # specific indices
    uv run python local_generate_train.py --limit 1                 # smoke: 1 missing shard
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

import generate_rows as gr

MANIFEST = "marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/train"
OUT = "marin-us-central1/protein-structure/MarinFold/exp105_ccoord_v1/documents/train"
TOTAL = 2067
PROCS = int(os.environ.get("RERUN_PROCS", "32"))


def _fs() -> gcsfs.GCSFileSystem:
    # No billing project: the AFDB read path is public, and the marin-* buckets
    # are ours. Forcing project=/requester_pays= is what triggers the 403.
    return gcsfs.GCSFileSystem()


def out_name(idx: int) -> str:
    return f"ccoord_v1-{idx:05d}-of-{TOTAL:05d}.parquet"


def do_shard(idx: int):
    name = out_name(idx)
    gcs_out = f"{OUT}/{name}"
    try:
        fs = _fs()
        if fs.exists(gcs_out):
            return (idx, -1, -1, "skip")
        rows = pq.read_table(f"gs://{MANIFEST}/shard_{idx:05d}.parquet").to_pylist()
        out = list(gr.generate_shard(rows, None, cif_uri_column="gcs_uri", fetch_concurrency=16))
        # write to a temp object then it's atomic-ish; gcsfs write is a single PUT.
        with fs.open(gcs_out, "wb") as f:
            pq.write_table(pa.Table.from_pylist(out), f, compression="zstd")
        return (idx, len(rows), len(out), "ok")
    except Exception as e:  # noqa: BLE001 — report + continue; a failed shard is retried next run
        return (idx, -1, -1, f"ERR {type(e).__name__}: {str(e)[:150]}\n{traceback.format_exc()[-400:]}")


def missing_shards() -> list[int]:
    fs = _fs()
    present = {p.rsplit("/", 1)[-1] for p in fs.glob(f"{OUT}/*.parquet")}
    return [i for i in range(TOTAL) if out_name(i) not in present]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default=None, help="Comma-separated indices; default = all missing.")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of shards this run (smoke).")
    ap.add_argument("--reverse", action="store_true",
                    help="Process high indices first (descending). Use when a "
                         "concurrent ascending iris job is grinding the same "
                         "train/ dir, so the two sweeps meet in the middle.")
    args = ap.parse_args(argv)

    if args.shards:
        todo = [int(x) for x in args.shards.split(",") if x.strip()]
    else:
        todo = missing_shards()
    if args.reverse:
        todo = sorted(todo, reverse=True)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"local train gen: {len(todo)} shard(s) to do, {PROCS} procs (spawn)", flush=True)
    if not todo:
        print("nothing to do — train/ is complete.", flush=True)
        return 0

    # `spawn` (not fork): fresh interpreter per child so the numba/pyconfind/gemmi
    # C-extension state isn't inherited across fork (that segfaults children ->
    # BrokenProcessPool). On a broken pool, re-scan missing shards and retry — the
    # skip-existing check makes this resume cleanly and drops any single bad shard
    # after a few rounds.
    ctx = mp.get_context("spawn")
    ok = skip = err = 0
    remaining = list(todo)
    broken_rounds = 0
    while remaining and broken_rounds < 8:
        batch = remaining
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=PROCS, mp_context=ctx) as pool:
                futs = {pool.submit(do_shard, i): i for i in batch}
                for fut in as_completed(futs):
                    idx, nin, nout, status = fut.result()
                    done += 1
                    if status == "ok":
                        ok += 1
                    elif status == "skip":
                        skip += 1
                    else:
                        err += 1
                        print(f"  FAIL shard {idx}: {status}", flush=True)
                    if done % 10 == 0 or status.startswith("ERR"):
                        print(f"[+{done}/{len(batch)}] ok={ok} skip={skip} err={err} last=shard{idx} {nout}/{nin}", flush=True)
        except BrokenProcessPool as exc:
            broken_rounds += 1
            print(f"  BrokenProcessPool (round {broken_rounds}): {exc}; re-scanning missing + retrying", flush=True)
            time.sleep(5)
        # Re-scan GCS for what's still missing (covers both crashed in-flight and
        # shards finished by the concurrent iris job).
        remaining = sorted(missing_shards(), reverse=args.reverse)
        print(f"  remaining missing: {len(remaining)}", flush=True)
    print(f"DONE: ok={ok} skip={skip} err={err}; remaining_missing={len(remaining)} broken_rounds={broken_rounds}", flush=True)
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
