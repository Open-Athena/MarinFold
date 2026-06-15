# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Local recovery: regenerate the iris straggler shards with correct indices.

The at-scale iris run left a scattered set of shards unwritten (workers that
died / spilled to a far region and stalled). This regenerates exactly those
manifest shards locally (this box's cores + the cat_file fetch fix), writing
each with its ORIGINAL `contacts_v1-{idx:05d}-of-{total:05d}.parquet` name so
the GCS documents/<split>/ dir ends up complete and correctly ordered, then
uploads to GCS. Idempotent: skips shards already present on GCS.
"""
import os, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa, pyarrow.parquet as pq
import gcsfs
import generate_rows as gr

MANIFEST = os.path.expanduser("~/exp53_scratch/selection_manifest")
RR = os.path.expanduser("~/exp53_scratch/rerun")
GCS_OUT = "marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
TOTALS = {"train": 2067, "val": 22, "test": 22}
PROC = int(os.environ.get("RERUN_PROCS", "32"))

def do_shard(args):
    split, idx = args
    total = TOTALS[split]
    name = f"contacts_v1-{idx:05d}-of-{total:05d}.parquet"
    gcs_path = f"{GCS_OUT}/{split}/{name}"
    try:
        fs = gcsfs.GCSFileSystem(project="hai-gcp-models")
        if fs.exists(gcs_path):
            return (split, idx, -1, -1, "skip")
        rows = pq.read_table(f"{MANIFEST}/{split}/shard_{idx:05d}.parquet").to_pylist()
        out = list(gr.generate_shard(rows, cif_uri_column="gcs_uri", fetch_concurrency=16))
        local = f"{RR}/{split}-{name}"
        pq.write_table(pa.Table.from_pylist(out), local)
        fs.put_file(local, gcs_path)
        os.remove(local)
        return (split, idx, len(rows), len(out), "ok")
    except Exception as e:
        return (split, idx, -1, -1, f"ERR {type(e).__name__}: {str(e)[:120]}\n{traceback.format_exc()[-300:]}")

def main():
    tasks = []
    for split in ("train", "val", "test"):
        f = f"{RR}/{split}_missing.txt"
        if os.path.exists(f):
            tasks += [(split, int(x)) for x in open(f).read().split() if x.strip()]
    print(f"rerun: {len(tasks)} shards, {PROC} procs", flush=True)
    done = ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=PROC) as pool:
        futs = {pool.submit(do_shard, t): t for t in tasks}
        for fut in as_completed(futs):
            split, idx, nin, nout, status = fut.result()
            done += 1
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else:
                err += 1; print(f"  FAIL {split}/{idx}: {status}", flush=True)
            if done % 10 == 0 or done == len(tasks) or status.startswith("ERR"):
                print(f"[{done}/{len(tasks)}] ok={ok} skip={skip} err={err}  last={split}/{idx} {nout}/{nin}", flush=True)
    print(f"DONE: ok={ok} skip={skip} err={err}", flush=True)
    return 1 if err else 0

if __name__ == "__main__":
    sys.exit(main())
