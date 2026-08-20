# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local at-scale generation of the contacts-and-crops-v1 corpus (#132).

A coordinate corpus (contacts-and-crops-v1, #130/PR #131) over exp53's exact
selection manifest — the **same proteins** as the contacts-v1 corpus — so it
lines up 1:1 with contacts-v1 / ccoord / think. Generated **locally**
(48-core workstation; the AFDB structures are the public
`public-datasets-deepmind-alphafold-v4` bucket).

Shard-parallel: one process per manifest shard (`ProcessPoolExecutor`), each
shard read from a local copy of the exp53 manifest, fetched + generated via
the exp105-style per-row worker (`generate_rows.generate_shard`), and written
to a local output shard named
`contacts_and_crops_v1-{idx:05d}-of-{total:05d}.parquet` so the
train/val/test + round-descending layout carries through unchanged.
Idempotent: shards already present in the output dir are skipped (atomic
write via a `.tmp` + `os.replace`), so the job is resumable after an
interruption.

Fetch failures / unparseable / multi-chain / out-of-range structures are
dropped (lenient), matching exp53 (drop, don't backfill).

Usage::

    # smoke the small splits first:
    python generate_local.py --split val  --manifest ~/exp132_scratch/manifest \
        --out ~/exp132_scratch/documents --procs 48
    python generate_local.py --split test ...
    python generate_local.py --split train ...      # the big one (~24h)
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_and_crops_v1 import GenerationConfig

import generate_rows as gr

# Shard totals per split in the exp53 manifest (used only for the output file
# name so it matches the contacts_v1-{idx}-of-{total} convention of the sibling
# corpora).
TOTALS = {"train": 2067, "val": 22, "test": 22}
SPLITS = ("train", "val", "test")
STRUCTURE_NAME = "contacts-and-crops-v1"


def _shard_paths(manifest_dir: Path, split: str) -> list[Path]:
    return sorted((manifest_dir / split).glob("shard_*.parquet"))


def _out_name(split: str, idx: int) -> str:
    return f"{STRUCTURE_NAME.replace('-', '_')}-{idx:05d}-of-{TOTALS[split]:05d}.parquet"


def _do_shard(args: tuple[str, str, str, int, int]) -> tuple:
    """Generate one manifest shard → one output parquet. Runs in a subprocess."""
    split, manifest_path, out_dir, idx, fetch_conc = args
    out_path = os.path.join(out_dir, split, _out_name(split, idx))
    try:
        if os.path.exists(out_path):
            return (split, idx, -1, -1, 0, "skip")
        rows = pq.read_table(manifest_path).to_pylist()
        out = list(gr.generate_shard(
            rows, cif_uri_column="gcs_uri", config=GenerationConfig(),
            fetch_concurrency=fetch_conc, structure_name=STRUCTURE_NAME,
        ))
        tok_total = sum(r.get("num_tokens", 0) or 0 for r in out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        pq.write_table(pa.Table.from_pylist(out), tmp, compression="zstd")
        os.replace(tmp, out_path)  # atomic: a present shard is always complete
        return (split, idx, len(rows), len(out), tok_total, "ok")
    except Exception as e:  # noqa: BLE001 — a bad shard shouldn't kill the pool
        return (split, idx, -1, -1, 0,
                f"ERR {type(e).__name__}: {str(e)[:160]}\n{traceback.format_exc()[-400:]}")


def run_split(split: str, manifest_dir: Path, out_dir: Path, procs: int,
              fetch_conc: int) -> int:
    shards = _shard_paths(manifest_dir, split)
    if not shards:
        print(f"[{split}] no manifest shards under {manifest_dir/split}", flush=True)
        return 1
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    tasks = [(split, str(p), str(out_dir), i, fetch_conc) for i, p in enumerate(shards)]
    print(f"[{split}] {len(tasks)} shards, {procs} procs, fetch_conc={fetch_conc}",
          flush=True)

    t0 = time.time()
    done = ok = skip = err = 0
    rows_in = rows_out = tok_sum = 0
    with ProcessPoolExecutor(max_workers=procs) as pool:
        futs = {pool.submit(_do_shard, t): t for t in tasks}
        for fut in as_completed(futs):
            sp, idx, nin, nout, tok_total, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1; rows_in += nin; rows_out += nout; tok_sum += tok_total
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  FAIL {sp}/{idx}: {status}", flush=True)
            if done % 10 == 0 or done == len(tasks) or status.startswith("ERR"):
                dt = time.time() - t0
                rate = rows_out / dt if dt else 0
                eta = (len(tasks) - done) / (done / dt) / 60 if done and dt else 0
                print(f"[{split} {done}/{len(tasks)}] ok={ok} skip={skip} err={err} "
                      f"docs={rows_out} ({rate:.0f}/s) elapsed={dt/60:.1f}m "
                      f"eta={eta:.0f}m", flush=True)
    dt = time.time() - t0
    mean_tok = tok_sum / rows_out if rows_out else 0
    print(f"[{split}] DONE ok={ok} skip={skip} err={err} | docs={rows_out}/{rows_in} "
          f"| mean_tokens/doc={mean_tok:.0f} | {dt/60:.1f} min", flush=True)
    return 1 if err else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=(*SPLITS, "all"), required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Local dir with <split>/shard_*.parquet (exp53 manifest).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Local output dir; writes <split>/contacts_and_crops_v1-*.parquet.")
    ap.add_argument("--procs", type=int, default=min(48, (os.cpu_count() or 8)))
    ap.add_argument("--fetch-conc", type=int, default=16,
                    help="Per-shard fetch threads overlapping pyconfind.")
    args = ap.parse_args(argv)

    # val/test first, then train — a natural smoke before the big run.
    order = ("val", "test", "train")
    splits = order if args.split == "all" else (args.split,)
    rc = 0
    for sp in splits:
        rc |= run_split(sp, args.manifest, args.out, args.procs, args.fetch_conc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
