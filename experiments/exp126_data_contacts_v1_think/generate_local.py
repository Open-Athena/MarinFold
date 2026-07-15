# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local at-scale generation of the think-augmented contacts-v1 corpus (#126).

A think (`GenerationConfig(think=True)`) twin of the exp53 contacts-v1 corpus,
generated **locally** (48-core workstation; AFDB is public) over exp53's exact
selection manifest, so it lines up 1:1 with the non-think corpus.

Shard-parallel: one process per manifest shard (`ProcessPoolExecutor`), each
shard read from a local copy of the exp53 manifest, fetched + generated via the
byte-faithful exp53 per-row worker (`generate_rows.generate_shard`, `think=True`),
and written to a local output shard named
`contacts_v1-{idx:05d}-of-{total:05d}.parquet` so the train/val/test +
round-descending layout carries through unchanged. Idempotent: shards already
present in the output dir are skipped, so the job is resumable after an
interruption.

Fetch failures / unparseable / multi-chain / out-of-range structures are
dropped (lenient), matching exp53 (#53: drop, don't backfill).

Usage::

    # Download the manifest once (see download_manifest.sh), then:
    python generate_local.py --split val   --manifest ~/exp126_scratch/manifest \
        --out ~/exp126_scratch/documents --procs 48
    python generate_local.py --split test  ...
    python generate_local.py --split train ...      # the big one

Run `--split all` to do all three in order.
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

from marinfold.document_structures.contacts_v1 import GenerationConfig

import generate_rows as gr

# Shard totals per split in the exp53 manifest (used only for the output file
# name so it matches exp53's contacts_v1-{idx}-of-{total} convention).
TOTALS = {"train": 2067, "val": 22, "test": 22}
SPLITS = ("train", "val", "test")


def _shard_paths(manifest_dir: Path, split: str) -> list[Path]:
    return sorted((manifest_dir / split).glob("shard_*.parquet"))


def _out_name(split: str, idx: int) -> str:
    return f"contacts_v1-{idx:05d}-of-{TOTALS[split]:05d}.parquet"


def _do_shard(args: tuple[str, str, str, int, bool]) -> tuple:
    """Generate one manifest shard → one output parquet. Runs in a subprocess."""
    split, manifest_path, out_dir, idx, think = args
    out_path = os.path.join(out_dir, split, _out_name(split, idx))
    try:
        if os.path.exists(out_path):
            return (split, idx, -1, -1, 0, "skip")
        rows = pq.read_table(manifest_path).to_pylist()
        cfg = GenerationConfig(think=think)
        out = list(gr.generate_shard(
            rows, cif_uri_column="gcs_uri", config=cfg, fetch_concurrency=16,
        ))
        think_total = sum(r.get("think_tokens", 0) or 0 for r in out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        pq.write_table(pa.Table.from_pylist(out), tmp)
        os.replace(tmp, out_path)  # atomic: a present shard is always complete
        return (split, idx, len(rows), len(out), think_total, "ok")
    except Exception as e:  # noqa: BLE001 — a bad shard shouldn't kill the pool
        return (split, idx, -1, -1, 0,
                f"ERR {type(e).__name__}: {str(e)[:160]}\n{traceback.format_exc()[-400:]}")


def run_split(split: str, manifest_dir: Path, out_dir: Path, procs: int,
              think: bool) -> int:
    shards = _shard_paths(manifest_dir, split)
    if not shards:
        print(f"[{split}] no manifest shards under {manifest_dir/split}", flush=True)
        return 1
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    tasks = [(split, str(p), str(out_dir), i, think) for i, p in enumerate(shards)]
    print(f"[{split}] {len(tasks)} shards, {procs} procs, think={think}", flush=True)

    t0 = time.time()
    done = ok = skip = err = 0
    rows_in = rows_out = think_sum = 0
    with ProcessPoolExecutor(max_workers=procs) as pool:
        futs = {pool.submit(_do_shard, t): t for t in tasks}
        for fut in as_completed(futs):
            sp, idx, nin, nout, think_total, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1; rows_in += nin; rows_out += nout; think_sum += think_total
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  FAIL {sp}/{idx}: {status}", flush=True)
            if done % 20 == 0 or done == len(tasks) or status.startswith("ERR"):
                dt = time.time() - t0
                rate = rows_out / dt if dt else 0
                print(f"[{split} {done}/{len(tasks)}] ok={ok} skip={skip} err={err} "
                      f"docs={rows_out} ({rate:.0f}/s) elapsed={dt/60:.1f}m", flush=True)
    dt = time.time() - t0
    mean_think = think_sum / rows_out if rows_out else 0
    print(f"[{split}] DONE ok={ok} skip={skip} err={err} | docs={rows_out}/{rows_in} "
          f"| mean_think_tokens/doc={mean_think:.1f} | {dt/60:.1f} min", flush=True)
    return 1 if err else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=(*SPLITS, "all"), required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Local dir with <split>/shard_*.parquet (exp53 manifest).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Local output dir; writes <split>/contacts_v1-*.parquet.")
    ap.add_argument("--procs", type=int, default=min(48, (os.cpu_count() or 8)))
    ap.add_argument("--no-think", action="store_true",
                    help="Disable think (for a control run); default emits think.")
    args = ap.parse_args(argv)

    splits = SPLITS if args.split == "all" else (args.split,)
    rc = 0
    for sp in splits:
        rc |= run_split(sp, args.manifest, args.out, args.procs, not args.no_think)
    return rc


if __name__ == "__main__":
    sys.exit(main())
