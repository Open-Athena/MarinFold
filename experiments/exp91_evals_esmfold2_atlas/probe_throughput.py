# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""In-region throughput probe for the materialize (stage 3) decode.

Stage-3 decode is **S3-read-latency bound**, not CPU bound: a local probe measured
decode at ~6 ms/structure while the in-region ``ds.take`` of a scattered
``structure_blob`` costs ~200 ms/structure, so the 96 vCPUs sit ~97% idle waiting on
S3. Throughput therefore scales with the number of *concurrent in-flight takes*
(worker processes), capped by RAM (each ``take`` buffers whole Lance column pages,
~GBs) and eventually by S3 itself.

A full 20k-rep chunk takes ~40 min, so we cannot probe by completing chunks. Instead
each worker loops ``take``+decode on small sub-batches of **real sorted plan indices**
(same access pattern as the run) and bumps a shared counter; the driver runs each
``(workers, take_batch)`` config for a fixed wall-clock window and reports aggregate
structures/s plus the MemAvailable low-water mark. That directly answers: how many
workers before throughput plateaus, and how low does free RAM get.

Run on the real target box type (c7i.24xlarge / 192 GB) so the RAM floor is real.
Reads plan chunks from ``--work-dir/plan`` (the launcher syncs them from S3).
"""

import argparse
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import atlas_io
import materialize as M

_T0 = time.monotonic()


def _log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:8.1f}s] {msg}", flush=True)


def _mem_avail_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return -1.0


def _worker(indices: np.ndarray, take_batch: int, stop_at: float,
            counter, blobs_only: bool) -> None:
    """Loop take+decode of `take_batch`-sized windows until `stop_at`; count reps."""
    ds = atlas_io.open_folds()
    cols = ["structure_blob"] if blobs_only else ["protein_hash", "structure_blob", "sequence"]
    i = 0
    n = len(indices)
    while time.monotonic() < stop_at and i < n:
        sub = indices[i:i + take_batch]
        i += take_batch
        tbl = ds.take(sub.tolist(), columns=cols)
        blobs = tbl.column("structure_blob").to_pylist()
        if not blobs_only:
            hashes = tbl.column("protein_hash").to_pylist()
            for h, b in zip(hashes, blobs):
                M.decode_to_cif(b, name=h[:4])
        with counter.get_lock():
            counter.value += len(sub)
        del tbl, blobs
    # wrap around if a worker exhausts its slice before the window closes
    if i >= n and time.monotonic() < stop_at:
        with counter.get_lock():
            counter.value += 0  # slice exhausted; leave it (probe window just ends short)


def _load_indices(plan_dir: Path, n_chunks: int) -> np.ndarray:
    """Concatenate row_index from the first `n_chunks` plan chunks (sorted access)."""
    chunk_ids = sorted(int(p.stem.split("_")[1])
                       for p in plan_dir.glob("chunk_*.parquet"))[:n_chunks]
    arrs = [pq.read_table(plan_dir / f"chunk_{c:05d}.parquet",
                          columns=["row_index"]).column("row_index").to_numpy()
            for c in chunk_ids]
    return np.concatenate(arrs)


def run_config(indices: np.ndarray, workers: int, take_batch: int, seconds: int,
               blobs_only: bool) -> dict:
    """Run one (workers, take_batch) config for `seconds`; return throughput + RAM floor."""
    # Give each worker a disjoint contiguous slice so the access pattern stays sorted.
    # Spawn (not fork): Lance is not fork-safe — a forked child hangs on first take.
    ctx = mp.get_context("spawn")
    slices = np.array_split(indices, workers)
    counter = ctx.Value("q", 0)
    stop_at = time.monotonic() + seconds
    procs = [ctx.Process(target=_worker,
                         args=(slices[w], take_batch, stop_at, counter, blobs_only))
             for w in range(workers)]

    mem_floor = [float("inf")]
    stop_flag = threading.Event()

    def sample() -> None:
        while not stop_flag.is_set():
            mem_floor[0] = min(mem_floor[0], _mem_avail_gb())
            time.sleep(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    t0 = time.monotonic()
    sampler.start()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    stop_flag.set()
    elapsed = time.monotonic() - t0
    rate = counter.value / elapsed
    return {"workers": workers, "take_batch": take_batch, "structures": counter.value,
            "seconds": round(elapsed, 1), "per_sec": round(rate),
            "mem_floor_gb": round(mem_floor[0], 1),
            "eta_h_66_76M": round(66_760_000 / max(1e-9, rate) / 3600, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True, help="holds plan/")
    ap.add_argument("--seconds", type=int, default=90, help="window per config")
    ap.add_argument("--index-chunks", type=int, default=400,
                    help="plan chunks to source row indices from (8M reps / 400)")
    ap.add_argument("--workers", type=int, nargs="+",
                    default=[64, 128, 192, 256], help="worker counts to sweep")
    ap.add_argument("--take-batch", type=int, nargs="+", default=[512],
                    help="take/decode sub-batch sizes to sweep")
    ap.add_argument("--blobs-only", action="store_true",
                    help="skip decode (isolate pure S3 read throughput)")
    args = ap.parse_args()

    _log(f"loading indices from first {args.index_chunks} plan chunks…")
    indices = _load_indices(args.work_dir / "plan", args.index_chunks)
    _log(f"{len(indices):,} sorted row indices; total RAM "
         f"{_mem_avail_gb():.0f} GB avail; sweeping "
         f"workers={args.workers} x take_batch={args.take_batch} @ {args.seconds}s each")

    results = []
    for tb in args.take_batch:
        for w in args.workers:
            r = run_config(indices, w, tb, args.seconds, args.blobs_only)
            results.append(r)
            _log(f"RESULT workers={r['workers']:<4} take_batch={r['take_batch']:<5} "
                 f"-> {r['per_sec']:>5}/s  RAM_floor={r['mem_floor_gb']:>6} GB  "
                 f"full-run ETA {r['eta_h_66_76M']}h  ({r['structures']:,} in {r['seconds']}s)")

    _log("=== SWEEP COMPLETE ===")
    best = max(results, key=lambda r: r["per_sec"] if r["mem_floor_gb"] > 12 else -1)
    _log(f"best safe (RAM floor >12 GB): workers={best['workers']} "
         f"take_batch={best['take_batch']} -> {best['per_sec']}/s, "
         f"ETA {best['eta_h_66_76M']}h, RAM floor {best['mem_floor_gb']} GB")


if __name__ == "__main__":
    main()
