# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize selected Atlas reps into contacts-v1-ready parquet shards (stage 3).

All data here comes from the **ESM Atlas** (``folds_1B.lance``); "afdb-24M" never
enters as a data source. ``pipeline.py`` produces ``selected_manifest.csv`` — one
row per chosen cluster representative (``protein_hash`` + metadata), but **no
structures**. This is the final step: it pulls those rows' ``structure_blob`` from
the Atlas, decodes each to mmCIF text, and writes parquet in the column layout
``contacts-v1 generate`` reads (``entry_id`` + ``cif_content``; the generator's
defaults, see ``marinfold/document_structures/contacts_v1/parse.py``).

Two stages, both resumable and heavily logged (the lessons from the funnel runs):

1. ``plan`` — locate each manifest rep's *absolute Atlas row index* with one cheap
   pass over the small ``protein_hash`` column (never the multi-TB ``structure_blob``),
   then sort by row index and split into ``plan/chunk_NNNNN.parquet`` files of
   ``--chunk-size`` reps each (rep metadata joined in). ~66.76M reps -> ~3.3k chunks.

2. ``run`` — decode. Worker processes pull chunks; each does a Lance ``take`` of only
   its reps' ``structure_blob`` (so we read ~1.65 TB, not the whole ~27 TB blob
   column), decodes to mmCIF, and writes one ``parts/part_NNNNN.parquet``. **A part
   file IS the checkpoint**: a relaunch skips any chunk whose part already exists
   (locally or in the ``--skip-list`` the launcher builds from S3), so a crash costs
   at most the in-flight chunks.

``full`` runs ``plan`` (unless a plan is already present) then ``run``.

Smoke test: ``--limit N`` caps the plan to the first N reps it locates and stops the
scan early, so ``full --limit 500`` finishes in seconds end to end.

Output parquet columns (per rep). Names align with timodonnell/afdb-24M where we have
the field (entry_id, cif_content, seq_len, global_plddt, seq_cluster_id, split) so the
two datasets union cleanly; the rest are Atlas-specific extras:
  entry_id        protein_hash (afdb-24M id column)
  cif_content     decoded mmCIF text (per-residue pLDDT in the B-factor column)
  seq_len         residue count (afdb-24M)
  global_plddt    mean pLDDT 0-1 (afdb-24M name; renamed from the Atlas mean_plddt)
  seq_cluster_id  our linclust @40%-id cluster (afdb-24M name; renamed from cluster_id)
  split           constant "train" (afdb-24M name)
  sequence        residue sequence (from the decoded structure) [extra]
  seq_ok          bool: decoded sequence == Atlas ``sequence`` column (integrity) [extra]
  ptm, plddt_std, cluster_size   from the manifest [extra]
  source          constant "esm-atlas-v1" provenance tag [extra]

Publishing the result to HuggingFace is a **separate, sign-off-gated** step
(cross-cloud, >10 GB, CC BY-SA) — this script never does it.
"""

import argparse
import ctypes
import ctypes.util
import json
import multiprocessing as mp
import os
import shutil
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import atlas_io

# Match the contacts-v1 generator's column defaults so the parquet feeds it unchanged.
ID_COLUMN = "entry_id"
CIF_COLUMN = "cif_content"
SOURCE_TAG = "esm-atlas-v1"
_CARRY_COLS = ["seq_len", "mean_plddt", "ptm", "plddt_std", "cluster_id",
               "cluster_size"]
# Rename carried manifest columns to the timodonnell/afdb-24M column names so the two
# datasets union cleanly (entry_id/cif_content/seq_len already match). Our clustering is
# by sequence (linclust @ 40% id), so cluster_id maps to afdb's seq_cluster_id. The
# afdb-only columns (uniprot_accession, tax_id, organism_name, struct_cluster_id,
# gcs_uri) are meaningless for metagenomic Atlas predictions and are not fabricated;
# our extra columns (sequence, seq_ok, ptm, plddt_std, cluster_size, source) are kept.
_AFDB_RENAME = {"mean_plddt": "global_plddt", "cluster_id": "seq_cluster_id"}
_SPLIT_TAG = "train"  # afdb-24M 'split' column; all distilled reps are training data

# Decode sub-batch size (stage run). Each worker holds at most this many decoded
# mmCIFs at once, so peak RAM = _TAKE_BATCH x ~250 KB x workers, independent of
# --chunk-size. Keeps a 64-worker pool well under the 192 GB box (the whole-chunk
# design held ~3 GB/worker and OOM-killed it instantly on pool startup).
_TAKE_BATCH = 2000

# ---------------------------------------------------------------------------
# Logging (mirrors pipeline.py: timestamped, flushed, with resource snapshots,
# so the S3-shipped live log always shows where a long run actually is).
# ---------------------------------------------------------------------------
_T0 = time.monotonic()


def _log(msg: str) -> None:
    # Wall-clock UTC prefix so log lines correlate with S3 object mtimes and
    # CloudWatch (CPU/NetworkIn) when diagnosing a wedge; the monotonic offset is
    # per-process (workers reset it on spawn) so it alone can't be lined up.
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]"
          f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=False)


def _release_memory() -> None:
    """Return freed memory to the OS after each decode sub-batch.

    Two allocators creep independently over a chunk's sub-batches: Arrow's memory
    pool retains freed ``take()`` buffers (the fat ``structure_blob`` reads), and
    glibc retains the decoded-CIF ``str`` arenas. Releasing both each sub-batch keeps
    per-worker RSS flat, so 64+ synchronized workers don't OOM-cascade the 192 GB box
    partway through the first chunk. ``malloc_trim`` is glibc-only; guard it.
    """
    pa.default_memory_pool().release_unused()
    try:
        _LIBC.malloc_trim(0)
    except (AttributeError, OSError):
        pass  # non-glibc libc: Arrow release alone still bounds the bulk of the growth


# A single ds.take that exceeds this is logged even without --log-io: it is the
# early-warning signal for the S3-connection wedge (a hung take never returns, so a
# "take START" with no matching completion pins the exact worker + sub-batch).
SLOW_TAKE_WARN_S = 30.0


def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return -1.0


def _resources(work: Path) -> str:
    du = shutil.disk_usage(work)
    return (f"disk {du.used / 1e12:.2f}/{du.total / 1e12:.2f} TB used, "
            f"{du.free / 1e12:.2f} TB free; RAM avail {_mem_avail_gb():.0f} GB")


def decode_to_cif(blob: bytes, name: str) -> tuple[str, str]:
    """Decode one ``structure_blob`` to ``(mmCIF text, sequence)``.

    The mmCIF carries per-residue pLDDT in the B-factor column.
    """
    struct = atlas_io.decode_structure_blob(blob)
    doc = atlas_io.atom37_to_structure(struct, name=name).make_mmcif_document()
    return doc.as_string(), struct["sequence"]


# ===========================================================================
# Stage 1: plan — locate reps' Atlas row indices, split into chunk files.
# ===========================================================================
# The membership set is a *sorted uint64 array* of each rep hash's high 64 bits.
# Lance is not fork-safe, so the worker pools use "spawn"; the array is therefore
# published in a single POSIX shared-memory block that every worker attaches
# read-only (a Python set of 66.76M strings pickled to each spawned worker would
# blow up both transfer and RAM). Truncation to 64 bits can only produce false
# *positives* (never misses a real rep), and those are dropped by the exact join
# on the full protein_hash after the scan. ``_HKEYS`` is used only by the
# in-process smoke path; spawned workers attach the shared block by name.
_HKEYS: np.ndarray | None = None


def _keys_for(hashes) -> np.ndarray:
    """High-64-bit uint64 key for each hex protein_hash."""
    return np.fromiter((int(h[:16], 16) for h in hashes), dtype=np.uint64,
                       count=len(hashes))


def _plan_scan_range(spec: dict) -> tuple[int, int, str]:
    """Scan one Atlas row range's ``protein_hash`` and emit rows that are reps.

    Writes ``emits_<tag>.parquet`` with columns ``row_index`` (absolute) +
    ``protein_hash``. Returns ``(n_seen, n_hit, path)``. ``stop_after`` (smoke)
    stops the worker once it has emitted that many reps.
    """
    work = Path(spec["work"])
    tag = spec["tag"]
    start, stop = spec["start"], spec["stop"]
    batch_rows = spec["batch_rows"]
    stop_after = spec.get("stop_after")

    # Attach the shared membership array (spawned workers), or fall back to the
    # module global (the in-process smoke path passes no shm).
    shm = None
    if spec.get("shm_name"):
        shm = shared_memory.SharedMemory(name=spec["shm_name"])
        hkeys = np.ndarray((spec["shm_len"],), dtype=np.uint64, buffer=shm.buf)
    else:
        hkeys = _HKEYS

    ds = atlas_io.open_folds()
    scanner = ds.scanner(columns=["protein_hash"], offset=start,
                         limit=stop - start, batch_size=batch_rows,
                         batch_readahead=4, fragment_readahead=2)
    out_rows: list[dict] = []
    n_seen = n_hit = 0
    row = start
    for batch in scanner.to_batches():
        hashes = batch.column("protein_hash").to_pylist()
        keys = _keys_for(hashes)
        idx = np.searchsorted(hkeys, keys)
        idx = np.clip(idx, 0, len(hkeys) - 1)
        hit = hkeys[idx] == keys
        capped = False
        for j in np.nonzero(hit)[0]:
            out_rows.append({"row_index": row + int(j), "protein_hash": hashes[j]})
            n_hit += 1
            if stop_after is not None and n_hit >= stop_after:
                capped = True  # stop precisely at `limit`, mid-batch (smoke)
                break
        n_seen += len(hashes)
        row += len(hashes)
        if n_seen % (batch_rows * 20) < batch_rows:
            _log(f"[plan {tag}] seen {n_seen:,} located {n_hit:,}; {_resources(work)}")
        if capped:
            _log(f"[plan {tag}] hit stop_after={stop_after:,} at row ~{row:,}")
            break

    if shm is not None:
        shm.close()
    path = work / f"emits_{tag}.parquet"
    schema = pa.schema([("row_index", pa.int64()), ("protein_hash", pa.string())])
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), path)
    _log(f"[plan {tag}] done: located {n_hit:,}/{n_seen:,} -> {path}")
    return n_seen, n_hit, str(path)


def stage_plan(manifest: Path, work: Path, *, scan_workers: int, chunk_size: int,
               batch_rows: int, limit: int | None, scan_rows: int | None = None) -> None:
    """Build ``plan/chunk_NNNNN.parquet`` for every manifest rep (resumable)."""
    global _HKEYS
    plan_dir = work / "plan"
    done_marker = plan_dir / "_plan_done.json"
    if done_marker.exists():
        meta = json.loads(done_marker.read_text())
        _log(f"[plan] already complete: {meta['n_reps']:,} reps in "
             f"{meta['n_chunks']:,} chunks; skipping")
        return
    plan_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[plan] loading manifest hashes from {manifest}")
    sel = pd.read_csv(manifest, dtype={"protein_hash": str, "cluster_id": str})
    if "protein_hash" not in sel.columns:
        raise SystemExit(f"{manifest} has no 'protein_hash' column")
    # NB: --limit does NOT subset the manifest here. The reps are scattered across
    # the 1.1B Atlas by row position, so wanting a *specific* handful (e.g. head(N))
    # would force a scan of nearly the whole dataset to locate them. Instead we keep
    # the full membership set and stop the scan after locating N reps (below), which
    # grabs the first N *encountered* in row order — dense (~6%), so ~N/0.06 rows.
    hashes = sel["protein_hash"].astype(str).to_numpy()
    _HKEYS = np.sort(_keys_for(hashes))
    _log(f"[plan] {len(hashes):,} reps wanted; membership array "
         f"{_HKEYS.nbytes / 1e6:.0f} MB; {_resources(work)}")

    ds = atlas_io.open_folds()
    total = ds.count_rows()
    del ds
    if scan_rows is not None:  # testing: bound the plan scan to the first N rows
        total = min(total, scan_rows)

    # Smoke (--limit): single worker that stops as soon as it has located `limit`
    # reps, so we never scan the whole 1.1B just to grab a handful.
    if limit is not None:
        specs = [{"work": str(work), "tag": "0", "start": 0, "stop": total,
                  "batch_rows": batch_rows, "stop_after": limit}]
        emits = [_plan_scan_range(specs[0])]
    else:
        # Publish the membership array in shared memory so spawned (fork-unsafe
        # Lance) workers attach it read-only instead of each pickling a copy.
        shm = shared_memory.SharedMemory(create=True, size=_HKEYS.nbytes)
        shared = np.ndarray(_HKEYS.shape, dtype=_HKEYS.dtype, buffer=shm.buf)
        shared[:] = _HKEYS[:]
        step = total // scan_workers
        specs = []
        for w in range(scan_workers):
            s = w * step
            e = total if w == scan_workers - 1 else s + step
            specs.append({"work": str(work), "tag": str(w), "start": s, "stop": e,
                          "batch_rows": batch_rows, "stop_after": None,
                          "shm_name": shm.name, "shm_len": int(_HKEYS.shape[0])})
        _log(f"[plan] scanning {total:,} rows across {scan_workers} spawn workers")
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(scan_workers) as pool:
                emits = pool.map(_plan_scan_range, specs)
        finally:
            shm.close()
            shm.unlink()

    n_located = sum(e[1] for e in emits)
    _log(f"[plan] located {n_located:,} reps; joining metadata + chunking")

    # Concatenate emits, join the FULL protein_hash against the manifest (drops any
    # 64-bit-truncation false positives), sort by row_index so each chunk is a
    # contiguous Atlas range (efficient take), and split into chunk files.
    emit_tbl = pa.concat_tables([pq.read_table(e[2]) for e in emits])
    located = emit_tbl.to_pandas().drop_duplicates("protein_hash")
    # Join the located reps' FULL protein_hash against the manifest (already loaded
    # as `sel`), which also drops any 64-bit-truncation false positives.
    plan = located.merge(sel, on="protein_hash", how="inner").sort_values("row_index")
    keep = ["row_index", "protein_hash"] + [c for c in _CARRY_COLS if c in plan.columns]
    plan = plan[keep].reset_index(drop=True)

    n_chunks = (len(plan) + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        part = plan.iloc[c * chunk_size:(c + 1) * chunk_size]
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False),
                       plan_dir / f"chunk_{c:05d}.parquet")
    for e in emits:  # emits were scratch; the chunk files supersede them
        Path(e[2]).unlink(missing_ok=True)
    done_marker.write_text(json.dumps({"n_reps": int(len(plan)), "n_chunks": n_chunks,
                                       "chunk_size": chunk_size}))
    _log(f"[plan] DONE: {len(plan):,} reps -> {n_chunks:,} chunks in {plan_dir}")


# ===========================================================================
# Stage 2: run — decode each chunk's structures, write a checkpointed part.
# ===========================================================================
_DS = None  # per-worker Lance handle (set in the pool initializer)


def _worker_init() -> None:
    global _DS
    _DS = atlas_io.open_folds()


def _materialize_chunk(spec: dict) -> tuple[int, int, int, float]:
    """Decode one plan chunk to ``parts/part_NNNNN.parquet``. Idempotent.

    Returns ``(chunk_id, n_written, n_seq_mismatch, seconds)``. If the part already
    exists the chunk is skipped (this is the resume mechanism).

    Memory is bounded to one ``_TAKE_BATCH`` of reps at a time: each sub-batch is
    fetched (``take``), decoded, and streamed into the part via a ``ParquetWriter``
    row group, then freed. Peak worker RAM is therefore ``_TAKE_BATCH`` decoded
    mmCIFs (~hundreds of MB), **independent of ``--chunk-size``** — the earlier
    "hold the whole 20k-rep chunk of decoded CIFs" design multiplied ~3 GB by the
    worker count and instantly OOM-killed the 192 GB box the moment the 64-worker
    pool started (analogous to the scan's uncapped-readahead wedge).

    Integrity check (always on, ~free): the Atlas ``sequence`` column — a field
    independent of ``structure_blob`` — is fetched alongside the blob and compared
    against the blob-decoded sequence. (The mmCIF's residues are named straight
    from the decoded sequence, so re-parsing the structure would only re-derive it;
    the separate ``sequence`` column is the meaningful cross-check.) A mismatch does
    **not** abort — it is recorded in a per-row ``seq_ok`` bool (downstream can
    filter) and tallied — so one bad rep can't kill a multi-hour run.
    """
    work = Path(spec["work"])
    cid = spec["chunk_id"]
    parts_dir = work / "parts"
    out = parts_dir / f"part_{cid:05d}.parquet"
    if out.exists():
        return cid, -1, 0, 0.0  # already done (n_written=-1 signals "skipped")

    take_batch = spec.get("take_batch", _TAKE_BATCH)
    log_io = spec.get("log_io", False)
    pid = os.getpid()
    t0 = time.monotonic()
    plan = pq.read_table(work / "plan" / f"chunk_{cid:05d}.parquet").to_pandas()
    n_sub = (len(plan) + take_batch - 1) // take_batch
    if log_io:
        _log(f"[run] w{pid} chunk {cid}: START {len(plan):,} reps / {n_sub} sub-batches")
    tmp = parts_dir / f".part_{cid:05d}.parquet.tmp"
    writer = None
    n_written = 0
    n_mismatch = 0
    try:
        # Process the chunk in sub-batches so peak RAM is bounded regardless of chunk size.
        for b0 in range(0, len(plan), take_batch):
            sb = b0 // take_batch
            sub = plan.iloc[b0:b0 + take_batch]
            indices = sub["row_index"].to_numpy().tolist()
            # Time the take: a hung S3 read is the known wedge, so log a "take START"
            # (with --log-io) whose missing completion pins the stuck worker+sub-batch,
            # and always warn on a take slower than SLOW_TAKE_WARN_S.
            if log_io:
                _log(f"[run] w{pid} chunk {cid} sb {sb}/{n_sub}: take START ({len(sub)} rows)")
            t_take0 = time.monotonic()
            tbl = _DS.take(indices, columns=["protein_hash", "structure_blob", "sequence"])
            t_take = time.monotonic() - t_take0
            if t_take > SLOW_TAKE_WARN_S:
                _log(f"[run] w{pid} chunk {cid} sb {sb}/{n_sub}: SLOW take {t_take:.1f}s "
                     f"({len(sub)} rows); RAM avail {_mem_avail_gb():.0f} GB")
            elif log_io:
                _log(f"[run] w{pid} chunk {cid} sb {sb}/{n_sub}: take {t_take:.1f}s; "
                     f"RAM avail {_mem_avail_gb():.0f} GB")
            got_hash = tbl.column("protein_hash").to_pylist()
            blobs = tbl.column("structure_blob").to_pylist()
            ref_seqs = tbl.column("sequence").to_pylist()

            rows = []
            for i in range(len(sub)):
                h = sub["protein_hash"].iloc[i]
                # take() preserves order, but guard against any row_index/hash drift.
                if got_hash[i] != h:
                    _log(f"[run] WARNING chunk {cid}: hash mismatch at {b0 + i} "
                         f"({got_hash[i]} != {h}); skipping row")
                    continue
                cif, sequence = decode_to_cif(blobs[i], name=h[:4])
                seq_ok = sequence == ref_seqs[i]  # decoded seq vs Atlas sequence column
                if not seq_ok:
                    n_mismatch += 1
                    _log(f"[run] WARNING chunk {cid} rep {h}: decoded sequence != Atlas "
                         f"sequence (len {len(sequence)} vs {len(ref_seqs[i])}); seq_ok=False")
                row = {ID_COLUMN: h, CIF_COLUMN: cif, "sequence": sequence,
                       "seq_ok": seq_ok, "split": _SPLIT_TAG, "source": SOURCE_TAG}
                for c in _CARRY_COLS:
                    if c in sub.columns:
                        row[_AFDB_RENAME.get(c, c)] = sub[c].iloc[i]
                rows.append(row)

            if not rows:
                del tbl, got_hash, blobs, ref_seqs
                _release_memory()
                continue
            batch_tbl = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(tmp, batch_tbl.schema, compression="zstd")
            else:
                batch_tbl = batch_tbl.cast(writer.schema)  # keep every row group identical
            writer.write_table(batch_tbl)
            n_written += len(rows)
            del tbl, got_hash, blobs, ref_seqs, rows, batch_tbl
            # Return Arrow's retained arena memory to the OS each sub-batch. Without
            # this the default memory pool holds freed take() buffers, so RSS creeps
            # over a chunk's ~40 sub-batches until 96 synchronized workers OOM-cascade
            # the 192 GB box ~halfway through the first chunk (observed).
            _release_memory()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:  # no rows (all hash-drifted, effectively impossible) — empty part
        pq.write_table(pa.table({ID_COLUMN: pa.array([], pa.string())}), tmp,
                       compression="zstd")
    os.replace(tmp, out)  # atomic: a part file only appears once fully written
    return cid, n_written, n_mismatch, time.monotonic() - t0


def stage_run(work: Path, *, workers: int, skip_list: Path | None,
              take_batch: int = _TAKE_BATCH, num_shards: int = 1, shard_id: int = 0,
              max_chunks: int | None = None, log_io: bool = False) -> None:
    """Decode all planned chunks into checkpointed parts (resumable).

    Sharding (``num_shards``/``shard_id``): each shard owns the chunks whose id is
    ``id % num_shards == shard_id``, so N boxes can decode disjoint chunk sets into
    the same ``parts/`` prefix with no coordination (the work is S3-read-latency
    bound, so throughput scales ~linearly per box). ``max_chunks`` caps the number
    of chunks decoded — used by the throughput probe to time a small burst.
    """
    plan_dir = work / "plan"
    parts_dir = work / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(int(p.stem.split("_")[1]) for p in plan_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise SystemExit(f"no plan chunks in {plan_dir}; run the 'plan' stage first")
    if num_shards > 1:
        chunks = [c for c in chunks if c % num_shards == shard_id]

    # Resume: skip chunks whose part is already present locally or in the S3-derived
    # skip list (chunk ids, one per line) the launcher builds at boot.
    done = {int(p.stem.split("_")[1]) for p in parts_dir.glob("part_*.parquet")}
    if skip_list and skip_list.exists():
        done |= {int(x) for x in skip_list.read_text().split() if x.strip().isdigit()}
    todo = [c for c in chunks if c not in done]
    if max_chunks is not None:
        todo = todo[:max_chunks]
    shard_tag = f"shard {shard_id}/{num_shards}, " if num_shards > 1 else ""
    _log(f"[run] {shard_tag}{len(chunks):,} chunks in shard; {len(done):,} already done; "
         f"{len(todo):,} to do on {workers} workers (take_batch {take_batch:,}); "
         f"{_resources(work)}")
    if not todo:
        _log("[run] nothing to do — all chunks materialized")
        return

    specs = [{"work": str(work), "chunk_id": c, "take_batch": take_batch,
              "log_io": log_io} for c in todo]
    n_done = n_rows = n_mismatch = 0
    t_start = time.monotonic()
    ctx = mp.get_context("spawn")  # Lance is not fork-safe; each worker opens its own handle
    # maxtasksperchild=1: recycle each worker after ONE chunk so no per-process memory
    # accumulates across chunks. The diagnostic showed RAM stepping from ~55 GB avail
    # (workers' 1st chunk) down to ~25 GB (2nd chunk) — cross-chunk creep that would
    # OOM over the full run's ~26 chunks/worker. A fresh process per chunk resets it to
    # the safe single-chunk level; reopening Lance per ~5-min chunk is ~1% overhead.
    with ctx.Pool(workers, initializer=_worker_init, maxtasksperchild=1) as pool:
        for cid, written, mism, secs in pool.imap_unordered(_materialize_chunk, specs):
            n_done += 1
            if written >= 0:
                n_rows += written
                n_mismatch += mism
            rate = n_done / max(1e-9, (time.monotonic() - t_start))
            eta_h = (len(todo) - n_done) / max(1e-9, rate) / 3600
            flag = f", {n_mismatch:,} seq-mismatch" if n_mismatch else ""
            _log(f"[run] chunk {cid:05d}: {written:,} structures in {secs:.0f}s "
                 f"({n_done}/{len(todo)} done, {n_rows:,} rows{flag}, ETA {eta_h:.1f}h); "
                 f"{_resources(work)}")
    _log(f"[run] DONE: materialized {n_rows:,} structures across {n_done} chunks; "
         f"{n_mismatch:,} had seq_ok=False (Atlas seq != decoded seq)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["plan", "run", "full"],
                    help="plan (locate rows) | run (decode) | full (plan then run)")
    ap.add_argument("--manifest", type=Path, help="selected_manifest.csv (plan/full)")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="scratch + output root (holds plan/ and parts/)")
    ap.add_argument("--chunk-size", type=int, default=20_000,
                    help="reps per plan chunk / output part")
    ap.add_argument("--scan-workers", type=int, default=1,
                    help="parallel fork workers for the plan scan")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel fork workers for decode")
    ap.add_argument("--batch-rows", type=int, default=100_000,
                    help="Lance scan batch size in the plan stage")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: materialize only the first N located reps")
    ap.add_argument("--scan-rows", type=int, default=None,
                    help="cap Atlas rows considered in the plan scan (testing)")
    ap.add_argument("--skip-list", type=Path, default=None,
                    help="file of chunk ids (from S3) to treat as already done")
    ap.add_argument("--take-batch", type=int, default=_TAKE_BATCH,
                    help="reps per Lance take/decode sub-batch (bounds worker RAM; "
                         "smaller lets more workers run, raising S3 read concurrency)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split chunks across this many boxes (chunk_id %% num_shards)")
    ap.add_argument("--shard-id", type=int, default=0, help="this box's shard index")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="decode at most N chunks then stop (throughput probe)")
    ap.add_argument("--log-io", action="store_true",
                    help="log every take/decode sub-batch (per-worker) to diagnose "
                         "an S3-read wedge; noisy, use for short diagnostic runs")
    args = ap.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"--shard-id {args.shard_id} out of range for "
                         f"--num-shards {args.num_shards}")

    if args.stage in ("plan", "full"):
        if args.manifest is None:
            raise SystemExit("--manifest is required for the plan/full stages")
        _log(f"=== stage plan: START — {_resources(args.work_dir)}")
        stage_plan(args.manifest, args.work_dir, scan_workers=args.scan_workers,
                   chunk_size=args.chunk_size, batch_rows=args.batch_rows,
                   limit=args.limit, scan_rows=args.scan_rows)
        _log(f"=== stage plan: DONE — {_resources(args.work_dir)}")
    if args.stage in ("run", "full"):
        _log(f"=== stage run: START — {_resources(args.work_dir)}")
        stage_run(args.work_dir, workers=args.workers, skip_list=args.skip_list,
                  take_batch=args.take_batch, num_shards=args.num_shards,
                  shard_id=args.shard_id, max_chunks=args.max_chunks, log_io=args.log_io)
        _log(f"=== stage run: DONE — {_resources(args.work_dir)}")


if __name__ == "__main__":
    main()
