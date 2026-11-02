# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B worker — one CoreWeave shard: staged backbones -> redesigned documents.

Runs as an independent 1xH100 task on cw-rno2a (no gang; see the root
`AGENTS.md` "Single-GPU inference fan-out" recipe). One task per GPU, 8 tasks
per node, each with a slice of the node's 128 vCPUs.

The two halves of the work land on different hardware and roughly balance:

* **ProteinMPNN on the GPU** — ~0.1 s per backbone for all 8 designs.
* **pyconfind on the CPUs** — ~4.2 s per backbone for 8 documents, spread over
  the task's cores, so ~0.3 s of wall-clock at 15 cores.

So the loop is: read a chunk of backbones, design them all on the GPU in
exact-length batches, then fan the document generation out over a process pool
while the next chunk designs. Neither device idles waiting for the other.

Why a process pool and not threads: pyconfind's contact loop is the CPU cost,
and a thread pool would serialize on the GIL for the Python-side work around
it. Each worker process pays the rotamer-library parse once (memoized in
`generate_rows`), which is why the pool is created once per task, not per chunk.

**All object storage goes through fsspec**, never pyarrow's native S3 or the
`aws` CLI: iris injects CoreWeave's endpoint and credentials as an `FSSPEC_S3`
blob that only fsspec/s3fs reads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


def _log(message: str) -> None:
    print(f"[exp266-cw] {message}", file=sys.stderr, flush=True)


def _read_shard(uri: str, columns: list[str]) -> list[dict]:
    with fsspec.open(uri, "rb") as handle:
        table = pq.read_table(handle, columns=columns)
    return table.to_pylist()


def _write_shard(uri: str, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    with fsspec.open(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")


def _documents_for_one(payload):
    """Process-pool entry point: (staged row, designs) -> document rows."""
    import generate_rows

    row, designs = payload
    return generate_rows.documents_for_designs(
        row, designs, rotamer_library=generate_rows._load_rotamer_library()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", required=True,
                    help="Staged backbone parquet glob (s3:// via fsspec).")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix; one parquet per input file.")
    ap.add_argument("--shard", required=True, metavar="I/N",
                    help="This task's slice of the input files, interleaved.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=2048,
                    help="Backbones designed on the GPU before their documents "
                         "are handed to the CPU pool.")
    ap.add_argument("--max-batch", type=int, default=256,
                    help="Backbones per ProteinMPNN call (before the 8x design "
                         "replication).")
    ap.add_argument("--max-batch-residues", type=int, default=100_000,
                    help="Padded residues per call, counting the 8x design "
                         "replication. Device memory scales at ~0.52 MB per "
                         "padded residue (measured), so 100k ~ 52 GB — sized "
                         "for an 80 GB H100 with headroom, and the knob to turn "
                         "down first on OOM.")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Smoke cap: staged files this task processes.")
    ap.add_argument("--cpu-workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Processes generating documents. Defaults to the "
                         "task's core count minus one for the GPU feeder.")
    args = ap.parse_args()

    # Imported after argparse so --help works without torch in the image.
    from redesign import load_model

    columns = ["entry_id", "chain_id", "resnum_start", "sequence", "coords_milli",
               "ca_plddt", "struct_cluster_id", "seq_cluster_id", "split", "round",
               "native_contacts_emitted", "native_sha1"]

    shard_i, num_shards = (int(x) for x in args.shard.split("/"))
    fs, _ = fsspec.core.url_to_fs(args.input_glob)
    all_files = sorted(fs.glob(args.input_glob))
    if not all_files:
        raise FileNotFoundError(f"no staged backbones match {args.input_glob}")
    # Interleave, don't block: Stage A sorted the manifest by seq_len, so a
    # contiguous slice would hand one task every long protein and that task
    # would set the wall-clock for the whole fan-out.
    my_files = all_files[shard_i::num_shards]
    if args.max_files is not None:
        my_files = my_files[: args.max_files]
    _log(f"shard {shard_i}/{num_shards}: {len(my_files)} of {len(all_files)} files")

    load_model(args.device)            # once per task, before any input is read

    # Warm pyconfind's rotamer library BEFORE forking the pool. Each child
    # would otherwise call `cached_rotamer_library()` itself, and 14 processes
    # downloading + extracting the same 6.2 MB tarball into one shared cache
    # directory race: the loser parses a half-written library and dies with
    #   ValueError: setting an array element with a sequence ...
    #   (23755,) + inhomogeneous part
    # Warming here means the fork children inherit the populated
    # `functools.cache` and never touch the download path at all. (A warm local
    # cache hides this completely, which is why it only appeared on-cluster.)
    import generate_rows
    t_warm = time.perf_counter()
    if generate_rows._load_rotamer_library() is None:
        raise RuntimeError(
            "rotamer library failed to load in the parent; children would each "
            "retry the download and race. Fix the library fetch before scaling."
        )
    _log(f"rotamer library warm in {time.perf_counter() - t_warm:.1f}s; "
         f"model on {args.device}; {args.cpu_workers} document processes")

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
        for path in my_files:
            _process_file(fs.unstrip_protocol(path), args, columns, pool)

    _log(f"shard {shard_i}/{num_shards} done in {time.perf_counter()-started:.0f}s")
    return 0


def _process_file(uri, args, columns, pool) -> None:
    """One staged parquet -> one document parquet, resumable."""
    from redesign import BackboneEntry, batch_by_exact_length, design_batch
    from backbone import backbone_coords_from_row

    stem = uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".parquet")
    out_uri = f"{args.out_prefix.rstrip('/')}/documents-{stem}.parquet"
    out_fs, out_path = fsspec.core.url_to_fs(out_uri)
    if out_fs.exists(out_path):
        # Resume: a preempted task re-runs from the first unwritten file rather
        # than redoing the shard. Batch band is preemptible by design.
        _log(f"skip {stem} (output exists)")
        return

    t0 = time.perf_counter()
    rows = _read_shard(uri, columns)
    _log(f"{stem}: read {len(rows)} backbones in {time.perf_counter()-t0:.1f}s")

    out_rows: list[dict] = []
    n_designed = 0
    t_gpu = t_cpu = 0.0

    for start in range(0, len(rows), args.chunk_size):
        chunk = rows[start : start + args.chunk_size]
        by_id = {r["entry_id"]: r for r in chunk}

        t = time.perf_counter()
        entries = [
            BackboneEntry(r["entry_id"], r["sequence"], backbone_coords_from_row(r))
            for r in chunk
        ]
        designs_by_entry: dict[str, list] = {}
        for batch in batch_by_exact_length(
            entries, max_batch=args.max_batch,
            max_batch_residues=args.max_batch_residues,
        ):
            for design in design_batch(batch, device=args.device):
                designs_by_entry.setdefault(design.entry_id, []).append(design)
        t_gpu += time.perf_counter() - t

        t = time.perf_counter()
        payloads = [
            (by_id[entry_id], designs)
            for entry_id, designs in designs_by_entry.items()
        ]
        for records in pool.map(_documents_for_one, payloads, chunksize=8):
            out_rows.extend(records)
        t_cpu += time.perf_counter() - t

        n_designed += len(chunk)
        rate = n_designed / (time.perf_counter() - t0)
        _log(f"{stem}: {n_designed}/{len(rows)} backbones, {len(out_rows)} "
             f"documents ({rate:.1f} backbones/s; gpu {t_gpu:.0f}s cpu {t_cpu:.0f}s)")

    _write_shard(out_uri, out_rows)
    elapsed = time.perf_counter() - t0
    _log(f"{stem}: wrote {len(out_rows)} documents to {out_uri} in {elapsed:.0f}s "
         f"({len(rows)/elapsed:.1f} backbones/s, gpu {t_gpu:.0f}s cpu {t_cpu:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
