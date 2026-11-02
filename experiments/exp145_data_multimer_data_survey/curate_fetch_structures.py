# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch AFDB dimer coordinate files into tar shards, locally or on S3.

Phase 2 of ``CURATION_PLAN.md``. Two modes:

``--sweep`` is the rate pilot. It fetches the same batch size at several worker
counts and reports files/s for each. Run on 2026-09-05 over 8,012 structures it
found throughput linear in workers with no throttling at all (1.44 / 2.85 / 5.96 /
11.54 files/s at 1 / 2 / 4 / 8), because ``alphafold.ebi.ac.uk/files/`` is served
out of Google Cloud Storage rather than EBI's own web tier. We cap at 8 workers
anyway; no rate limit is published and EMBL-EBI's terms reserve the right to block
anyone degrading service for others.

The default mode is the real fetch. Design decisions that are not obvious:

*Tar shards, not one object per structure.* A million individual S3 keys is a
million PUTs (a third of the total AWS cost at this scale) and makes every later
listing slow. Structures go into ``shard_XXXXX.tar`` of ``--shard-size`` members
each, written whole.

*Resume is at shard granularity.* Checking whether each of a million structures
already exists would itself be a million requests. A shard that is already present
at the destination is skipped entirely, so an interrupted run redoes at most one
shard's worth of work.

*Missing models are recorded, not just logged.* Part of the release 404s (28 of
2,000 in one pilot batch, non-transient on serial retry). The dataset is defined
by what actually downloaded, so each shard writes a sidecar JSON naming its
missing ids, and those aggregate into the published manifest.

*S3 goes through pyarrow, never boto3.* On ephemeral EC2 workers ``pip install
boto3`` wedges the box about 60 seconds in; ``pyarrow.fs.S3FileSystem`` picks up
the instance role automatically. This cost exp91 two days to find.

Usage::

    python3 curate_fetch_structures.py --sweep --n 2000
    python3 curate_fetch_structures.py --split val \
        --ids data/curation/dimer_split_assignment_id30_cov50.csv.gz \
        --out s3://bucket/prefix/ --shard 0 --n-shards 8 --workers 8
"""

import argparse
import csv
import gzip
import http.client
import io
import json
import platform
import queue
import socket
import tarfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FILES_BASE = "https://alphafold.ebi.ac.uk/files/"
MODEL_VERSION = 1
CONFIDENT_CSVS = ("confident_homodimers.csv.gz", "confident_heterodimers.csv.gz")

# 429 is the one we would be hunting for. The 5xx family is retried because a
# server returns them under load, which is a throttle wearing a different hat.
# 404 is deliberately absent: those models are missing from the release.
RETRY_CODES = (429, 500, 502, 503, 504)
ATTEMPTS = 5
PROGRESS_SECONDS = 15
MAX_WORKERS = 8

TIMING_COLUMNS = [
    "stem", "kind", "wire_bytes", "decoded_bytes",
    "elapsed_seconds", "attempts", "http_status",
    "worker", "n_workers", "runner_tag", "hostname", "platform", "timestamp_utc",
]


class Destination:
    """Somewhere to put a finished shard. Local directory or S3 prefix.

    Both back ends expose the same three operations, so the fetch loop never
    branches on which one it has.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri.rstrip("/")
        self.is_s3 = self.uri.startswith("s3://")
        if self.is_s3:
            # Imported here because pyarrow is only needed for the S3 back end and
            # a local run should not require it.
            import pyarrow.fs as pafs

            path = self.uri[len("s3://"):]
            bucket = path.split("/", 1)[0]
            self.fs = pafs.S3FileSystem(region=pafs.resolve_s3_region(bucket))
            self.root = path
        else:
            self.root = self.uri
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def existing(self) -> set[str]:
        """Names already present, so completed shards can be skipped on resume."""
        if not self.is_s3:
            return {p.name for p in Path(self.root).iterdir()}
        import pyarrow.fs as pafs

        selector = pafs.FileSelector(self.root, allow_not_found=True, recursive=False)
        return {info.base_name for info in self.fs.get_file_info(selector)}

    def write(self, name: str, payload: bytes) -> None:
        if not self.is_s3:
            (Path(self.root) / name).write_bytes(payload)
            return
        # compression=None is required rather than merely tidy: pyarrow defaults to
        # "detect" and gzips any object whose name ends in .gz. Our shard names do
        # not, but a future caller passing one would get a doubly-compressed object
        # that reads back as garbage. The read side has the mirror-image trap: it
        # silently gunzips on open, which is what broke the first EC2 smoke test.
        with self.fs.open_output_stream(f"{self.root}/{name}", compression=None) as handle:
            handle.write(payload)


class Counters:
    """Thread-safe tallies. Every worker touches these, so they take the lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.wire_bytes = 0
        self.decoded_bytes = 0
        self.retries = 0
        self.throttled = 0
        self.missing: list[str] = []
        self.failed: list[str] = []
        self.rows: list[dict[str, object]] = []
        self.members: list[tuple[str, bytes]] = []


def fetch_one(model_id: str, timeout: int) -> tuple[bytes, int, int, int]:
    """GET one model. Returns (gzip bytes, decoded length, attempts, status).

    The server compresses only when the client asks, and urllib never asks, so the
    header is explicit. urllib also does not decode the response, which is what we
    want: the compressed body is what goes into the tar.

    Retries are broad on purpose. ``http.client.RemoteDisconnected`` is neither an
    ``HTTPError`` nor a ``URLError``, and a narrower clause let it kill a
    multi-hour run outright during the sequence fetch.
    """
    url = f"{FILES_BASE}{model_id}-model_v{MODEL_VERSION}.cif"
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = response.status
            if body[:2] != b"\x1f\x8b":
                # Uncompressed reply. Keep the archive uniform but report the true
                # wire size, and flag it by negating the status.
                return gzip.compress(body), len(body), attempt + 1, -status
            return body, len(gzip.decompress(body)), attempt + 1, status
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            code = getattr(exc, "code", None)
            if (code is not None and code not in RETRY_CODES) or attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def worker(
    index: int,
    jobs: "queue.Queue[tuple[str, str] | None]",
    counters: Counters,
    n_workers: int,
    timeout: int,
    meta: dict[str, str],
) -> None:
    while True:
        item = jobs.get()
        if item is None:
            return
        model_id, kind = item
        start = time.time()
        try:
            body, decoded, attempts, status = fetch_one(model_id, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            with counters.lock:
                counters.done += 1
                counters.missing.append(model_id)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a shard
            with counters.lock:
                counters.done += 1
                counters.failed.append(model_id)
            print(f"  ! {model_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        elapsed = time.time() - start
        with counters.lock:
            counters.done += 1
            counters.wire_bytes += len(body)
            counters.decoded_bytes += decoded
            counters.retries += attempts - 1
            counters.throttled += attempts > 1
            counters.members.append((f"{model_id}.cif.gz", body))
            counters.rows.append({
                "stem": model_id,
                "kind": kind,
                "wire_bytes": len(body),
                "decoded_bytes": decoded,
                "elapsed_seconds": round(elapsed, 4),
                "attempts": attempts,
                "http_status": status,
                "worker": index,
                "n_workers": n_workers,
                **meta,
            })


def run_batch(
    ids: list[tuple[str, str]],
    n_workers: int,
    timeout: int,
    meta: dict[str, str],
) -> tuple[Counters, dict[str, object]]:
    """Fetch ``ids`` with ``n_workers`` threads. Returns the counters and a summary."""
    jobs: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
    for item in ids:
        jobs.put(item)
    for _ in range(n_workers):
        jobs.put(None)

    counters = Counters()
    threads = [
        threading.Thread(
            target=worker,
            args=(i, jobs, counters, n_workers, timeout, meta),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    start = time.time()
    for thread in threads:
        thread.start()

    # Poll often so ``elapsed`` is the batch's real wall time rather than a
    # multiple of the print interval, but print only every PROGRESS_SECONDS.
    last_print = start
    while any(t.is_alive() for t in threads):
        time.sleep(0.2)
        now = time.time()
        if now - last_print < PROGRESS_SECONDS:
            continue
        with counters.lock:
            done, wire = counters.done, counters.wire_bytes
        print(
            f"  {done:,}/{len(ids):,}  {done / (now - start):.1f} files/s  "
            f"{wire / 1e6:.0f} MB",
            flush=True,
        )
        last_print = now
    for thread in threads:
        thread.join()

    elapsed = time.time() - start
    ok = len(counters.members)
    return counters, {
        "n_workers": n_workers,
        "n_requested": len(ids),
        "n_ok": ok,
        "n_missing_404": len(counters.missing),
        "n_failed": len(counters.failed),
        "n_needing_retry": counters.throttled,
        "total_retries": counters.retries,
        "elapsed_seconds": round(elapsed, 1),
        "files_per_second": round(ok / elapsed, 2) if elapsed else 0.0,
        "wire_mb_per_second": round(counters.wire_bytes / 1e6 / elapsed, 2) if elapsed else 0.0,
        "mean_wire_kb": round(counters.wire_bytes / 1e3 / max(ok, 1), 1),
        "mean_decoded_kb": round(counters.decoded_bytes / 1e3 / max(ok, 1), 1),
        "compression_ratio": round(counters.decoded_bytes / max(counters.wire_bytes, 1), 2),
    }


def pack_tar(members: list[tuple[str, bytes]]) -> bytes:
    """Build an uncompressed tar in memory. The members are already gzipped."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, payload in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def load_ids_from_list(path: Path, split: str | None) -> list[tuple[str, str]]:
    """Read the id list to fetch.

    Accepts either a download list from ``curate_val_subunit_clusters.py`` or a
    split assignment from ``curate_val_split.py``. ``--split val`` against the
    assignment file is the uncollapsed path: it fetches every val-eligible
    complex, which is what following PINDER means here, since PINDER clusters
    every system rather than a sequence-collapsed subset of them.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        reader = csv.DictReader(handle)
        if split is not None and "split" not in (reader.fieldnames or []):
            raise SystemExit(f"--split {split} given but {path} has no 'split' column")
        rows = [
            (row["model_entity_id"], row["kind"])
            for row in reader
            if split is None or row["split"] == split
        ]
    if not rows:
        raise SystemExit(f"{path}: no rows selected")
    return rows


def load_ids_strided(data_dir: Path, stride: int) -> list[tuple[str, str]]:
    """Take every ``stride``-th confident model, for the rate pilot.

    Striding rather than taking a head: the metadata CSVs are blocked by organism,
    so the first n rows are one species and would give a throughput number for a
    single size regime.
    """
    picked: list[tuple[str, str]] = []
    for name in CONFIDENT_CSVS:
        with gzip.open(data_dir / name, "rt") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if i % stride == 0:
                    picked.append((row["model_entity_id"], row["kind"]))
    picked.sort()
    return picked


def append_timings(path: Path, rows: list[dict[str, object]]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/curation"))
    parser.add_argument("--ids", type=Path, help="download list CSV; omit for --sweep")
    parser.add_argument("--split", help="keep only rows with this split, e.g. val")
    parser.add_argument("--out", default="data/curation/structures", help="directory or s3:// prefix")
    parser.add_argument("--shard", type=int, default=0, help="this host's index")
    parser.add_argument("--n-shards", type=int, default=1, help="number of hosts")
    parser.add_argument("--shard-size", type=int, default=2000, help="structures per tar")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sweep", action="store_true", help="rate pilot at 1, 2, 4, 8 workers")
    parser.add_argument("--n", type=int, default=2000, help="files per batch, --sweep only")
    parser.add_argument("--stride", type=int, default=97, help="id stride, --sweep only")
    args = parser.parse_args()

    if args.workers > MAX_WORKERS:
        raise SystemExit(
            f"--workers {args.workers} exceeds the self-imposed cap of {MAX_WORKERS}. "
            f"EBI publishes no rate limit; see CURATION_PLAN.md phase 2 before raising it."
        )

    meta = {
        "runner_tag": "local",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destination = Destination(args.out)

    if args.sweep:
        ids = load_ids_strided(args.data_dir, args.stride)
        worker_counts = [1, 2, 4, 8]
        needed = args.n * len(worker_counts)
        if len(ids) < needed:
            raise SystemExit(f"stride {args.stride} yields {len(ids):,} ids, need {needed:,}")
        print(f"{needed:,} model ids, stride {args.stride}, out {destination.uri}\n")
        results = []
        for i, n_workers in enumerate(worker_counts):
            # Disjoint blocks so a later batch never benefits from a file an
            # earlier batch already pulled through EBI's caches.
            block = ids[i * args.n:(i + 1) * args.n]
            print(f"[{i + 1}/{len(worker_counts)}] {n_workers} worker(s), {len(block):,} files")
            counters, summary = run_batch(block, n_workers, args.timeout, meta)
            destination.write(f"sweep_{n_workers:02d}w.tar", pack_tar(counters.members))
            append_timings(args.data_dir / "fetch_timings.csv", counters.rows)
            print(f"  -> {json.dumps(summary)}\n", flush=True)
            results.append(summary)
        (args.data_dir / "fetch_pilot_summary.json").write_text(
            json.dumps({"endpoint": FILES_BASE, "batches": results, **meta}, indent=2)
        )
        return

    if not args.ids:
        raise SystemExit("--ids is required unless --sweep is given")

    everything = load_ids_from_list(args.ids, args.split)
    # Interleaved, not contiguous: a contiguous block would hand one host all the
    # long sequences and set the wall clock for the whole fan-out.
    mine = everything[args.shard::args.n_shards]
    shards = [mine[i:i + args.shard_size] for i in range(0, len(mine), args.shard_size)]
    present = destination.existing()
    print(
        f"{len(everything):,} ids, shard {args.shard}/{args.n_shards} takes {len(mine):,} "
        f"in {len(shards)} tar(s) of {args.shard_size}\n"
        f"  destination {destination.uri}, {len(present):,} object(s) already there\n",
        flush=True,
    )

    for i, block in enumerate(shards):
        name = f"shard_{args.shard:03d}_{i:05d}.tar"
        if name in present:
            print(f"[{i + 1}/{len(shards)}] {name} exists, skipping")
            continue
        print(f"[{i + 1}/{len(shards)}] {name}, {len(block):,} files", flush=True)
        counters, summary = run_batch(block, args.workers, args.timeout, meta)
        if counters.failed:
            raise RuntimeError(
                f"{name}: {len(counters.failed)} non-404 failures after {ATTEMPTS} "
                f"attempts each, first {counters.failed[:5]}. Shard not written."
            )
        destination.write(name, pack_tar(counters.members))
        destination.write(
            f"{name}.json",
            json.dumps({"shard": name, "missing_404": sorted(counters.missing), **summary}).encode(),
        )
        append_timings(args.data_dir / "fetch_timings.csv", counters.rows)
        print(f"  -> {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
