# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream AlphaFold DB's bulk ``sequences.fasta`` and keep only AFCDB's accessions.

AFCDB ships no FASTA and its ``modelEntityId`` is an opaque numeric id, so the
subunit sequences have to come from AlphaFold DB's 110 GB bulk file. Mirroring
that file is neither necessary nor allowed under the 10 GB rule: this streams it
over HTTP range requests, keeps only the ~21.4M accessions AFCDB references, and
writes a gzipped FASTA of roughly 2 GB. Transfer is large; storage is not.

The file is split into ``--workers`` byte ranges. Worker *i* owns the records
whose **start offset** falls in ``[start_i, start_i+1)``, so it skips forward to
the first record boundary at or after its start and stops once a record begins
at or after its end. That partition is exact and lossless, with no record
duplicated or dropped at a seam.

Fail-loud: an HTTP error, a malformed record, an unterminated final record, or
any requested accession that the source does not contain aborts the run. A
silently short sequence file would silently under-decontaminate the corpus.

Run it on an Iris CPU pod pinned near EBI, not on the workstation, and write
the result to the co-located bucket (see ``AGENTS.md``: co-locate a job's I/O
with its compute zone)::

    iris --cluster marin job run --no-wait --enable-extra-resources \\
        --cpu=8 --memory=16GB --disk=100GB --zone=europe-west4-a \\
        -- uv run --with duckdb --with fsspec --with gcsfs \\
           python fetch_sequences.py \\
             --accessions gs://marin-eu-west4/.../inputs/accessions.parquet \\
             --out gs://marin-eu-west4/.../sequences --workers 32

Locally, ``--normalized`` derives the accession set from the census Parquet
instead, and ``--out`` may be any local directory.
"""

import argparse
import json
import re
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import fsspec

SEQUENCES_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/sequences.fasta"
#: Longest AFDB record we will tolerate without seeing a record boundary. AFDB
#: fragments cap at 2,700 residues, so this is ~100x headroom; exceeding it means
#: the stream is not the file we think it is.
MAX_RECORD_BYTES = 4 << 20
_READ_CHUNK = 1 << 20
_HEADER = re.compile(rb"^>(?P<token>\S+)")


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def accession_from_token(token: bytes) -> bytes:
    """``AFDB:AF-<acc>-F1`` / ``sp|<acc>|NAME`` / ``<acc>`` -> ``<acc>``."""
    if token.startswith(b"AFDB:"):
        token = token[5:]
    if token.startswith(b"AF-") and b"-F" in token:
        return token[3 : token.rindex(b"-F")]
    fields = token.split(b"|")
    if len(fields) >= 3 and fields[0] in (b"sp", b"tr"):
        return fields[1]
    return token


def _duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    try:  # Needed for gs:// inputs; harmless and offline-safe when already present.
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except duckdb.Error:
        pass
    return con


def required_accessions(normalized_glob: str) -> set[bytes]:
    """Every UniProt accession either subunit of any AFCDB model refers to."""
    src = f"read_parquet({_sql_literal(normalized_glob)}, union_by_name=true)"
    rows = _duckdb().execute(
        f"""
        SELECT DISTINCT accession FROM (
            SELECT accession_a AS accession FROM {src}
            UNION ALL SELECT accession_b FROM {src}
        ) WHERE accession IS NOT NULL
        """
    ).fetchall()
    return {row[0].encode() for row in rows}


def listed_accessions(path: str) -> set[bytes]:
    """The pre-extracted accession list, so a pod need not read 2 GB of census.

    A remote list is copied down whole before being opened. Reading Parquet
    straight off ``gs://`` would mean ranged reads through gcsfs, which is the
    416 failure mode that bit earlier experiments on cluster, and would also
    need DuckDB credentials the pod's service account does not hand it.
    """
    if "://" in path:
        with fsspec.open(path, "rb") as remote:
            payload = remote.read()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as local:
            local.write(payload)
            path = local.name
    rows = _duckdb().execute(
        f"SELECT DISTINCT accession FROM read_parquet({_sql_literal(path)}) "
        "WHERE accession IS NOT NULL"
    ).fetchall()
    return {row[0].encode() for row in rows}


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
    if not length:
        raise RuntimeError(f"{url}: server did not report a Content-Length")
    return int(length)


@dataclass
class ShardResult:
    """What one byte range contributed."""

    index: int
    records: int
    matched: int
    bytes_read: int
    seconds: float


class _RangeReader:
    """Buffered forward reader over an open-ended HTTP range."""

    def __init__(self, url: str, start: int) -> None:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
        self._response = urllib.request.urlopen(request, timeout=300)
        if self._response.status != 206:
            raise RuntimeError(
                f"{url}: expected 206 for a range request, got {self._response.status}"
            )
        self._buffer = b""
        self._eof = False
        self.consumed = start

    def fill(self, minimum: int) -> None:
        while len(self._buffer) < minimum and not self._eof:
            chunk = self._response.read(_READ_CHUNK)
            if not chunk:
                self._eof = True
                break
            self._buffer += chunk

    def readline(self) -> bytes:
        while b"\n" not in self._buffer and not self._eof:
            self.fill(len(self._buffer) + _READ_CHUNK)
        if not self._buffer:
            return b""
        index = self._buffer.find(b"\n")
        if index < 0:
            line, self._buffer = self._buffer, b""
        else:
            line, self._buffer = self._buffer[: index + 1], self._buffer[index + 1 :]
        self.consumed += len(line)
        return line

    def close(self) -> None:
        self._response.close()


def _scan_range(
    url: str,
    wanted: set[bytes],
    index: int,
    start: int,
    end: int,
    out_path: str,
    progress_seconds: float = 60.0,
) -> ShardResult:
    """Write every wanted record whose start offset lies in ``[start, end)``."""
    began = time.monotonic()
    # Open one byte early so we can tell whether ``start`` is itself a record
    # boundary. A range that starts exactly on a header owns that record; without
    # this check it would skip forward and the previous range, which stops at
    # ``start``, would not emit it either.
    reader = _RangeReader(url, start - 1 if start > 0 else 0)
    records = matched = 0
    first_start = start
    last_report = began

    try:
        if start > 0:
            reader.fill(2)
            previous = reader._buffer[:1]
            reader._buffer = reader._buffer[1:]
            reader.consumed += 1
            at_boundary = previous == b"\n" and reader._buffer[:1] == b">"
            if not at_boundary:
                # We opened mid-record; discard it. Its owner is the previous
                # range. ``readline`` always leaves us at a line start, so
                # testing for ``>`` there is a true record-boundary test.
                skipped = 0
                while True:
                    line = reader.readline()
                    if not line:
                        return ShardResult(index, 0, 0, reader.consumed - start, 0.0)
                    skipped += len(line)
                    if skipped > MAX_RECORD_BYTES:
                        raise RuntimeError(
                            f"range {index}: no record boundary within "
                            f"{MAX_RECORD_BYTES} bytes of offset {start}"
                        )
                    reader.fill(1)
                    if reader._buffer[:1] == b">":
                        break
            first_start = reader.consumed

        with fsspec.open(str(out_path), "wb", compression="gzip") as sink:
            record_start = first_start
            while record_start < end:
                header = reader.readline()
                if not header:
                    break
                if not header.startswith(b">"):
                    raise RuntimeError(
                        f"range {index}: expected a header at offset "
                        f"{record_start}, got {header[:60]!r}"
                    )
                match = _HEADER.match(header)
                if match is None:
                    raise RuntimeError(f"range {index}: unparseable header {header[:60]!r}")
                accession = accession_from_token(match.group("token"))
                body: list[bytes] = []
                while True:
                    reader.fill(1)
                    if not reader._buffer and reader._eof:
                        break
                    if reader._buffer[:1] == b">":
                        break
                    line = reader.readline()
                    if not line:
                        break
                    body.append(line)
                records += 1
                if accession in wanted:
                    matched += 1
                    sink.write(header)
                    sink.writelines(body)
                record_start = reader.consumed
                now = time.monotonic()
                if now - last_report >= progress_seconds:
                    done = record_start - first_start
                    span = max(end - first_start, 1)
                    print(
                        f"range {index:>3}: {100 * done / span:5.1f}% "
                        f"({done / 1e9:.2f}/{span / 1e9:.2f} GB) "
                        f"{records:,} records, {matched:,} matched, "
                        f"{done / 1e6 / max(now - began, 1e-9):.1f} MB/s",
                        flush=True,
                    )
                    last_report = now
    finally:
        reader.close()
    return ShardResult(
        index, records, matched, reader.consumed - start, time.monotonic() - began
    )


def fetch(
    normalized_glob: str | None,
    out_dir: str | Path,
    *,
    url: str = SEQUENCES_URL,
    workers: int = 16,
    accession_list: str | None = None,
    byte_limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Filter the bulk FASTA down to AFCDB's accessions, in parallel ranges.

    One pod runs at ~14.5 MB/s because the scan is Python-bound, not
    network-bound, so the whole file takes hours on a single worker.
    ``--shard-count`` splits the file across pods: shard *i* owns the global
    window ``[total*i/N, total*(i+1)/N)`` and subdivides it into ``workers``
    ranges. The same start-offset ownership rule applies at both levels, so
    pod seams are exact for the same reason range seams are.

    ``byte_limit`` restricts the scan to the first N bytes of the source. It is
    for smoke tests only and makes the completeness check meaningless, so it
    disables it explicitly rather than letting a partial run look complete.
    A sharded run cannot check completeness either — only the union can — so it
    defers to ``verify``.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for {shard_count}")
    out_dir = str(out_dir).rstrip("/")
    fs, _ = fsspec.core.url_to_fs(out_dir)
    fs.makedirs(out_dir, exist_ok=True)
    loading = time.monotonic()
    if accession_list:
        wanted = listed_accessions(accession_list)
    elif normalized_glob:
        wanted = required_accessions(normalized_glob)
    else:
        raise ValueError("one of --accessions or --normalized is required")
    if not wanted:
        raise ValueError("no accessions to fetch")
    print(
        f"loaded {len(wanted):,} accessions in {time.monotonic() - loading:.1f}s",
        flush=True,
    )
    total = content_length(url)
    scan = min(total, byte_limit) if byte_limit else total
    window_start = round(scan * shard_index / shard_count)
    window_end = round(scan * (shard_index + 1) / shard_count)
    span = window_end - window_start
    bounds = [window_start + round(span * i / workers) for i in range(workers + 1)]
    shard_paths = [
        f"{out_dir}/afcdb_sequences-{shard_index:04d}-{i:04d}.fasta.gz"
        for i in range(workers)
    ]

    print(
        f"shard {shard_index + 1}/{shard_count}: scanning bytes "
        f"[{window_start:,}, {window_end:,}) = {span / 1e9:.1f} GB of {url} "
        f"across {workers} ranges for {len(wanted):,} accessions",
        flush=True,
    )
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_scan_range, url, wanted, i, bounds[i], bounds[i + 1], shard_paths[i])
            for i in range(workers)
        ]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - began

    records = sum(r.records for r in results)
    matched = sum(r.matched for r in results)
    read = sum(r.bytes_read for r in results)
    summary = {
        "source_url": url,
        "source_bytes": total,
        "scanned_bytes": scan,
        "complete_scan": scan == total,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "window_start": window_start,
        "window_end": window_end,
        "bytes_read": read,
        "workers": workers,
        "seconds": round(elapsed, 1),
        "throughput_mb_s": round(read / 1e6 / max(elapsed, 1e-9), 2),
        "source_records": records,
        "requested_accessions": len(wanted),
        "matched_accessions": matched,
        "shards": shard_paths,
        "per_shard": [
            {
                "index": r.index,
                "records": r.records,
                "matched": r.matched,
                "bytes_read": r.bytes_read,
                "seconds": round(r.seconds, 1),
            }
            for r in results
        ],
    }
    with fsspec.open(
        f"{out_dir}/fetch_sequences-{shard_index:04d}.json", "w"
    ) as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    if byte_limit:
        print("byte_limit set: skipping the completeness check (smoke run)")
        return summary
    if shard_count > 1:
        print(f"shard {shard_index}/{shard_count} done; run `verify` on the union")
        return summary
    if matched != len(wanted):
        raise RuntimeError(
            f"source provided {matched:,} of {len(wanted):,} requested accessions; "
            f"{len(wanted) - matched:,} are missing. Wrote {out_dir} for inspection; "
            "an incomplete sequence set would silently under-decontaminate the corpus"
        )
    return summary


def verify(out_dir: str, *, expected_accessions: int | None = None) -> dict[str, Any]:
    """Check that the shard summaries tile the whole file and miss nothing.

    A sharded fetch can only be judged as a union: each pod sees a fraction of
    the accessions and none can tell on its own whether the set is complete.
    This asserts the windows are contiguous from 0 to the source length and
    that every requested accession was matched exactly once across shards.
    """
    out_dir = out_dir.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(out_dir)
    paths = sorted(fs.glob(f"{out_dir}/fetch_sequences-*.json"))
    if not paths:
        raise FileNotFoundError(f"no shard summaries under {out_dir}")

    summaries = []
    for path in paths:
        with fs.open(path, "r") as handle:
            summaries.append(json.load(handle))
    summaries.sort(key=lambda s: s["shard_index"])

    count = summaries[0]["shard_count"]
    if len(summaries) != count:
        raise RuntimeError(
            f"{out_dir}: found {len(summaries)} shard summaries, expected {count}"
        )
    if any(s["shard_count"] != count for s in summaries):
        raise RuntimeError(f"{out_dir}: shards disagree on shard_count")
    if any(not s["complete_scan"] for s in summaries):
        raise RuntimeError(f"{out_dir}: a shard was byte-limited; this is a smoke run")

    source_bytes = summaries[0]["source_bytes"]
    cursor = 0
    for summary in summaries:
        if summary["window_start"] != cursor:
            raise RuntimeError(
                f"shard {summary['shard_index']} starts at "
                f"{summary['window_start']:,}, expected {cursor:,}: the windows "
                "do not tile the file and records fell through a gap"
            )
        cursor = summary["window_end"]
    if cursor != source_bytes:
        raise RuntimeError(f"windows end at {cursor:,}, source is {source_bytes:,}")

    requested = summaries[0]["requested_accessions"]
    matched = sum(s["matched_accessions"] for s in summaries)
    result = {
        "out_dir": out_dir,
        "shards": count,
        "source_bytes": source_bytes,
        "source_records": sum(s["source_records"] for s in summaries),
        "requested_accessions": requested,
        "matched_accessions": matched,
        "seconds_max_shard": max(s["seconds"] for s in summaries),
        "throughput_mb_s_total": round(
            source_bytes / 1e6 / max(max(s["seconds"] for s in summaries), 1e-9), 1
        ),
        "complete": matched == requested,
    }
    print(json.dumps(result, indent=2))
    if expected_accessions is not None and requested != expected_accessions:
        raise RuntimeError(
            f"shards were built for {requested:,} accessions, expected "
            f"{expected_accessions:,}"
        )
    if matched != requested:
        raise RuntimeError(
            f"union matched {matched:,} of {requested:,} accessions; "
            f"{requested - matched:,} are missing. An incomplete sequence set "
            "would silently under-decontaminate the corpus"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", help="Normalized Parquet or glob.")
    parser.add_argument(
        "--accessions", help="Parquet with an 'accession' column (pod-friendly)."
    )
    parser.add_argument("--out", required=True, help="Directory or fsspec URL.")
    parser.add_argument("--url", default=SEQUENCES_URL)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--byte-limit", type=int, default=None, help="Smoke only: scan the first N bytes."
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check an existing sharded fetch under --out instead of fetching.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        verify(args.out)
        return 0
    summary = fetch(
        args.normalized,
        args.out,
        url=args.url,
        workers=args.workers,
        accession_list=args.accessions,
        byte_limit=args.byte_limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
