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

    uv run python fetch_sequences.py \\
        --normalized '/data/exp294_predicted_complexes/metadata/normalized_*.parquet' \\
        --out /data/exp294_predicted_complexes/sequences \\
        --workers 16
"""

import argparse
import gzip
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

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


def required_accessions(normalized_glob: str) -> set[bytes]:
    """Every UniProt accession either subunit of any AFCDB model refers to."""
    con = duckdb.connect()
    src = f"read_parquet({_sql_literal(normalized_glob)}, union_by_name=true)"
    rows = con.execute(
        f"""
        SELECT DISTINCT accession FROM (
            SELECT accession_a AS accession FROM {src}
            UNION ALL SELECT accession_b FROM {src}
        ) WHERE accession IS NOT NULL
        """
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
    out_path: Path,
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

        with gzip.open(out_path, "wb", compresslevel=6) as sink:
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
    finally:
        reader.close()
    return ShardResult(
        index, records, matched, reader.consumed - start, time.monotonic() - began
    )


def fetch(
    normalized_glob: str,
    out_dir: Path,
    *,
    url: str = SEQUENCES_URL,
    workers: int = 16,
) -> dict[str, Any]:
    """Filter the bulk FASTA down to AFCDB's accessions, in parallel ranges."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = required_accessions(normalized_glob)
    if not wanted:
        raise ValueError(f"{normalized_glob}: no accessions to fetch")
    total = content_length(url)
    bounds = [round(total * i / workers) for i in range(workers + 1)]
    shard_paths = [out_dir / f"afcdb_sequences-{i:04d}.fasta.gz" for i in range(workers)]

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
        "bytes_read": read,
        "workers": workers,
        "seconds": round(elapsed, 1),
        "throughput_mb_s": round(read / 1e6 / max(elapsed, 1e-9), 2),
        "source_records": records,
        "requested_accessions": len(wanted),
        "matched_accessions": matched,
        "shards": [str(path.resolve()) for path in shard_paths],
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
    (out_dir / "fetch_sequences.json").write_text(json.dumps(summary, indent=2) + "\n")
    if matched != len(wanted):
        raise RuntimeError(
            f"source provided {matched:,} of {len(wanted):,} requested accessions; "
            f"{len(wanted) - matched:,} are missing. Wrote {out_dir} for inspection; "
            "an incomplete sequence set would silently under-decontaminate the corpus"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", required=True, help="Normalized Parquet or glob.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=SEQUENCES_URL)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = fetch(args.normalized, args.out, url=args.url, workers=args.workers)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
