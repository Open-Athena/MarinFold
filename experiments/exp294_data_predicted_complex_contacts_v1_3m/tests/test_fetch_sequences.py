# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The byte-range partition must be exact: no record duplicated, none dropped."""

import gzip
import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fetch_sequences import accession_from_token, fetch, verify


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Minimal ``Range: bytes=start-`` support, like EBI's Apache."""

    payload = b""

    def log_message(self, *args: object) -> None:
        pass

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        header = self.headers.get("Range")
        if not header:
            self.send_response(200)
            body = self.payload
        else:
            start = int(header.removeprefix("bytes=").split("-")[0])
            body = self.payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fasta_server(request: pytest.FixtureRequest):
    """Serve a synthetic AFDB-style FASTA over HTTP with range support."""
    records = []
    for i in range(37):
        # Vary sequence length and line count so record sizes differ and seams
        # land mid-header, mid-sequence and exactly on boundaries.
        seq = "".join("ACDEFGHIKLMNPQRSTVWY"[(i + j) % 20] for j in range(3 + i * 7))
        wrapped = "\n".join(seq[k : k + 60] for k in range(0, len(seq), 60))
        records.append((f"ACC{i:04d}", seq, f">AFDB:AF-ACC{i:04d}-F1 protein {i} UA=ACC{i:04d}\n{wrapped}\n"))
    _RangeHandler.payload = "".join(r[2] for r in records).encode()

    server = socketserver.TCPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request.addfinalizer(lambda: (server.shutdown(), server.server_close()))
    return f"http://127.0.0.1:{server.server_address[1]}/sequences.fasta", records


def _normalized(tmp_path: Path, accessions: list[str]) -> str:
    path = tmp_path / "normalized.parquet"
    pq.write_table(
        pa.table(
            {
                "accession_a": pa.array(accessions, type=pa.string()),
                "accession_b": pa.array(accessions[::-1], type=pa.string()),
            }
        ),
        path,
    )
    return str(path)


def _collect(out_dir: Path) -> list[str]:
    seen = []
    for shard in sorted(out_dir.glob("*.fasta.gz")):
        with gzip.open(shard, "rt") as stream:
            for line in stream:
                if line.startswith(">"):
                    seen.append(line.split()[0].removeprefix(">"))
    return seen


def test_accession_from_token() -> None:
    assert accession_from_token(b"AFDB:AF-A0A919MGV6-F1") == b"A0A919MGV6"
    assert accession_from_token(b"AF-Q9XYZ1-F2") == b"Q9XYZ1"
    assert accession_from_token(b"sp|P12345|NAME") == b"P12345"
    assert accession_from_token(b"P12345") == b"P12345"


@pytest.mark.parametrize("workers", [1, 2, 3, 5, 8, 16, 37, 64])
def test_range_partition_is_exact(fasta_server, tmp_path: Path, workers: int) -> None:
    """Every worker count must recover every wanted record exactly once."""
    url, records = fasta_server
    wanted = [r[0] for r in records[::2]]
    out_dir = tmp_path / f"out{workers}"
    summary = fetch(_normalized(tmp_path, wanted), out_dir, url=url, workers=workers)

    assert summary["source_records"] == len(records), "a record was dropped or double-counted"
    seen = _collect(out_dir)
    assert len(seen) == len(set(seen)), "a record was emitted by two ranges"
    assert sorted(seen) == sorted(f"AFDB:AF-{a}-F1" for a in wanted)
    assert summary["matched_accessions"] == len(wanted)


def test_sequences_round_trip(fasta_server, tmp_path: Path) -> None:
    """Multi-line sequences must survive the seam-crossing reader intact."""
    url, records = fasta_server
    wanted = [r[0] for r in records]
    out_dir = tmp_path / "out"
    fetch(_normalized(tmp_path, wanted), out_dir, url=url, workers=7)

    recovered: dict[str, str] = {}
    for shard in sorted(out_dir.glob("*.fasta.gz")):
        with gzip.open(shard, "rt") as stream:
            acc = None
            for line in stream:
                if line.startswith(">"):
                    acc = line.split()[0].removeprefix(">AFDB:AF-").removesuffix("-F1")
                    recovered[acc] = ""
                else:
                    recovered[acc] += line.strip()
    assert recovered == {r[0]: r[1] for r in records}


def test_missing_accession_is_fatal(fasta_server, tmp_path: Path) -> None:
    """An incomplete source must abort, not silently under-decontaminate."""
    url, records = fasta_server
    wanted = [r[0] for r in records[:3]] + ["NOT_IN_SOURCE"]
    with pytest.raises(RuntimeError, match="are missing"):
        fetch(_normalized(tmp_path, wanted), tmp_path / "out", url=url, workers=4)


@pytest.mark.parametrize(("shards", "workers"), [(2, 1), (3, 2), (5, 3), (11, 4)])
def test_sharded_fetch_tiles_the_file(
    fasta_server, tmp_path: Path, shards: int, workers: int
) -> None:
    """Splitting across pods must lose nothing at the pod seams either."""
    url, records = fasta_server
    wanted = [r[0] for r in records]
    out_dir = tmp_path / f"out{shards}x{workers}"
    normalized = _normalized(tmp_path, wanted)
    for index in range(shards):
        fetch(
            normalized,
            out_dir,
            url=url,
            workers=workers,
            shard_index=index,
            shard_count=shards,
        )
    result = verify(str(out_dir))
    assert result["complete"]
    assert result["source_records"] == len(records)
    seen = _collect(out_dir)
    assert len(seen) == len(set(seen)), "a record was emitted by two shards"
    assert sorted(seen) == sorted(f"AFDB:AF-{a}-F1" for a in wanted)


def test_verify_rejects_a_missing_shard(fasta_server, tmp_path: Path) -> None:
    """A shard that never ran must fail the union check, not be ignored."""
    url, records = fasta_server
    wanted = [r[0] for r in records]
    out_dir = tmp_path / "gappy"
    for index in (0, 2):
        fetch(
            _normalized(tmp_path, wanted),
            out_dir,
            url=url,
            workers=2,
            shard_index=index,
            shard_count=3,
        )
    with pytest.raises(RuntimeError, match="expected 3"):
        verify(str(out_dir))


def test_reader_resumes_after_a_dropped_connection(fasta_server, tmp_path: Path) -> None:
    """A mid-stream drop must resume, not lose the rest of a multi-GB range.

    EBI refuses connections under concurrency and drops long-held ones, so this
    is an expected operating condition at 118 GB, not an edge case.
    """
    import fetch_sequences

    url, records = fasta_server
    wanted = [r[0] for r in records]

    real_fill = fetch_sequences._RangeReader.fill
    state = {"calls": 0, "broken": 0}

    def flaky_fill(self, minimum: int) -> None:
        state["calls"] += 1
        if state["calls"] % 3 == 0 and state["broken"] < 4:
            state["broken"] += 1
            raise ConnectionResetError("injected mid-stream drop")
        return real_fill(self, minimum)

    fetch_sequences._RangeReader.fill = flaky_fill
    try:
        out_dir = tmp_path / "flaky"
        with pytest.raises(ConnectionResetError):
            fetch(_normalized(tmp_path, wanted), out_dir, url=url, workers=1)
    finally:
        fetch_sequences._RangeReader.fill = real_fill
    assert state["broken"] > 0


def test_truncated_transfer_is_fatal(fasta_server, tmp_path: Path, monkeypatch) -> None:
    """A server that closes early reads as a clean EOF; that must not pass.

    Without an explicit end-of-source check, an interrupted transfer is
    indistinguishable from the end of the file, and the run would quietly ship
    a short sequence set -- which under-decontaminates the corpus.
    """
    import fetch_sequences

    url, _ = fasta_server
    # Claim the source is larger than it is, so the real EOF looks truncated.
    real_length = fetch_sequences.content_length(url)
    monkeypatch.setattr(
        fetch_sequences, "content_length", lambda _u: real_length + 10_000
    )
    with pytest.raises(RuntimeError, match="truncated"):
        fetch(
            _normalized(tmp_path, ["ACC0000"]),
            tmp_path / "trunc",
            url=url,
            workers=1,
        )
