# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tar walking must find the members the manifest names, over real HTTP."""

import http.server
import io
import socketserver
import sys
import tarfile
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from extract import MEMBER_TEMPLATE, http_range, walk_members

#: The member name of a real AFCDB model, checked by hand against
#: heterodimers/shard_1003_batch_2.tar. `model_id` already carries the `AF-`
#: prefix, so wrapping it again produced `AF-AF-...` and a full-archive walk
#: that found nothing -- 20 tars and 197,333 header reads for zero documents.
REAL_MODEL_ID = "AF-0000000207677021"
REAL_MEMBER = "AF-0000000207677021-model_v1.cif.zst"


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        header = self.headers.get("Range")
        start, end = 0, len(self.payload) - 1
        if header:
            lo, _, hi = header.removeprefix("bytes=").partition("-")
            start = int(lo)
            end = int(hi) if hi else len(self.payload) - 1
        body = self.payload[start : end + 1]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if header:
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(self.payload)}"
            )
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def tar_server(request: pytest.FixtureRequest):
    """Serve a synthetic AFCDB-shaped tar with range support."""
    buffer = io.BytesIO()
    names = [f"AF-000000020000{i:04d}-model_v1.cif.zst" for i in range(12)]
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for index, name in enumerate(names):
            # Vary payload sizes so header offsets are not a fixed stride.
            data = bytes((index * 7 + j) % 251 for j in range(500 + index * 313))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            if index % 3 == 0:  # homodimer chunk_* tars also carry a .pdb.zst
                extra = b"x" * 64
                pdb = tarfile.TarInfo(name.replace(".cif.zst", ".pdb.zst"))
                pdb.size = len(extra)
                tar.addfile(pdb, io.BytesIO(extra))
    _RangeHandler.payload = buffer.getvalue()

    server = socketserver.TCPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request.addfinalizer(lambda: (server.shutdown(), server.server_close()))
    return f"http://127.0.0.1:{server.server_address[1]}/probe.tar", names


def test_member_template_matches_a_real_afcdb_member_name() -> None:
    """Pins the exact name against one verified by hand in the live archive."""
    assert MEMBER_TEMPLATE.format(model_id=REAL_MODEL_ID) == REAL_MEMBER


def test_walk_locates_every_requested_member(tar_server) -> None:
    url, names = tar_server
    wanted = {names[1], names[4], names[9]}
    reads: list[int] = []
    found = walk_members(url, wanted, header_reads=reads)
    assert set(found) == wanted
    for name, (offset, size) in found.items():
        payload = http_range(url, offset, size)
        expected = 500 + names.index(name) * 313
        assert len(payload) == size == expected, f"{name} payload is wrong"


def test_walk_stops_once_everything_is_found(tar_server) -> None:
    """The walk is the per-tar cost, so it must not run past the last hit."""
    url, names = tar_server
    early: list[int] = []
    walk_members(url, {names[0]}, header_reads=early)
    late: list[int] = []
    walk_members(url, {names[-1]}, header_reads=late)
    assert early[0] < late[0], "an early member should cost fewer header reads"


def test_a_name_that_is_not_in_the_tar_is_reported_not_invented(tar_server) -> None:
    """The AF-AF- bug looked exactly like this: a full walk returning nothing."""
    url, names = tar_server
    reads: list[int] = []
    found = walk_members(url, {f"AF-{names[0]}"}, header_reads=reads)
    assert found == {}
    assert reads[0] > len(names), "a miss walks the whole tar"


def test_a_shared_zstd_decompressor_corrupts_under_threads() -> None:
    """Pins the root cause of three failed probe runs.

    Sharing one ZstdDecompressor across the fetch pool corrupts its C-level
    state. Locally that surfaces as "Unknown frame descriptor"; on a pod it
    surfaced as SIGSEGV with no Python traceback, which is why it took three
    runs to find. extract.py therefore caches the *module* and builds a
    decompressor per call.
    """
    import os
    import threading

    import zstandard

    payload = zstandard.ZstdCompressor().compress(os.urandom(200_000))
    shared = zstandard.ZstdDecompressor()

    def hammer(sink: list[str], use_shared: bool) -> None:
        for _ in range(40):
            try:
                decompressor = shared if use_shared else zstandard.ZstdDecompressor()
                decompressor.decompress(payload, max_output_size=1 << 22)
            except Exception as error:  # noqa: BLE001 - that is the point
                sink.append(f"{type(error).__name__}: {error}")
                return

    per_call: list[str] = []
    threads = [threading.Thread(target=hammer, args=(per_call, False)) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert per_call == [], "a per-call decompressor must be safe under threads"


def test_extract_does_not_cache_a_decompressor_instance() -> None:
    """The safe shape is structural, so guard it rather than trusting review."""
    import extract

    source = Path(extract.__file__).read_text()
    assert "ZstdDecompressor()" in source, "the module should build them per call"
    assert '_GENERATOR["zstd"] = zstandard.ZstdDecompressor()' not in source
    assert '_GENERATOR["zstandard"] = zstandard' in source


def _manifest(tmp_path: Path, url: str, names: list[str]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "source_model_key": f"homodimer|{n.removesuffix('-model_v1.cif.zst')}",
            "model_id": n.removesuffix("-model_v1.cif.zst"),
            "complex_type": "homodimer",
            "accession_a": "A1",
            "accession_b": "A1",
            "total_residues": 200,
            "quality_ratio": 1.2,
            "ipsae_score": 0.7,
            "pdockq2_score": 0.3,
            "confidence_tier": "A",
            "source_tar_uri": f"{url}?tar={i % 4}",
        }
        for i, n in enumerate(names)
    ]
    path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_shards_stride_and_together_cover_every_tar(tar_server, tmp_path: Path) -> None:
    """Striding balances pods; contiguous slicing would not.

    The sorted tar list groups heterodimer shards, homodimer chunk_* and
    homodimer shard_* together, and those differ by an order of magnitude in
    members per tar, so a contiguous slice hands one pod every 7.5 GB archive.
    """
    import extract

    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)
    seen: list[list[str]] = []
    for index in range(3):
        rows = extract.duckdb.connect().execute(
            f"SELECT DISTINCT source_tar_uri FROM read_parquet('{manifest}') "
            "ORDER BY source_tar_uri"
        ).fetchall()
        all_tars = [r[0] for r in rows]
        seen.append(all_tars[index::3])
    flat = [t for group in seen for t in group]
    assert sorted(flat) == sorted({t for group in seen for t in group})
    assert len(flat) == 4, "every tar is covered exactly once across shards"


def test_shard_index_must_be_in_range() -> None:
    import extract

    with pytest.raises(ValueError, match="out of range"):
        extract.run("unused.parquet", "unused", shard_index=3, shard_count=3)
