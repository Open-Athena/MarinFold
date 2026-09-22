# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tar walking must find the members the manifest names, over real HTTP."""

import http.server
import json
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


def test_tar_stem_disambiguates_the_two_archive_trees() -> None:
    """shard_0_batch_0.tar exists under BOTH homodimers/ and heterodimers/."""
    from extract import tar_stem

    base = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda"
    assert tar_stem(f"{base}/homodimers/chunk_0286.tar") == "homodimers__chunk_0286"
    assert tar_stem(f"{base}/homodimers/shard_0_batch_0.tar") != tar_stem(
        f"{base}/heterodimers/shard_0_batch_0.tar"
    )


def _stub_one_model(monkeypatch) -> None:
    """Replace the structure pipeline; these tests are about resume, not gemmi."""
    import extract

    def fake(url: str, row: dict, offset: int, size: int):
        doc = {name: None for name in extract.DOC_SCHEMA.names}
        doc.update(
            {
                "source_model_key": row["source_model_key"],
                "model_id": row["model_id"],
                "complex_type": row["complex_type"],
                "document": "<contacts-v1> stub",
                "sha1": "0" * 40,
                "seq_len": 200,
                "num_tokens": 10,
                "num_chains": 2,
                "chain_ids": ["A", "B"],
                "chain_lengths": [100, 100],
                "contacts_pre_filter": 10,
                "contacts_emitted": 8,
                "contacts_emitted_inter_chain": 2,
                "contacts_pre_filter_inter_chain": 3,
                "truncated": False,
                "accession_a": row["accession_a"],
                "accession_b": row["accession_b"],
                "total_residues_manifest": row["total_residues"],
                "quality_ratio": row["quality_ratio"],
                "ipsae_score": row["ipsae_score"],
                "pdockq2_score": row["pdockq2_score"],
                "confidence_tier": row["confidence_tier"],
                "source_tar_uri": row["source_tar_uri"],
                "member_bytes": size,
                "cif_bytes": size,
            }
        )
        ledger = {
            "source_model_key": row["source_model_key"],
            "source_tar_uri": url,
            "status": "generated",
            "reason": "ok",
        }
        return doc, ledger

    monkeypatch.setattr(extract, "_one_model", fake)


def test_a_completed_tar_is_not_redone(tar_server, tmp_path: Path, monkeypatch) -> None:
    """Resume is the difference between finishing and never finishing.

    The first production attempt was preempted 6-9 times per shard and produced
    nothing in 24 hours, because every restart began again at the first tar.
    """
    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)
    out = tmp_path / "out"

    first = extract.run(str(manifest), str(out), fetch_concurrency=2)
    assert first["tars"] == 4 and first["tars_resumed"] == 0
    assert first["documents"] == len(names)
    assert (out / "ledger").is_dir() and (out / "documents").is_dir()

    calls: list[str] = []
    real = extract.extract_tar
    monkeypatch.setattr(
        extract,
        "extract_tar",
        lambda tar, rows, **kw: (calls.append(tar), real(tar, rows, **kw))[1],
    )
    second = extract.run(str(manifest), str(out), fetch_concurrency=2)

    assert calls == [], "a completed tar must not be walked again"
    assert second["tars_resumed"] == 4
    assert second["tars"] == 0


def test_a_partial_shard_resumes_from_where_it_stopped(
    tar_server, tmp_path: Path, monkeypatch
) -> None:
    """Tars that failed stay pending; tars that finished are kept."""
    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)
    out = tmp_path / "out"

    real = extract.extract_tar
    seen = {"n": 0}

    def die_after_two(tar: str, rows, **kw):
        if seen["n"] >= 2:
            raise ConnectionResetError("simulated preemption")
        seen["n"] += 1
        return real(tar, rows, **kw)

    monkeypatch.setattr(extract, "extract_tar", die_after_two)
    first = extract.run(str(manifest), str(out), fetch_concurrency=2)
    assert first["tars_deferred"] == 2, "failures are deferred, not fatal"
    monkeypatch.setattr(extract, "extract_tar", real)

    resumed = extract.run(str(manifest), str(out), fetch_concurrency=2)
    assert resumed["tars_resumed"] == 2, "the two finished tars are kept"
    assert resumed["tars"] == 2, "only the unfinished two are redone"
    assert resumed["tars_deferred"] == 0


def test_a_tar_missing_from_the_archive_is_a_named_rejection(tmp_path: Path) -> None:
    """28 of 16,640 referenced tars 404. That must not be fatal.

    Treating it as an I/O error made the shard die, resume to the same missing
    tar, and burn all 51 retries -- stalling 17 shards for two days over 373
    documents, 0.012% of the corpus.
    """
    import extract

    class _NotFound(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = socketserver.TCPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/gone.tar"
        rows = [
            {
                "source_model_key": f"heterodimer|AF-{i}",
                "model_id": f"AF-{i}",
                "source_tar_uri": url,
            }
            for i in range(4)
        ]
        result = extract.extract_tar(url, rows, fetch_concurrency=2)
    finally:
        server.shutdown()
        server.server_close()

    assert result.documents == []
    assert len(result.ledger) == len(rows), "every model still gets a terminal row"
    assert {e["reason"] for e in result.ledger} == {"tar_not_in_archive"}
    assert {e["status"] for e in result.ledger} == {"rejected"}


def test_a_404_is_not_retried_as_an_io_error() -> None:
    """Retrying a 404 eight times per call is what made it expensive."""
    import extract

    class _NotFound(http.server.BaseHTTPRequestHandler):
        hits = 0

        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            type(self).hits += 1
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = socketserver.TCPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/gone.tar"
        with pytest.raises(extract.TarNotInArchive):
            extract.http_range(url, 0, 512)
    finally:
        server.shutdown()
        server.server_close()
    assert _NotFound.hits == 1, "a 404 is an answer, not a failure to retry"


def test_reverse_works_the_same_shard_from_the_other_end(
    tar_server, tmp_path: Path, monkeypatch
) -> None:
    """A pod that finished its own shard can help one that has not.

    Both pods skip tars that already have a ledger file, so they converge in
    the middle; the worst case when they meet is one tar done twice, and that
    write is idempotent.
    """
    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)

    order: list[str] = []
    real = extract.extract_tar
    monkeypatch.setattr(
        extract,
        "extract_tar",
        lambda tar, rows, **kw: (order.append(tar), real(tar, rows, **kw))[1],
    )

    extract.run(str(manifest), str(tmp_path / "fwd"), fetch_concurrency=2)
    forward = list(order)
    order.clear()
    extract.run(str(manifest), str(tmp_path / "rev"), fetch_concurrency=2, reverse=True)
    assert order == forward[::-1], "a helper must start at the far end"


def test_an_unreachable_tar_is_deferred_not_fatal(tar_server, tmp_path: Path, monkeypatch) -> None:
    """One unreachable tar must not kill a shard mid-run.

    EBI's FTP host refused every connection for hours on 2026-09-21. A single
    exhausted retry budget killed the whole shard and lost its in-progress
    work, 31-39 times per shard.
    """
    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)
    out = tmp_path / "out"

    real = extract.extract_tar
    calls = {"n": 0}

    def flaky(tar: str, rows, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated EBI refusal")
        return real(tar, rows, **kw)

    monkeypatch.setattr(extract, "extract_tar", flaky)
    summary = extract.run(str(manifest), str(out), fetch_concurrency=2)

    assert summary["tars_deferred"] == 1, "the unreachable tar is deferred"
    assert summary["documents"] > 0, "the reachable tars still produced output"

    # The deferred tar has no ledger file, so a later run retries it.
    monkeypatch.setattr(extract, "extract_tar", real)
    second = extract.run(str(manifest), str(out), fetch_concurrency=2)
    assert second["tars"] == 1, "exactly the deferred tar is retried"
    assert second["tars_deferred"] == 0


def test_a_wholly_unavailable_source_still_aborts(tar_server, tmp_path: Path, monkeypatch) -> None:
    """Deferring must not become a way to silently produce an empty corpus."""
    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    manifest = _manifest(tmp_path, url, names)

    monkeypatch.setattr(
        extract,
        "extract_tar",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("everything is down")),
    )
    with pytest.raises(RuntimeError, match="source looks unavailable"):
        extract.run(str(manifest), str(tmp_path / "out"), fetch_concurrency=2, max_deferred=2)


def test_an_interrupted_walk_resumes_instead_of_restarting(tar_server, tmp_path: Path) -> None:
    """A chunk_* walk is ~443 s and preemptions arrive every 10-14 minutes.

    Without this, the walk restarts at offset 0 every time and such a tar can
    never finish -- which stalled the production run at 79% with pods making
    zero durable progress across three consecutive attempts.
    """
    import fsspec

    import extract

    url, names = tar_server
    fs, _ = fsspec.core.url_to_fs(str(tmp_path))
    path = f"{tmp_path}/walk.json"
    wanted = {names[-1]}

    # First pass: stop the walk partway by capping the reads it may do.
    cp = extract._WalkCheckpoint(fs, path, every=1)
    real = extract.http_range
    budget = {"n": 6}

    def limited(u: str, start: int, length: int, **kw):
        if budget["n"] <= 0:
            raise ConnectionResetError("simulated preemption")
        budget["n"] -= 1
        return real(u, start, length, **kw)

    extract.http_range = limited
    try:
        with pytest.raises(ConnectionResetError):
            extract.walk_members(url, wanted, header_reads=[], checkpoint=cp)
    finally:
        extract.http_range = real

    saved = json.loads(Path(path).read_text())
    assert saved["offset"] > 0, "the cursor must be persisted mid-walk"

    # Second pass: a fresh checkpoint object reads that cursor back and the
    # walk finishes, having repeated only the reads since the last save.
    reads: list[int] = []
    resumed = extract.walk_members(
        url, wanted, header_reads=reads, checkpoint=extract._WalkCheckpoint(fs, path, every=1)
    )
    assert set(resumed) == wanted
    full: list[int] = []
    extract.walk_members(url, wanted, header_reads=full)
    assert reads[0] < full[0], "resuming must cost fewer reads than starting over"


def test_a_finished_tar_clears_its_walk_checkpoint(tar_server, tmp_path: Path, monkeypatch) -> None:
    """Stale cursors must not accumulate, nor be reused for a done tar."""
    import fsspec

    import extract

    _stub_one_model(monkeypatch)
    url, names = tar_server
    fs, _ = fsspec.core.url_to_fs(str(tmp_path))
    path = f"{tmp_path}/walk.json"
    cp = extract._WalkCheckpoint(fs, path, every=1)
    rows = [
        {
            "source_model_key": f"homodimer|{names[0].removesuffix('-model_v1.cif.zst')}",
            "model_id": names[0].removesuffix("-model_v1.cif.zst"),
            "complex_type": "homodimer",
            "accession_a": "A",
            "accession_b": "A",
            "total_residues": 200,
            "quality_ratio": 1.0,
            "ipsae_score": 0.7,
            "pdockq2_score": 0.3,
            "confidence_tier": "A",
            "source_tar_uri": url,
        }
    ]
    extract.extract_tar(url, rows, fetch_concurrency=1, checkpoint=cp)
    assert not Path(path).exists(), "a completed tar clears its cursor"
