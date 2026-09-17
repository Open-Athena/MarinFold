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

    def do_GET(self) -> None:  # noqa: N802
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
