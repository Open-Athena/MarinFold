"""Exercise real fsspec coroutine cancellation without an S3 service."""

import asyncio
import json
import time

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from checkpoint_io import CheckpointIO, load_resume_marker


def test_relocation_uses_local_successor_after_original_source_is_removed() -> None:
    fs = MemoryFileSystem()
    source, root = "s3://east/migration-test/step-10", "s3://rno/migration-test"
    original = {"uri": source, "step": 10, "tokens": 123, "world": 8}
    fs.pipe_file(source + "/complete.json", json.dumps(original).encode())
    assert load_resume_marker(fs, root, source) == original
    successor = {**original, "uri": root + "/step-11", "step": 11, "tokens": 456}
    fs.pipe_file(successor["uri"] + "/complete.json", json.dumps(successor).encode())
    fs.pipe_file(root + "/resume.json", json.dumps(successor).encode())
    fs.rm(source, recursive=True)
    assert load_resume_marker(fs, root, source) == successor


def test_relocation_rejects_cross_region_or_uncommitted_local_pointer() -> None:
    fs = MemoryFileSystem()
    root = "s3://rno/migration-invalid"
    marker = {"uri": "s3://east/migration-invalid/step-10", "step": 10}
    fs.pipe_file(root + "/resume.json", json.dumps(marker).encode())
    with pytest.raises(ValueError, match="output region"):
        load_resume_marker(fs, root, None)
    marker["uri"] = root + "/step-10"
    fs.pipe_file(root + "/resume.json", json.dumps(marker).encode())
    fs.pipe_file(
        marker["uri"] + "/complete.json", json.dumps({**marker, "step": 9}).encode()
    )
    with pytest.raises(ValueError, match="differs"):
        load_resume_marker(fs, root, None)


class DelayedStorage(AsyncFileSystem):
    """In-memory async storage with configurable stuck writes."""

    cachable = False

    def __init__(self, delays: list[float], *, fail: bool = False) -> None:
        super().__init__()
        self.delays = iter(delays)
        self.fail = fail
        self.calls = 0
        self.cancelled = 0
        self.objects: dict[str, bytes] = {}

    async def _pipe_file(self, path: str, value: bytes, **kwargs: object) -> None:
        self.calls += 1
        if self.fail:
            raise PermissionError("write denied")
        try:
            await asyncio.sleep(next(self.delays))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.objects[path] = value


def test_stuck_upload_is_cancelled_and_retried() -> None:
    fs = DelayedStorage([10, 0])
    io = CheckpointIO(fs, budget_seconds=5, operation_seconds=0.05)
    io.call("pipe_file", "model", b"weights")
    assert fs.objects == {"model": b"weights"}
    assert fs.calls == 2
    assert fs.cancelled == 1


def test_permanent_stall_fails_without_advancing_pointer() -> None:
    fs = DelayedStorage([10, 10, 10])
    fs.objects["resume.json"] = b"previous complete checkpoint"
    io = CheckpointIO(fs, budget_seconds=5, operation_seconds=0.05)
    began = time.monotonic()
    with pytest.raises(TimeoutError, match="3 attempts"):
        io.call("pipe_file", "model", b"new weights")
        io.call("pipe_file", "resume.json", b"new checkpoint")
    assert time.monotonic() - began < 3
    assert fs.objects == {"resume.json": b"previous complete checkpoint"}
    assert fs.cancelled == 3


def test_checkpoint_budget_bounds_all_retries() -> None:
    fs = DelayedStorage([10] * 20)
    io = CheckpointIO(fs, budget_seconds=0.1, operation_seconds=0.05, attempts=20)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        io.call("pipe_file", "model", b"weights")
    assert fs.calls < 20
    assert not fs.objects


def test_permission_error_is_not_retried() -> None:
    fs = DelayedStorage([], fail=True)
    io = CheckpointIO(fs)
    with pytest.raises(PermissionError, match="write denied"):
        io.call("pipe_file", "model", b"weights")
    assert fs.calls == 1
