"""Bound checkpoint storage waits before another rank's NCCL watchdog expires.

The cluster's S3 settings allow five botocore retries per request, each with a
120-second read timeout, nested inside s3fs retries. A stalled upload can thus
outlive the ten-minute collective timeout on ranks waiting for its completion.
Use fsspec's coroutine deadline as an outer bound shared by the checkpoint.
"""

import json
import time
from typing import Any

from fsspec.asyn import AsyncFileSystem, FSTimeoutError


class CheckpointIO:
    """Apply one wall-clock budget to a checkpoint's idempotent S3 operations."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        budget_seconds: float = 480,
        operation_seconds: float = 90,
        attempts: int = 3,
    ) -> None:
        if budget_seconds <= 0 or operation_seconds <= 0 or attempts < 1:
            raise ValueError("Checkpoint deadlines and attempt count must be positive")
        self.fs = fs
        self.deadline = time.monotonic() + budget_seconds
        self.operation_seconds = operation_seconds
        self.attempts = attempts

    def call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Retry timed-out requests within a shared budget; propagate other errors."""
        for attempt in range(1, self.attempts + 1):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Checkpoint storage budget exhausted before {operation}"
                )
            began = time.monotonic()
            print(
                json.dumps(
                    {"checkpoint_io": operation, "attempt": attempt, "state": "started"}
                ),
                flush=True,
            )
            try:
                result = getattr(self.fs, operation)(
                    *args, timeout=min(self.operation_seconds, remaining), **kwargs
                )
            except FSTimeoutError as error:
                print(
                    json.dumps(
                        {
                            "checkpoint_io": operation,
                            "attempt": attempt,
                            "state": "timeout",
                        }
                    ),
                    flush=True,
                )
                if attempt == self.attempts:
                    raise TimeoutError(
                        f"Checkpoint {operation} timed out after {attempt} attempts"
                    ) from error
            else:
                print(
                    json.dumps(
                        {
                            "checkpoint_io": operation,
                            "state": "finished",
                            "seconds": time.monotonic() - began,
                        }
                    ),
                    flush=True,
                )
                return result
        raise AssertionError("Unreachable retry state")
