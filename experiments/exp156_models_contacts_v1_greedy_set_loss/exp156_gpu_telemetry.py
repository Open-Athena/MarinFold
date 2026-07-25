# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Temporary exp156 copy of the durable JSONL/GPU telemetry helpers.

Remove this file after marin-community/marin#7641 lands in the published Marin package.
"""

import dataclasses
import contextlib
import csv
import datetime as dt
import json
import logging
import multiprocessing
import posixpath
import queue
import select
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventType
from enum import StrEnum
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

_DONE = object()


class BackpressurePolicy(StrEnum):
    """Queue behavior when the writer actor is full."""

    BLOCK = "block"
    DROP = "drop"


@dataclass(frozen=True)
class JsonlChunkWriterConfig:
    """Configuration for :class:`JsonlChunkWriter`.

    Args:
        output_uri: Destination directory for ``parts/part-*.jsonl`` and
            ``manifest.json``.
        records_per_chunk: Number of records per durable chunk file.
        max_queue_items: Maximum queued records before backpressure policy is
            applied.
        backpressure_policy: Whether ``write`` blocks or drops when the queue is
            full.
        log_every: Log writer progress after this many enqueued records.
    """

    output_uri: str
    records_per_chunk: int = 120
    max_queue_items: int = 10_000
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.BLOCK
    log_every: int = 1_000

    def __post_init__(self) -> None:
        if not self.output_uri:
            raise ValueError("output_uri must be non-empty")
        if self.records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        if self.max_queue_items <= 0:
            raise ValueError("max_queue_items must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if isinstance(self.backpressure_policy, str):
            object.__setattr__(self, "backpressure_policy", BackpressurePolicy(self.backpressure_policy))
        if not isinstance(self.backpressure_policy, BackpressurePolicy):
            raise ValueError("backpressure_policy must be a BackpressurePolicy")


@dataclass(frozen=True)
class JsonlChunkWriterStats:
    """Snapshot of writer counters."""

    records_enqueued: int
    records_written: int
    records_dropped: int
    chunks_written: int
    bytes_written: int
    max_queue_size_observed: int


class JsonlChunkWriter:
    """JSONL writer backed by a queue and writer thread.

    ``write`` serializes the object and enqueues one JSON line. Remote I/O is
    performed only by the writer thread. Queue backpressure is controlled by
    ``JsonlChunkWriterConfig.backpressure_policy``.
    """

    def __init__(self, config: JsonlChunkWriterConfig):
        self.config = config
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=config.max_queue_items)
        self._thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._chunks: list[dict[str, Any]] = []
        self._records_enqueued = 0
        self._records_written = 0
        self._records_dropped = 0
        self._chunks_written = 0
        self._bytes_written = 0
        self._max_queue_size_observed = 0
        self._closed = False
        self._writer_error: str | None = None

    def __enter__(self) -> "JsonlChunkWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> None:
        """Start the writer actor thread."""
        if self._thread is not None:
            raise RuntimeError("JsonlChunkWriter is already started")
        self._started_at = dt.datetime.now(dt.UTC).isoformat()
        self._thread = threading.Thread(target=self._run_writer, name="jsonl-chunk-writer", daemon=False)
        self._thread.start()

    def write(self, obj: Any) -> bool:
        """Serialize and enqueue one JSON object.

        Returns ``True`` when the record was accepted. Returns ``False`` when
        the object is not JSON-serializable, the writer is closed, or the queue
        is full and ``backpressure_policy`` is ``DROP``.
        """
        try:
            line = json.dumps(obj, default=_json_default, sort_keys=True, separators=(",", ":")) + "\n"
        except (TypeError, ValueError):
            self._records_dropped += 1
            if self._records_dropped == 1 or self._records_dropped % self.config.log_every == 0:
                logger.warning(
                    "jsonl-writer dropping non-json record output_uri=%s dropped=%d",
                    self.config.output_uri,
                    self._records_dropped,
                )
            return False
        if self._closed:
            return False
        self._raise_if_writer_failed()
        if self.config.backpressure_policy is BackpressurePolicy.BLOCK:
            self._put_record_with_backpressure(line)
        else:
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                self._raise_if_writer_failed()
                self._records_dropped += 1
                if self._records_dropped == 1 or self._records_dropped % self.config.log_every == 0:
                    logger.warning(
                        "jsonl-writer dropping records output_uri=%s queue_size=%d dropped=%d",
                        self.config.output_uri,
                        self._queue.qsize(),
                        self._records_dropped,
                    )
                return False

        self._records_enqueued += 1
        queue_size = self._queue.qsize()
        self._max_queue_size_observed = max(self._max_queue_size_observed, queue_size)
        if self._records_enqueued % self.config.log_every == 0:
            logger.info(
                "jsonl-writer queued output_uri=%s enqueued=%d queue_size=%d dropped=%d",
                self.config.output_uri,
                self._records_enqueued,
                queue_size,
                self._records_dropped,
            )
        return True

    def close(self) -> None:
        """Signal completion, wait for final flush, and write the manifest."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._put_done_signal()
            self._thread.join()
        if self._writer_error is not None:
            raise RuntimeError(f"JSONL writer failed: {self._writer_error}")

    def stats(self) -> JsonlChunkWriterStats:
        """Return a best-effort counter snapshot."""
        return JsonlChunkWriterStats(
            records_enqueued=self._records_enqueued,
            records_written=self._records_written,
            records_dropped=self._records_dropped,
            chunks_written=self._chunks_written,
            bytes_written=self._bytes_written,
            max_queue_size_observed=self._max_queue_size_observed,
        )

    def _put_record_with_backpressure(self, line: str) -> None:
        while True:
            self._raise_if_writer_failed()
            try:
                self._queue.put(line, timeout=1.0)
                return
            except queue.Full:
                continue

    def _put_done_signal(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put(_DONE, timeout=1.0)
                return
            except queue.Full:
                continue

    def _raise_if_writer_failed(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"JSONL writer failed: {self._writer_error}")
        if self._thread is not None and not self._thread.is_alive() and not self._closed:
            raise RuntimeError("JSONL writer thread exited before close")

    def _run_writer(self) -> None:
        part_index = 0
        records: list[str] = []
        try:
            while True:
                item = self._queue.get()
                if item is _DONE:
                    break
                assert isinstance(item, str)
                records.append(item)
                if len(records) >= self.config.records_per_chunk:
                    self._flush(records, part_index)
                    part_index += 1
                    records = []
            if records:
                self._flush(records, part_index)
            self._ended_at = dt.datetime.now(dt.UTC).isoformat()
            self._write_manifest(completed=True)
        except Exception as exc:
            self._writer_error = f"{type(exc).__name__}: {exc}"
            self._ended_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                self._write_manifest(completed=False)
            except Exception:
                logger.exception("jsonl-writer failed to write failure manifest output_uri=%s", self.config.output_uri)

    def _flush(self, records: list[str], part_index: int) -> None:
        relative_path = f"parts/part-{part_index:06d}.jsonl"
        uri = f"{self.config.output_uri.rstrip('/')}/{relative_path}"
        body = "".join(records)
        self._write_text_file(uri, body)
        byte_count = len(body.encode("utf-8"))
        self._records_written += len(records)
        self._chunks_written += 1
        self._bytes_written += byte_count
        self._chunks.append(
            {
                "path": relative_path,
                "records": len(records),
                "bytes": byte_count,
                "written_at": dt.datetime.now(dt.UTC).isoformat(),
            }
        )
        logger.info(
            "jsonl-writer flush output_uri=%s part=%06d records=%d bytes=%d queue_size=%d dropped=%d",
            self.config.output_uri,
            part_index,
            len(records),
            byte_count,
            self._queue.qsize(),
            self._records_dropped,
        )

    def _write_manifest(self, *, completed: bool) -> None:
        manifest = {
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "completed": completed,
            "error": self._writer_error,
            "config": dataclasses.asdict(self.config),
            "records_enqueued": self._records_enqueued,
            "records_written": self._records_written,
            "records_dropped": self._records_dropped,
            "chunks_written": self._chunks_written,
            "bytes_written": self._bytes_written,
            "max_queue_size_observed": self._max_queue_size_observed,
            "chunks": list(self._chunks),
        }
        manifest_uri = f"{self.config.output_uri.rstrip('/')}/manifest.json"
        self._write_text_file(manifest_uri, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    @classmethod
    def _write_text_file(cls, uri: str, body: str) -> None:
        fs, path = fsspec.core.url_to_fs(uri)
        parent = posixpath.dirname(path)
        if parent:
            fs.mkdirs(parent, exist_ok=True)
        with fs.open(path, "wt") as f:
            f.write(body)


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")






DEFAULT_NVIDIA_SMI_FIELDS = (
    "timestamp",
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
)


@dataclass(frozen=True)
class NvidiaSmiTelemetryConfig:
    """Configuration for a durable ``nvidia-smi`` telemetry process.

    Args:
        output_uri: Destination directory for JSONL chunks and manifest.
        interval: Seconds between ``nvidia-smi`` samples.
        records_per_chunk: Number of samples per durable JSONL chunk file.
        max_queue_items: Maximum JSONL records queued in the writer actor before
            backpressure policy is applied.
        backpressure_policy: Whether telemetry writes block or drop when the
            writer queue is full.
        log_every: Log writer progress after this many accepted/dropped records.
        query_fields: ``nvidia-smi --query-gpu`` fields.
        command: Optional full command. Tests and non-NVIDIA probes can pass a
            command that emits ``nvidia-smi --format=csv``-style output.
        start_method: Multiprocessing start method used for the telemetry
            process.
        stop_timeout: Seconds to wait for graceful process shutdown before
            terminating it.
        require_command: Fail before training starts if the command executable
            is not present.
    """

    output_uri: str
    interval: float = 5.0
    records_per_chunk: int = 120
    max_queue_items: int = 10_000
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.BLOCK
    log_every: int = 1_000
    query_fields: tuple[str, ...] = DEFAULT_NVIDIA_SMI_FIELDS
    command: tuple[str, ...] = ()
    start_method: str = "spawn"
    stop_timeout: float = 30.0
    require_command: bool = True

    def __post_init__(self) -> None:
        if not self.output_uri:
            raise ValueError("output_uri must be non-empty")
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        if self.max_queue_items <= 0:
            raise ValueError("max_queue_items must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if isinstance(self.backpressure_policy, str):
            object.__setattr__(self, "backpressure_policy", BackpressurePolicy(self.backpressure_policy))
        if not isinstance(self.backpressure_policy, BackpressurePolicy):
            raise ValueError("backpressure_policy must be a BackpressurePolicy")
        if self.stop_timeout <= 0:
            raise ValueError("stop_timeout must be positive")
        if self.command and not self.command[0]:
            raise ValueError("command executable must be non-empty")
        if not self.command and not self.query_fields:
            raise ValueError("query_fields must be non-empty when command is not set")


def build_nvidia_smi_command(config: NvidiaSmiTelemetryConfig) -> tuple[str, ...]:
    """Return the command used by the telemetry process."""
    if config.command:
        return config.command
    return (
        "nvidia-smi",
        f"--query-gpu={','.join(config.query_fields)}",
        "--format=csv",
        "-l",
        _format_seconds_for_nvidia_smi(config.interval),
    )


@dataclass(frozen=True)
class NvidiaSmiTelemetryHandle:
    """Handle for a running telemetry process."""

    process: multiprocessing.Process
    stop_event: EventType
    stop_timeout: float

    def stop(self) -> None:
        """Request shutdown, flush the final chunk, and reap the process."""
        self.stop_event.set()
        self.process.join(timeout=self.stop_timeout)
        if self.process.is_alive():
            logger.warning("GPU telemetry process did not stop in %.1fs; terminating", self.stop_timeout)
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.exitcode not in (0, None):
            logger.warning("GPU telemetry process exited with code %s", self.process.exitcode)


def start_nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig) -> NvidiaSmiTelemetryHandle:
    """Start a background process that writes GPU telemetry JSONL chunks."""
    command = build_nvidia_smi_command(config)
    if config.require_command and shutil.which(command[0]) is None:
        raise FileNotFoundError(f"GPU telemetry command not found: {command[0]}")
    ctx = multiprocessing.get_context(config.start_method)
    stop_event = ctx.Event()
    process = ctx.Process(
        target=run_nvidia_smi_telemetry,
        args=(config, stop_event),
        name="nvidia-smi-telemetry",
        daemon=False,
    )
    process.start()
    return NvidiaSmiTelemetryHandle(process=process, stop_event=stop_event, stop_timeout=config.stop_timeout)


@contextlib.contextmanager
def nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig) -> Iterator[NvidiaSmiTelemetryHandle]:
    """Run durable GPU telemetry while the body executes."""
    handle = start_nvidia_smi_telemetry(config)
    try:
        yield handle
    finally:
        handle.stop()


def run_nvidia_smi_telemetry(config: NvidiaSmiTelemetryConfig, stop_event: EventType) -> None:
    """Run ``nvidia-smi`` and write parsed samples as JSONL chunks."""
    command = build_nvidia_smi_command(config)
    writer_config = JsonlChunkWriterConfig(
        output_uri=config.output_uri,
        records_per_chunk=config.records_per_chunk,
        max_queue_items=config.max_queue_items,
        backpressure_policy=config.backpressure_policy,
        log_every=config.log_every,
    )
    process: subprocess.Popen[str] | None = None
    with JsonlChunkWriter(writer_config) as writer:
        writer.write(
            {
                "record_type": "metadata",
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": list(command),
                "query_fields": list(config.query_fields),
                "interval": config.interval,
            }
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            header: list[str] | None = None
            while not stop_event.is_set():
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue
                row = next(csv.reader([line.rstrip("\n")]))
                if header is None:
                    header = [_normalize_field_name(field) for field in row]
                    continue
                values = {header[index]: value.strip() for index, value in enumerate(row) if index < len(header)}
                writer.write(
                    {
                        "record_type": "gpu_sample",
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "nvidia_smi": values,
                    }
                )
            if process.poll() is not None and process.returncode not in (0, None):
                stderr = process.stderr.read() if process.stderr is not None else ""
                writer.write(
                    {
                        "record_type": "error",
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "message": "telemetry command exited nonzero",
                        "returncode": process.returncode,
                        "stderr": stderr.strip(),
                    }
                )
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            writer.write(
                {
                    "record_type": "metadata",
                    "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "event": "telemetry_stop",
                }
            )


def _normalize_field_name(field: str) -> str:
    field = field.strip()
    if " [" in field:
        field = field.split(" [", 1)[0]
    return field.replace(".", "_").replace(" ", "_").replace("-", "_")


def _format_seconds_for_nvidia_smi(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))
    return str(seconds)


