# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared I/O patterns for document-structure data-generation pipelines.

The per-row worker inside a Zephyr ``map_shard`` body usually follows
the same pattern across document-structure ``generate`` impls: fetch a
per-row resource over the network, parse it, hand it to the algorithm,
emit a row. Two helpers here cover the parts that recur:

- :func:`read_object_bytes`: fetch one object's bytes. Returns ``None``
  on I/O failure. Uses a full GET so transparently-gzip-transcoded
  objects aren't silently truncated.
- :func:`thread_per_row_in_shard`: a ``map_shard`` body that overlaps
  per-row I/O with CPU work via a thread pool.

See ``.agents/skills/zephyr-pipeline-performance/SKILL.md`` for the
full performance rationale and the surrounding decisions
(``--worker-cpu 1``, region pinning, etc.).
"""

import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar

import fsspec

T = TypeVar("T")
U = TypeVar("U")


def thread_per_row_in_shard(
    items: Iterable[T],
    *,
    worker: Callable[[T], Optional[U]],
    fetch_concurrency: int = 32,
    thread_name_prefix: str = "shard",
) -> Iterator[U]:
    """Run ``worker`` on each item via a ``ThreadPoolExecutor`` and yield
    non-``None`` results in input order.

    Intended as the body of a Zephyr ``map_shard`` callback when each row
    requires a per-row network fetch (GCS GET, HF download, …) followed
    by CPU work (parse + generate). The thread pool overlaps fetches
    with the CPU work of rows that have already arrived. The GIL is
    released during both the network syscall and most C-extension
    parsers (gemmi, pyconfind, ...), so threading is the per-shard
    speedup that matters most for I/O-bound workers.

    The pool size is capped at ``len(items)`` so a small shard does not
    spawn dozens of idle threads. Output uses ``pool.map`` (not
    ``as_completed``), so the iteration order matches the input order.
    This matters when downstream readers want deterministic per-shard
    parquet content across runs.

    Worker exceptions propagate. If the worker raises, the surrounding
    Zephyr worker dies, which is the correct behavior for data-quality
    failures (a silently-dropped row corrupts the downstream training
    corpus). Use ``None`` *only* for designed-in filters: rows that the
    worker has explicitly determined are not eligible (e.g. a structure
    too small to fit the context budget). Lenient skip-on-error
    behavior, when truly needed, lives at the call site (e.g.
    ``read_object_bytes(uri, missing_ok=True)``), not here.

    Parameters
    ----------
    items
        The shard's per-row inputs (Zephyr passes a list-like).
    worker
        Per-row callable. Should return ``None`` to skip a row, or the
        output value to yield. Must be picklable if the Zephyr backend
        ships it cross-process. The call site typically builds one via
        ``functools.partial(per_row_fn, **shard_constants)``.
    fetch_concurrency
        Max concurrent in-flight workers. ``32`` works well for the
        common case where the per-row GCS GET (~30–80 ms) is an order
        of magnitude larger than the per-row CPU step. Bump it when
        the CPU step is heavier, but past ~64 the overhead starts to
        dominate.
    thread_name_prefix
        Surfaces in thread dumps and logs. Useful when babysitting a
        running Zephyr job. Convention: ``"<exp-tag>-fetch"`` (e.g.
        ``"exp5-fetch"``).

    Example
    -------
    Inside an ``exp<N>_data_*/cli.py`` ``map_shard`` body:

    .. code-block:: python

        from functools import partial
        from marinfold.document_structures.io import thread_per_row_in_shard

        def generate_shard(items, shard_info, *, cfg, fetch_concurrency, ...):
            worker = partial(_generate_doc_for_row, cfg=cfg, ...)
            yield from thread_per_row_in_shard(
                items, worker=worker,
                fetch_concurrency=fetch_concurrency,
                thread_name_prefix="exp<N>-fetch",
            )

    For pipelines that have *both* a URI-fetch path and an inline path
    (e.g. ``--cif-text-column`` for local testing), branch at the call
    site and use ``map`` for the inline path. There is no I/O to
    overlap, so the thread pool is pure overhead.
    """
    rows = list(items)
    if not rows:
        return
    workers = min(fetch_concurrency, len(rows))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=thread_name_prefix) as pool:
        for out in pool.map(worker, rows):
            if out is not None:
                yield out


def read_object_bytes(
    uri: str, *, missing_ok: bool = False, warn: bool = True,
) -> Optional[bytes]:
    """Read the full byte contents of a remote (or local) object via fsspec.

    **Raises on I/O failure by default**. This is deliberate: in a
    data-generation pipeline, a silently-dropped row corrupts the
    training corpus, and the cost of discovering that weeks later (when
    a model behaves oddly) far exceeds the cost of killing a Zephyr
    worker on a single bad fetch.

    Pass ``missing_ok=True`` only when a missing/unfetchable object is
    expected and acceptable (e.g. enumerating which entries in a
    candidate list are actually present in the bucket). Lenient mode
    returns ``None`` instead of raising, and, by default, emits a
    ``warnings.warn`` so the drop shows up in logs.

    Uses the filesystem's ``cat_file`` (a single full GET that reads to
    EOF) rather than a seekable ``open(...).read()``. The two differ in
    one important case: objects served with ``Content-Encoding: gzip``
    (server-side transcoded) report their compressed size in the
    ``Content-Length`` header, while the body is decompressed on the
    wire. A size-based read returns only that many bytes of the
    decompressed stream and silently truncates large objects
    mid-content. The downstream parser then surfaces this as a
    confusing "unexpected end of input" rather than as an I/O fault.
    ``cat_file`` reads to EOF, so the full decompressed object comes
    back every time.

    The common case is GCS objects uploaded with ``Content-Encoding:
    gzip`` metadata, which GCS transparently decodes on serve (see the
    GCS "transcoding" docs).

    Parameters
    ----------
    uri
        fsspec-recognised URI: a local path, ``file://``, ``gs://``,
        ``s3://``, ``hf://``, ``http(s)://``, etc.
    missing_ok
        Default ``False`` (raise on failure). Pass ``True`` only when
        the caller has audited the failure modes and decided a silent
        skip is correct. With ``True``, returns ``None`` instead of
        raising.
    warn
        Only consulted when ``missing_ok=True``. Default emits a
        ``warnings.warn`` on the drop so it shows up in worker logs.
        Pass ``warn=False`` for bulk scans where the log spam would
        drown out signal.
    """
    try:
        fs, path = fsspec.core.url_to_fs(uri)
        return fs.cat_file(path)
    except (OSError, ValueError) as exc:
        if not missing_ok:
            raise
        if warn:
            warnings.warn(f"fetch failed for {uri}: {exc}", stacklevel=2)
        return None
