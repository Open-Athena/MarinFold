# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Gzip-safe object fetch for the exp40 Zephyr worker.

A faithful, minimal copy of ``read_object_bytes`` from
``marinfold/document_structures/io.py`` (the canonical shared helper). exp40 is
a deliberately self-contained island — it vendors bio2token and pins torch to
2.9 for torch_xla — so it copies this ~1 function rather than taking a git
dependency on the whole ``marinfold`` package (and its transitive deps), which
would risk the torch pin. Keep the semantics identical to the canonical source;
if that one changes, mirror the change here.

The load-bearing detail: use a full ``cat_file`` GET, not a size-based
``open().read()``. GCS objects served with
``Content-Encoding: gzip`` report their *compressed* size in Content-Length, so
a size-based read silently truncates large cifs mid-``_atom_site`` and gemmi
then raises a confusing "unexpected end". ``cat_file`` reads to EOF.
"""

import warnings

import fsspec


def read_object_bytes(uri: str, *, missing_ok: bool = False, warn: bool = True) -> bytes | None:
    """Read the full byte contents of a remote (or local) object via fsspec.

    Raises on I/O failure by default (fail-loud: a silently-dropped row corrupts
    the training corpus). Pass ``missing_ok=True`` only where a missing object is
    expected and acceptable; it then returns ``None`` (and warns, unless
    ``warn=False``) instead of raising.
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
