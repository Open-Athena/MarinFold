# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared toolkit for MarinFold document-structure implementations.

A document structure is a protein-document format. Each format lives
as its own proper package under ``document_structures/<name>/``, from
its first commit — experiments import the impl from here rather than
carrying their own copy.

This subpackage holds the pieces every impl shares:

- :class:`EvalResult` — return shape of ``evaluate``.
- :func:`build_tokenizer` — build a ``PreTrainedTokenizerFast`` from
  an ordered token list.
- :func:`write_docs` / :func:`write_predictions` / :func:`write_eval`
  — parquet / jsonl writers for the three standard output shapes.
- :func:`read_object_bytes` / :func:`thread_per_row_in_shard` — fetch
  one object's bytes, and run a thread-pool ``map_shard`` body that
  overlaps per-row I/O with CPU work. Together they cover the per-row
  worker every data-generation pipeline ends up writing (see the
  ``zephyr-pipeline-performance`` skill).
"""

from marinfold.document_structures.core import EvalResult, build_tokenizer
from marinfold.document_structures.io import (
    read_object_bytes,
    thread_per_row_in_shard,
)
from marinfold.document_structures.writers import (
    write_docs,
    write_eval,
    write_predictions,
)

__all__ = [
    "EvalResult",
    "build_tokenizer",
    "read_object_bytes",
    "thread_per_row_in_shard",
    "write_docs",
    "write_eval",
    "write_predictions",
]
