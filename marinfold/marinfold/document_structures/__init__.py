# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared toolkit for MarinFold document-structure implementations.

A document structure is a protein-document format. Each format lives
as its own proper package under ``document_structures/<name>/`` once
graduated (and as a flat experiment dir under
``experiments/exp<N>_document_structures_<name>/`` while in
flight).

This subpackage holds the three pieces every impl shares:

- :class:`EvalResult` — return shape of ``evaluate``.
- :func:`build_tokenizer` — build a ``PreTrainedTokenizerFast`` from
  an ordered token list.
- :func:`write_docs` / :func:`write_predictions` / :func:`write_eval`
  — parquet / jsonl writers for the three standard output shapes.
"""

from marinfold.document_structures.core import EvalResult, build_tokenizer
from marinfold.document_structures.writers import (
    write_docs,
    write_eval,
    write_predictions,
)

__all__ = [
    "EvalResult",
    "build_tokenizer",
    "write_docs",
    "write_eval",
    "write_predictions",
]
