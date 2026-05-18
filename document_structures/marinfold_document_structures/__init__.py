# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared toolkit for MarinFold document structures.

A document structure is a protein-document format. Each format lives
as an experiment under
``experiments/exp<N>_document_structures_<name>/`` with its own
``cli.py`` (``generate`` / ``infer`` / ``evaluate`` / ``tokenizer``
subcommands). This library is the small set of pieces every impl
shares:

- :class:`EvalResult` — the return shape of ``evaluate``.
- :func:`build_tokenizer` — build a ``PreTrainedTokenizerFast`` from
  an ordered token list.
- :func:`write_docs` / :func:`write_predictions` / :func:`write_eval`
  — parquet / jsonl writers for the three standard output shapes.
"""

from marinfold_document_structures.core import EvalResult, build_tokenizer
from marinfold_document_structures.writers import (
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
