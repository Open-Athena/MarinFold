# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Interfaces + local CLI for MarinFold document structures.

A document structure is defined by **two files** under
``experiments/exp<N>_document_structures_<name>/``:

- ``generate.py`` — exports ``get_generator() -> Generator``.
- ``inference.py`` — exports ``get_inference() -> Inference``.

Both expose ``name``, ``context_length``, ``tokens()`` (they must
agree on the vocab) and an ``add_args`` hook for adding their own
CLI flags. The CLI subcommands ``generate``, ``infer``, ``evaluate``
load the corresponding file and dispatch to its ``run`` / ``predict``
/ ``evaluate`` method.
"""

from marinfold_document_structures.interface import (
    EvalResult,
    Generator,
    Inference,
    build_tokenizer,
    load_generator,
    load_inference,
)

__all__ = [
    "EvalResult",
    "Generator",
    "Inference",
    "build_tokenizer",
    "load_generator",
    "load_inference",
]
