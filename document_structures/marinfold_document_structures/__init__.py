# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard interface + local-testing CLI for MarinFold document structures.

A *document structure* declares its vocabulary via ``tokens()``,
generates training documents from input structures
(``generate_documents``), and scores models against ground-truth
structures (``evaluate``). Each concrete structure lives as an
experiment under ``experiments/exp<N>_document_structures_<name>/``
and exposes a ``get_structure()`` function returning a
``DocumentStructure``.

Production data-gen and eval wrappers live in ``data/`` and ``evals/``
respectively; this library only defines the interface, the
``build_tokenizer`` helper, and the local
``marinfold-document-structure`` CLI for poking at an implementation.
"""

from marinfold_document_structures.interface import (
    DocumentStructure,
    EvalResult,
    build_tokenizer,
    load_structure,
)

__all__ = ["DocumentStructure", "EvalResult", "build_tokenizer", "load_structure"]
