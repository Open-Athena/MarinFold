# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard interface + local-testing CLI for MarinFold document structures.

A *document structure* is a recipe for turning a protein structure
(e.g. a PDB) into a string the LM can train on, plus a corresponding
recipe for scoring a trained LM against ground-truth structures. Each
concrete structure lives as an experiment under
``experiments/exp<N>_document_structures_<slug>/`` and exposes a
``get_structure()`` function returning a ``DocumentStructure``.

Production data-gen and eval wrappers live in ``data/`` and ``evals/``
respectively; this library only defines the interface and the local
``marinfold-document-structure`` CLI for poking at an implementation.
"""

from marinfold_document_structures.interface import (
    DocumentStructure,
    EvalResult,
    load_structure,
)

__all__ = ["DocumentStructure", "EvalResult", "load_structure"]
