# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``DocumentStructure`` interface.

A document structure has two responsibilities:

1. **Generate**: given input data (e.g. PDB text, AFDB cif, a parsed
   structure), produce zero or more training documents.
2. **Evaluate**: given a trained model and a corpus of ground-truth
   structures, compute accuracy / calibration / generative-quality
   metrics.

Concrete implementations live as experiments under
``experiments/exp<N>_document_structures_<name>/``. The implementation
module must expose a top-level ``get_structure()`` function that
returns a ``DocumentStructure`` instance (or a ``STRUCTURE`` module-
level attribute holding one). The CLI in ``cli.py`` loads modules by
file path via ``importlib.util.spec_from_file_location`` — no
``pip install`` of the implementation is needed.

The interface is intentionally minimal. As real implementations
arrive (the first will be ``contacts-and-distances-v1`` ported from
``experiments/exp0_models_protein_docs_initial_port/``), this file
will gain typed argument shapes for ``input_record`` and
``ground_truth_record``. For now they're typed loosely so the first
implementation can pick what makes sense.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalResult:
    """Output of ``DocumentStructure.evaluate``.

    Attributes:
        metrics: scalar metrics keyed by name. Keys should be stable
            across runs of the same structure (so they can be tracked
            across model checkpoints).
        per_example: optional per-example records, e.g. a list of
            dicts with the input id, ground-truth target, model
            prediction, and per-example loss. May be large; the CLI
            writes these to a parquet at ``--out``.
        extras: free-form metadata about the eval run (model id,
            structure name, sample counts, etc.). Goes into the
            summary alongside ``metrics``.
    """

    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentStructure(Protocol):
    """Standard interface for a MarinFold document structure.

    Implementations should be cheap to construct — heavy resources
    (tokenizers, model checkpoints, PDB caches) should be lazy or
    held by the implementation's own state.
    """

    name: str
    """Stable identifier, e.g. ``"contacts-and-distances-v1"``.
    Matches the leading token in the corresponding training-doc
    format (so a doc looks like
    ``<contacts-and-distances-v1> <begin_sequence> ...``)."""

    tokenizer: str
    """HuggingFace tokenizer reference, ideally pinned to a revision
    (``timodonnell/protein-docs-tokenizer@<sha>``). See
    ``models/AGENTS.md`` on revision pinning."""

    context_length: int
    """Default maximum token length per generated document. The
    ``generate`` CLI flag ``--context-length`` overrides this."""

    def generate_documents(
        self,
        input_records: Iterator[Any],
        *,
        context_length: int | None = None,
        num_docs: int | None = None,
    ) -> Iterator[str]:
        """Yield training-document strings from ``input_records``.

        Args:
            input_records: iterator over implementation-specific
                input records (e.g. dicts with PDB text + metadata).
            context_length: maximum token length per produced
                document; defaults to ``self.context_length``.
            num_docs: optional cap. Implementations should stop
                yielding when reached.
        """
        ...

    def evaluate(
        self,
        *,
        model_path: str,
        ground_truth_records: Iterator[Any],
    ) -> EvalResult:
        """Run the model on ground-truth inputs and return metrics.

        Args:
            model_path: filesystem / GCS / HF reference to a trained
                model (the implementation decides how to load it).
            ground_truth_records: iterator over implementation-specific
                ground-truth records.
        """
        ...


def load_structure(impl_path: Path | str) -> DocumentStructure:
    """Load a ``DocumentStructure`` implementation from a file path.

    The module is loaded under a synthetic package name based on the
    file path (so two impls with the same module name in different
    experiment dirs don't collide). The module must expose either
    ``get_structure()`` (preferred) or a top-level ``STRUCTURE``
    attribute.

    Args:
        impl_path: path to a Python file implementing the structure.
    """
    p = Path(impl_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Document-structure implementation not found: {p}")

    mod_name = f"_marinfold_ds_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {p} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_structure"):
        structure = module.get_structure()
    elif hasattr(module, "STRUCTURE"):
        structure = module.STRUCTURE
    else:
        raise AttributeError(
            f"{p} does not expose get_structure() or STRUCTURE — "
            "the document-structure CLI requires one of them."
        )

    if not isinstance(structure, DocumentStructure):
        raise TypeError(
            f"{p} produced a {type(structure).__name__} that does not "
            "satisfy the DocumentStructure protocol. Required attrs/"
            "methods: name, tokenizer, context_length, generate_documents, evaluate."
        )
    return structure
