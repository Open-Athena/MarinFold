# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``DocumentStructure`` interface.

A document structure has three responsibilities:

1. **Declare its vocabulary** — ``tokens()`` returns the canonical
   ordered list of tokens that can appear in any document. The
   tokenizer is built from this list; the implementation does not
   own the tokenizer artifact, just its vocabulary.
2. **Generate** training documents from input data (PDB / mmCIF /
   parsed structures).
3. **Evaluate** trained models against ground-truth structures.

Concrete implementations live as experiments under
``experiments/exp<N>_document_structures_<name>/``. The implementation
module must expose a top-level ``get_structure()`` function that
returns a ``DocumentStructure`` instance (or a ``STRUCTURE`` module-
level attribute holding one). The CLI in ``cli.py`` loads modules by
file path via ``importlib.util.spec_from_file_location`` — no
``pip install`` of the implementation is needed.

The interface is intentionally minimal. As real implementations
arrive (the first being ``contacts-and-distances-v1`` under
``experiments/exp1_document_structures_contacts_and_distances_v1/``)
this file may grow typed argument shapes for ``input_record`` and
``ground_truth_record``. For now they're typed loosely so each impl
can pick what makes sense for its source data.
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
    (PDB caches, model loads, vllm engines) should be lazy or held
    by the implementation's own state.
    """

    name: str
    """Stable identifier, e.g. ``"contacts-and-distances-v1"``.
    Matches the leading token in the corresponding training-doc
    format (so a doc looks like
    ``<contacts-and-distances-v1> <begin_sequence> ...``)."""

    context_length: int
    """Default maximum token length per generated document. The
    ``generate`` CLI flag ``--context-length`` overrides this."""

    def tokens(self) -> list[str]:
        """The full ordered list of domain tokens this structure uses.

        ``build_tokenizer(structure)`` (see :func:`build_tokenizer`)
        constructs a ``PreTrainedTokenizerFast`` from this list by
        prepending the standard ``<pad>`` and ``<eos>`` specials.

        **Order is load-bearing.** New tokens must be APPENDED to the
        list to keep existing token IDs stable across revisions — a
        permutation would silently invalidate every checkpoint
        trained against the prior order. If a token reordering is
        ever genuinely required, the structure becomes a new
        ``contacts-and-distances-v2`` (separate experiment, separate
        graduation symlink) rather than a same-name update.
        """
        ...

    def iter_inputs(self, path: Path) -> Iterator[Any]:
        """Open a file or directory and yield records to feed ``generate_documents``.

        The shape of each yielded record is implementation-specific
        (e.g. a ``ParsedChain`` dataclass, a dict, a raw structure
        object). The ``marinfold-document-structure generate`` CLI
        passes the result straight through to ``generate_documents``.
        """
        ...

    def iter_ground_truth(self, path: Path) -> Iterator[Any]:
        """Same idea as ``iter_inputs`` but for ``evaluate``'s ground-truth side.

        Many structures will share the same parsing code between
        ``iter_inputs`` and ``iter_ground_truth``; they're separate
        methods because eval-time records may carry extra fields
        (e.g. ground-truth contact maps) that aren't needed for
        doc generation.
        """
        ...

    def generate_documents(
        self,
        input_records: Iterator[Any],
        *,
        context_length: int | None = None,
        num_docs: int | None = None,
    ) -> Iterator[str]:
        """Yield training-document strings from ``input_records``.

        Args:
            input_records: iterator over impl-specific records.
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
            ground_truth_records: iterator over impl-specific
                ground-truth records.
        """
        ...


def build_tokenizer(structure: DocumentStructure):
    """Construct a HuggingFace ``PreTrainedTokenizerFast`` from ``structure.tokens()``.

    Prepends the standard MarinFold specials in the canonical order:

    - ``<pad>`` at id 0
    - ``<eos>`` at id 1
    - then ``structure.tokens()`` in order, starting at id 2

    The resulting tokenizer is WordLevel with whitespace pre-
    tokenization, which matches the convention used by every
    MarinFold doc structure to date: documents are space-separated
    sequences of `<token>` strings and tokenize 1:1.

    Imports ``tokenizers`` / ``transformers`` lazily so callers that
    only need the structure's ``tokens()`` list (e.g. the data
    pipeline) don't pay the HF import cost.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    domain_tokens = list(structure.tokens())
    all_tokens = ["<pad>", "<eos>", *domain_tokens]
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    if "<UNK>" in vocab:
        unk_token = "<UNK>"
    else:
        unk_token = "<pad>"  # fall back to pad so unknown tokens at least don't crash

    tokenizer_model = WordLevel(vocab=vocab, unk_token=unk_token)
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token="<pad>",
        eos_token="<eos>",
    )


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
            "methods: name, context_length, tokens(), iter_inputs(), "
            "iter_ground_truth(), generate_documents(), evaluate()."
        )
    return structure
