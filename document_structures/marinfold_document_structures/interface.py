# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Interfaces for MarinFold document structures.

A *document structure* defines a protein-document format. Each
concrete format lives as an experiment under
``experiments/exp<N>_document_structures_<name>/`` and ships two
public entry-point files at the experiment dir root:

- ``generate.py`` — exports ``get_generator() -> Generator``.
- ``inference.py`` — exports ``get_inference() -> Inference``.

Both Protocols expose ``name``, ``context_length``, ``tokens()``;
they MUST agree on the vocab. Shared parsing / vocab code lives in
private modules (``_vocab.py``, ``_parse.py``, …) inside the same
experiment dir.

The ``marinfold-document-structure`` CLI loads these modules by file
path (``importlib.util.spec_from_file_location``) — no
``pip install`` of the implementation is needed.
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# EvalResult
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Output of ``Inference.evaluate``.

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


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class Generator(Protocol):
    """Generate training documents from input structures.

    Implementations live in ``<exp_dir>/generate.py`` and expose
    ``get_generator() -> Generator``.
    """

    name: str
    """Stable identifier, e.g. ``"contacts-and-distances-v1"``. Must
    equal the corresponding ``Inference.name`` for the same format."""

    context_length: int
    """Default token-budget per document. The impl can override via
    its own ``--context-length`` if it wants."""

    def tokens(self) -> list[str]:
        """Canonical ordered domain vocabulary.

        ``build_tokenizer(generator)`` constructs a
        ``PreTrainedTokenizerFast`` from this list. Order is load-
        bearing — appending is fine, reordering is a v2 event.
        """
        ...

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add the generator's args to ``parser``.

        Called by the CLI before parsing. The impl owns every flag
        except ``--out`` (which the CLI provides). Typical flags:
        ``--input``, ``--num-docs``, ``--context-length``, plus any
        generator-specific knobs (rank means, contact cutoffs, …).
        """
        ...

    def run(self, args: argparse.Namespace) -> Iterator[str]:
        """Yield training-document strings per the parsed args."""
        ...


@runtime_checkable
class Inference(Protocol):
    """Run a trained model on input structures (with or without ground truth).

    Implementations live in ``<exp_dir>/inference.py`` and expose
    ``get_inference() -> Inference``. Powers both the ``infer`` and
    ``evaluate`` CLI subcommands:

    - ``infer`` — given inputs only, return model predictions.
    - ``evaluate`` — given inputs + ground truth, return metrics
      computed against the predictions.

    ``evaluate`` is expected to share code with ``predict`` (typically
    by calling it internally), but exposing both as Protocol methods
    keeps the two subcommands' CLI surfaces distinct.
    """

    name: str
    context_length: int

    def tokens(self) -> list[str]: ...

    def add_args(
        self, parser: argparse.ArgumentParser, *, subcommand: str
    ) -> None:
        """Add inference args. ``subcommand`` is ``'infer'`` or ``'evaluate'``.

        Use the ``subcommand`` kwarg to register different flags for
        each (e.g. ``--ground-truth`` only makes sense for evaluate).
        """
        ...

    def predict(self, args: argparse.Namespace) -> Iterator[dict]:
        """Run the model on inputs; yield one prediction record per input.

        Each record is a JSON-serializable dict (e.g.
        ``{"entry_id": ..., "expected_distances": [...]}``). Used by
        the ``infer`` CLI subcommand.
        """
        ...

    def evaluate(self, args: argparse.Namespace) -> EvalResult:
        """Run the model + compare against ground truth; return metrics.

        Used by the ``evaluate`` CLI subcommand. Internally typically
        calls ``predict`` and scores the outputs.
        """
        ...


# --------------------------------------------------------------------------
# build_tokenizer (works on either Protocol)
# --------------------------------------------------------------------------


def build_tokenizer(component):
    """Build a ``PreTrainedTokenizerFast`` from ``component.tokens()``.

    Accepts either a ``Generator`` or an ``Inference`` — both expose
    ``tokens()`` and must agree on the vocab.

    Prepends the standard MarinFold specials:

    - ``<pad>`` at id 0
    - ``<eos>`` at id 1
    - then ``component.tokens()`` in order, starting at id 2

    The resulting tokenizer is WordLevel with whitespace pre-
    tokenization, which matches the convention every MarinFold doc
    structure has used to date: documents are space-separated
    sequences of ``<token>`` strings and tokenize 1:1.

    ``tokenizers`` / ``transformers`` are imported lazily so callers
    that only need ``tokens()`` don't pay the HF import cost.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    domain_tokens = list(component.tokens())
    all_tokens = ["<pad>", "<eos>", *domain_tokens]
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    unk_token = "<UNK>" if "<UNK>" in vocab else "<pad>"

    tokenizer_model = WordLevel(vocab=vocab, unk_token=unk_token)
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token="<pad>",
        eos_token="<eos>",
    )


# --------------------------------------------------------------------------
# Module-by-file-path loaders
# --------------------------------------------------------------------------


def _load_module_by_path(path: Path):
    """Load a Python module by file path under a synthetic package name.

    Also puts the file's parent directory on ``sys.path`` (idempotent)
    so the loaded module can ``from _vocab import ...`` / ``from
    _parse import ...`` to reach its private sibling modules — that's
    the intended shape of a document-structure impl (two public
    files + shared private helpers in the same dir).
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Implementation file not found: {p}")
    parent = str(p.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod_name = f"_marinfold_ds_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {p} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _impl_dir(path: Path | str) -> Path:
    p = Path(path).resolve()
    if not p.is_dir():
        raise NotADirectoryError(
            f"Document-structure impl dir not found: {p}. The CLI takes "
            "the experiment dir (containing generate.py + inference.py), "
            "not a specific file."
        )
    return p


def load_generator(impl_dir: Path | str) -> Generator:
    """Load the ``Generator`` from ``<impl_dir>/generate.py``.

    The module must expose ``get_generator()`` returning a value
    that satisfies the ``Generator`` Protocol.
    """
    dir_path = _impl_dir(impl_dir)
    gen_path = dir_path / "generate.py"
    if not gen_path.is_file():
        raise FileNotFoundError(
            f"{dir_path} does not contain a generate.py — the CLI's "
            "`generate` subcommand requires one. See "
            "document_structures/README.md for the layout."
        )
    module = _load_module_by_path(gen_path)
    if not hasattr(module, "get_generator"):
        raise AttributeError(
            f"{gen_path} does not export get_generator(). Add a "
            "top-level function returning a Generator-conforming object."
        )
    generator = module.get_generator()
    if not isinstance(generator, Generator):
        raise TypeError(
            f"{gen_path}:get_generator() returned a {type(generator).__name__} "
            "that does not satisfy the Generator protocol. Required: "
            "name, context_length, tokens(), add_args(parser), run(args)."
        )
    return generator


def load_inference(impl_dir: Path | str) -> Inference:
    """Load the ``Inference`` from ``<impl_dir>/inference.py``.

    The module must expose ``get_inference()`` returning a value
    that satisfies the ``Inference`` Protocol.
    """
    dir_path = _impl_dir(impl_dir)
    inf_path = dir_path / "inference.py"
    if not inf_path.is_file():
        raise FileNotFoundError(
            f"{dir_path} does not contain an inference.py — the CLI's "
            "`infer` / `evaluate` subcommands require one. See "
            "document_structures/README.md for the layout."
        )
    module = _load_module_by_path(inf_path)
    if not hasattr(module, "get_inference"):
        raise AttributeError(
            f"{inf_path} does not export get_inference(). Add a "
            "top-level function returning an Inference-conforming object."
        )
    inference = module.get_inference()
    if not isinstance(inference, Inference):
        raise TypeError(
            f"{inf_path}:get_inference() returned a {type(inference).__name__} "
            "that does not satisfy the Inference protocol. Required: "
            "name, context_length, tokens(), add_args(parser, subcommand=), "
            "predict(args), evaluate(args)."
        )
    return inference
