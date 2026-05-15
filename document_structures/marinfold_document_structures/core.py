# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types + tokenizer construction for MarinFold document structures.

A *document structure* defines a protein-document format. Each
concrete format lives as an experiment under
``experiments/exp<N>_document_structures_<name>/`` with its own
``cli.py`` (``generate`` / ``infer`` / ``evaluate`` / ``tokenizer``
subcommands) and the modules it dispatches to (``generate.py``,
``inference.py``, plus shared ``vocab.py`` / ``parse.py``).

This library is intentionally tiny: it holds the shared types
(``EvalResult``) and the one piece of logic every impl needs to
agree on (``build_tokenizer``). The output writers live next door in
:mod:`marinfold_document_structures.writers`. Everything else —
argparse surfaces, file IO conventions, the impl's algorithm — is
the impl's business.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    """Output of an impl's ``evaluate`` function.

    Attributes:
        metrics: scalar metrics keyed by name. Keys should be stable
            across runs of the same structure (so they can be tracked
            across model checkpoints).
        per_example: optional per-example records (e.g. one dict per
            (input, prediction, target, loss) row). Written to a
            parquet sibling by :func:`write_eval` when present.
        extras: free-form metadata about the eval run (model id,
            structure name, sample counts, etc.). Surfaces in the
            JSON ``extras`` block and as flattened parquet columns.
    """

    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def build_tokenizer(tokens: list[str]):
    """Build a ``PreTrainedTokenizerFast`` from an ordered token list.

    Prepends the standard MarinFold specials:

    - ``<pad>`` at id 0
    - ``<eos>`` at id 1
    - then the impl's ``tokens`` in order, starting at id 2

    The resulting tokenizer is WordLevel with whitespace pre-
    tokenization — documents are space-separated sequences of
    ``<token>`` strings that tokenize 1:1. Every MarinFold doc
    structure has used this shape to date; new impls should too
    unless they have a strong reason not to.

    ``tokenizers`` / ``transformers`` are imported lazily so callers
    that only need the impl's vocab list don't pay the HF import
    cost.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    domain_tokens = list(tokens)
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
