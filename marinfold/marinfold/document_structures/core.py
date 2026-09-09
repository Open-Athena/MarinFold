# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types, vocabularies, and tokenizer construction.

A *document structure* defines a protein-document format. Each
concrete format lives as an experiment under
``experiments/exp<N>_document_structures_<name>/`` with its own
``cli.py`` (``generate`` / ``infer`` / ``evaluate`` / ``tokenizer``
subcommands) and the modules it dispatches to (``generate.py``,
``inference.py``, plus shared ``vocab.py`` / ``parse.py``).

This library is intentionally tiny: it holds the shared types
(``EvalResult``) and the one piece of logic every impl needs to
agree on (``build_tokenizer``). The output writers live next door in
:mod:`marinfold.document_structures.writers`. Everything else —
argparse surfaces, file IO conventions, the impl's algorithm — is
the impl's business.
"""

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


STANDARD_TOKENS = ("<pad>", "<eos>")


@dataclass(frozen=True)
class VocabularyIdentity:
    """Compact identity carried by tokens and encoded documents."""

    name: str
    fingerprint: str
    size: int


@dataclass(frozen=True)
class Token:
    """One token ID tied to the vocabulary that assigned it."""

    text: str
    id: int
    vocabulary: VocabularyIdentity

    def __int__(self) -> int:
        return self.id


@dataclass(frozen=True)
class TokenFamily(Sequence[Token]):
    """An exhaustive, indexable family of declared vocabulary tokens."""

    name: str
    tokens: tuple[Token, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)


@dataclass(frozen=True)
class Vocabulary:
    """Immutable ordered vocabulary and source of vocabulary-bound tokens."""

    name: str
    domain_tokens: tuple[str, ...]
    _token_ids: MappingProxyType[str, int] = field(
        init=False, repr=False, compare=False
    )
    identity: VocabularyIdentity = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A vocabulary requires a name")
        domain_tokens = tuple(self.domain_tokens)
        all_tokens = (*STANDARD_TOKENS, *domain_tokens)
        duplicates = sorted(
            token for token in set(all_tokens) if all_tokens.count(token) > 1
        )
        if duplicates:
            raise ValueError(f"Vocabulary tokens must be unique: {duplicates}")
        invalid = [
            token for token in all_tokens if not token or token.split() != [token]
        ]
        if invalid:
            raise ValueError(
                f"Vocabulary tokens must be non-empty and whitespace-free: {invalid}"
            )
        object.__setattr__(self, "domain_tokens", domain_tokens)
        token_ids = MappingProxyType(
            {token: token_id for token_id, token in enumerate(all_tokens)}
        )
        object.__setattr__(self, "_token_ids", token_ids)
        digest = hashlib.sha256()
        for token in all_tokens:
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
        object.__setattr__(
            self,
            "identity",
            VocabularyIdentity(self.name, digest.hexdigest(), len(all_tokens)),
        )

    @property
    def tokens(self) -> tuple[str, ...]:
        """All tokens in ID order, including standard tokens."""
        return (*STANDARD_TOKENS, *self.domain_tokens)

    def __len__(self) -> int:
        return self.identity.size

    def token(self, text: str) -> Token:
        """Return a vocabulary-bound token, failing if it was not declared."""
        try:
            token_id = self._token_ids[text]
        except KeyError:
            raise KeyError(f"Token {text!r} is not declared by {self.name}") from None
        return Token(text, token_id, self.identity)

    def encode(self, tokens: Iterable[str]) -> tuple[Token, ...]:
        """Strictly encode already-separated tokens without an unknown fallback."""
        return tuple(self.token(token) for token in tokens)

    def family(
        self, name: str, template: str, *, count: int, start: int = 0
    ) -> TokenFamily:
        """Require and return a complete formatted token family."""
        if count <= 0:
            raise ValueError(f"Token family count must be positive, got {count}")
        return TokenFamily(
            name,
            tuple(
                self.token(template.format(index))
                for index in range(start, start + count)
            ),
        )


class VocabularyBuilder:
    """Mutable declaration helper that freezes into an ordered vocabulary."""

    def __init__(self, name: str):
        if not name:
            raise ValueError("A vocabulary requires a name")
        self.name = name
        self._tokens: list[str] = []

    def append(self, *tokens: str) -> "VocabularyBuilder":
        """Append explicitly minted tokens in ID order."""
        self._tokens.extend(tokens)
        return self

    def inherit(self, vocabulary: Vocabulary) -> "VocabularyBuilder":
        """Append another vocabulary's domain-token block unchanged."""
        self._tokens.extend(vocabulary.domain_tokens)
        return self

    def freeze(self) -> Vocabulary:
        """Validate and freeze the collected vocabulary."""
        return Vocabulary(self.name, tuple(self._tokens))


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


def build_tokenizer(vocabulary: Vocabulary | list[str]):
    """Build a ``PreTrainedTokenizerFast`` from a vocabulary declaration.

    Prepends the standard MarinFold specials:

    - ``<pad>`` at id 0
    - ``<eos>`` at id 1
    - then the impl's ``tokens`` in order, starting at id 2

    Passing a :class:`Vocabulary` is the normal path. A raw list remains
    available for legacy formats and low-level tools, but it carries no
    identity into documents and cannot provide strict token handles.

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

    domain_tokens = (
        list(vocabulary.domain_tokens)
        if isinstance(vocabulary, Vocabulary)
        else list(vocabulary)
    )
    all_tokens = [*STANDARD_TOKENS, *domain_tokens]
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
