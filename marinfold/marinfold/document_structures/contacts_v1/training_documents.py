# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-facing training-document variants for contacts-v1."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from marinfold.document_structures.contacts_v1.generate import (
    EmittedContact,
    GenerationResult,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    CONTEXT_LENGTH,
    DOC_TYPE,
    END,
    EOS,
    POSITIONS,
    THINK,
    VOCABULARY,
)
from marinfold.document_structures.core import Token
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    QUERY,
    AttentionLayout,
    Coordinate,
    Document,
    causal_training_document,
)


class ContactDocumentStyle(StrEnum):
    """Available model-facing representations of one selected contact set."""

    CAUSAL_SERIALIZED = "causal-serialized"
    BLOCK_CAUSAL_RELATIVE = "block-causal-relative"


RELATIVE_POSITION = Coordinate("relative_position")


@dataclass(frozen=True)
class DocumentConstructionConfig:
    """Select a model-facing representation independently of contact analysis."""

    style: ContactDocumentStyle = ContactDocumentStyle.CAUSAL_SERIALIZED
    think_tokens: int = 0
    max_seq_len: int = CONTEXT_LENGTH

    def __post_init__(self) -> None:
        if self.think_tokens < 0:
            raise ValueError(
                f"think_tokens must be non-negative, got {self.think_tokens}"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.style == ContactDocumentStyle.CAUSAL_SERIALIZED and self.think_tokens:
            raise ValueError(
                "think_tokens is only supported by block-causal-relative; "
                "use GenerationConfig(think=True) for serialized causal documents"
            )


def build_contact_training_document(
    generation: GenerationResult,
    *,
    config: DocumentConstructionConfig = DocumentConstructionConfig(),
) -> Document:
    """Build the configured model-facing document from canonical generation."""
    if config.style == ContactDocumentStyle.CAUSAL_SERIALIZED:
        return causal_document_from_generation(generation)
    if config.style == ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE:
        return block_causal_relative_document(
            generation,
            think_tokens=config.think_tokens,
            max_seq_len=config.max_seq_len,
        )
    raise ValueError(f"Unsupported contact document style: {config.style}")


def causal_document_from_generation(generation: GenerationResult) -> Document:
    """Strictly encode a canonical generated document under contacts-v1."""
    tokens = VOCABULARY.encode(generation.document.split())
    if len(tokens) != generation.num_tokens:
        raise ValueError(
            f"Encoded {len(tokens)} ids for {generation.num_tokens} contacts-v1 tokens"
        )
    return causal_training_document((*tokens, EOS))


def _endpoint_tokens(contact: EmittedContact) -> tuple[Token, Token]:
    canonical = (POSITIONS[contact.seq_i], POSITIONS[contact.seq_j])
    return canonical[::-1] if contact.flipped else canonical


def _add_weight(weights: dict[int, float], token: Token, amount: float = 1.0) -> None:
    token_id = int(token)
    weights[token_id] = weights.get(token_id, 0.0) + amount


def _contact_suffix_targets(
    contacts: tuple[EmittedContact, ...],
) -> tuple[tuple[Token, ...], tuple[Token, ...], np.ndarray]:
    """Build one teacher-forced suffix and its without-replacement oracle.

    Contact order and orientation come from the canonical generator. At every
    ``<contact>`` input, the next-position distribution is the marginal of a
    uniform draw over both orientations of every remaining edge. At the
    teacher-forced first endpoint, the second endpoint is uniform over the
    remaining incident edges. The current edge is removed before constructing
    the next contact slot.
    """
    remaining = [
        (contact, POSITIONS[contact.seq_i], POSITIONS[contact.seq_j])
        for contact in contacts
    ]
    suffix: list[Token] = []
    weight_rows: list[dict[int, float]] = []

    for contact in contacts:
        weight_rows.append({int(CONTACT): 1.0})
        suffix.append(CONTACT)

        first_weights: dict[int, float] = {}
        for _, first, second in remaining:
            _add_weight(first_weights, first)
            _add_weight(first_weights, second)
        weight_rows.append(first_weights)

        actual_first, actual_second = _endpoint_tokens(contact)
        suffix.append(actual_first)

        second_weights: dict[int, float] = {}
        for _, first, second in remaining:
            if int(actual_first) == int(first):
                _add_weight(second_weights, second)
            elif int(actual_first) == int(second):
                _add_weight(second_weights, first)
        if int(actual_second) not in second_weights:
            raise AssertionError("Teacher-forced contact is absent from its oracle")
        weight_rows.append(second_weights)
        suffix.append(actual_second)

        current, _, _ = remaining.pop(0)
        if current != contact:
            raise AssertionError("Contact order diverged while constructing targets")

    weight_rows.append({int(END): 1.0})
    suffix.append(END)

    candidate_by_id: dict[int, Token] = {}
    for token in (CONTACT, END, *suffix):
        candidate_by_id.setdefault(int(token), token)
    for row in weight_rows:
        for token_id in row:
            if token_id not in candidate_by_id:
                raise AssertionError("Oracle target is absent from the contact suffix")

    target_ids = tuple(candidate_by_id.values())
    target_index = {int(token): index for index, token in enumerate(target_ids)}
    target_weights = np.zeros((len(weight_rows), len(target_ids)), dtype=np.float32)
    for position, row in enumerate(weight_rows):
        for token_id, weight in row.items():
            target_weights[position, target_index[token_id]] = weight
    return tuple(suffix), target_ids, target_weights


def block_causal_relative_document(
    generation: GenerationResult,
    *,
    think_tokens: int = 0,
    max_seq_len: int = CONTEXT_LENGTH,
) -> Document:
    """Build a full-attention sequence prefix and causal contact-token suffix.

    The natural-order amino-acid sequence and framing form one bidirectional
    block. Optional observed ``<think>`` tokens and every teacher-forced
    contact token each occupy a successive singleton block. The last context
    token predicts the first suffix token; every suffix token then predicts the
    next one. Sparse weighted targets marginalize uniformly over all remaining
    contacts and orientations without exposing future target tokens.
    """
    if think_tokens < 0:
        raise ValueError(f"think_tokens must be non-negative, got {think_tokens}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    sequence_tokens = [
        VOCABULARY.token(f"<{residue.resname}>") for residue in generation.residues
    ]
    prefix_tokens: list[Token] = [DOC_TYPE, BEGIN_SEQUENCE]
    prefix_tokens.extend(sequence_tokens)
    prefix_tokens.append(BEGIN_STRUCTURE)
    context_tokens = [*prefix_tokens, *([THINK] * think_tokens)]
    suffix_tokens, target_ids, target_weights = _contact_suffix_targets(
        generation.contacts
    )
    token_ids = (*context_tokens, *suffix_tokens)
    if len(token_ids) > max_seq_len:
        raise ValueError(
            f"Block-causal document needs {len(token_ids)} tokens, "
            f"exceeding max_seq_len={max_seq_len}"
        )

    prefix_length = len(prefix_tokens)
    context_blocks = (0,) * prefix_length + tuple(range(1, think_tokens + 1))
    first_suffix_block = think_tokens + 1
    suffix_blocks = tuple(
        range(first_suffix_block, first_suffix_block + len(suffix_tokens))
    )
    context_relative_positions = (
        (RELATIVE_POSITION.missing,) * 2
        + tuple(range(len(sequence_tokens)))
        + (RELATIVE_POSITION.missing,) * (1 + think_tokens)
    )
    prediction_start = len(context_tokens) - 1
    query = np.zeros(len(token_ids), dtype=np.bool_)
    query[prediction_start : prediction_start + len(suffix_tokens)] = True

    document = Document(
        token_ids,
        {
            RELATIVE_POSITION: (
                *context_relative_positions,
                *range(len(suffix_tokens)),
            ),
            QUERY: query,
            ATTENTION_BLOCK: (*context_blocks, *suffix_blocks),
        },
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).unscored()
    return document.with_target_distribution(
        target_ids,
        target_weights,
        start=prediction_start,
    )


__all__ = [
    "ContactDocumentStyle",
    "DocumentConstructionConfig",
    "RELATIVE_POSITION",
    "block_causal_relative_document",
    "build_contact_training_document",
    "causal_document_from_generation",
]
