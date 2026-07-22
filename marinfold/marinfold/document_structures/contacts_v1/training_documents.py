# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-facing training-document variants for contacts-v1.

The canonical contacts-v1 serializer remains the source of residue and contact
selection. This module controls only how that selected example is presented to
a model, so new attention layouts and coordinates can be compared without
forking structure analysis.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from marinfold.document_structures.contacts_v1.generate import GenerationResult
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
    ScoreContext,
    causal_training_document,
)


class ContactDocumentStyle(StrEnum):
    """Available model-facing representations of one selected contact set."""

    CAUSAL_SERIALIZED = "causal-serialized"
    FULL_ATTENTION_RELATIVE = "full-attention-relative"
    BLOCK_CAUSAL_RELATIVE = "block-causal-relative"


class ContactTargetScoring(StrEnum):
    """How full-attention contact query slots are matched to gold targets."""

    ORDERED_TOKENS = "ordered-tokens"
    UNORDERED_CONTACTS = "unordered-contacts"


RELATIVE_POSITION = Coordinate("relative_position")


def unordered_contacts_score(logits: Any, context: ScoreContext) -> Any:
    """Score undirected contacts with a sequential multi-positive oracle.

    This scorer deliberately knows the contacts-v1 target layout: zero or more
    ``<contact> <p_i> <p_j>`` triples followed by one ordered ``<end>`` token.
    At each query slot, every not-yet-consumed gold contact is valid. The
    highest-probability remaining contact is consumed before the next slot, so
    predicting the same contact again receives no credit. Both endpoint
    orientations are marginalized. Everything happens after one model forward.
    """
    if context.target_ids is None:
        raise ValueError("Unordered contact scoring requires explicit targets")
    xp = _array_namespace(logits)
    target_ids = xp.asarray(context.target_ids)
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"Contact scorer received {logits.shape[0]} logit positions for "
            f"{target_ids.shape[0]} targets"
        )

    contact_token_count = target_ids.shape[0] - 1
    if contact_token_count < 0 or contact_token_count % 3:
        raise ValueError(
            f"Contact target length {target_ids.shape[0]} must be 3 * contacts + 1"
        )
    contact_count = contact_token_count // 3
    end_logits = logits[contact_token_count:]
    end_targets = target_ids[contact_token_count:]
    if contact_count == 0:
        return xp.mean(_token_cross_entropy(end_logits, end_targets))

    contact_logits = logits[:contact_token_count].reshape(
        contact_count, 3, logits.shape[-1]
    )
    target_contacts = target_ids[:contact_token_count].reshape(contact_count, 3)
    reversed_contacts = target_contacts[:, [0, 2, 1]]
    canonical_costs = _contact_match_costs(contact_logits, target_contacts)
    reversed_costs = _contact_match_costs(contact_logits, reversed_contacts)
    contact_log_probs = xp.logaddexp(-canonical_costs, -reversed_costs)

    indices = xp.arange(contact_count)
    available_targets = xp.ones((contact_count,), dtype=xp.bool_)
    slot_losses = []
    for slot in range(contact_count):
        remaining_log_probs = xp.where(
            available_targets,
            contact_log_probs[slot],
            xp.asarray(-xp.inf),
        )
        slot_losses.append(-_logsumexp(remaining_log_probs, xp=xp))
        target = xp.argmax(remaining_log_probs)
        available_targets = available_targets & (indices != target)

    end_losses = _token_cross_entropy(end_logits, end_targets)
    total_loss = xp.sum(xp.stack(slot_losses)) + xp.sum(end_losses)
    return total_loss / target_ids.shape[0]


def _logsumexp(values: Any, *, xp: Any) -> Any:
    maximum = xp.max(values)
    return maximum + xp.log(xp.sum(xp.exp(values - maximum)))


def _contact_match_costs(contact_logits: Any, target_contacts: Any) -> Any:
    xp = _array_namespace(contact_logits)
    contact_count = contact_logits.shape[0]
    maxima = xp.max(contact_logits, axis=-1, keepdims=True)
    log_normalizers = xp.log(xp.sum(xp.exp(contact_logits - maxima), axis=-1))
    log_normalizers = log_normalizers + maxima[..., 0]
    costs = xp.zeros((contact_count, contact_count), dtype=log_normalizers.dtype)
    for field in range(3):
        target_logits = xp.take(
            contact_logits[:, field, :], target_contacts[:, field], axis=-1
        )
        costs = costs + log_normalizers[:, field, None] - target_logits
    return costs


def _token_cross_entropy(logits: Any, target_ids: Any) -> Any:
    xp = _array_namespace(logits)
    targets = xp.asarray(target_ids)
    maxima = xp.max(logits, axis=-1, keepdims=True)
    log_normalizers = xp.log(xp.sum(xp.exp(logits - maxima), axis=-1))
    log_normalizers = log_normalizers + maxima[..., 0]
    target_logits = xp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return log_normalizers - target_logits


def _array_namespace(array: Any) -> Any:
    namespace = getattr(array, "__array_namespace__", None)
    return namespace() if namespace is not None else np


@dataclass(frozen=True)
class DocumentConstructionConfig:
    """Select a model-facing representation independently of contact analysis.

    ``think_tokens`` applies to the relative-position representations. These
    are observed pause positions between the natural-order sequence and hidden
    target slots. The target slots themselves also use ``<think>`` as a neutral
    input token and are distinguished by the ``QUERY`` coordinate.
    """

    style: ContactDocumentStyle = ContactDocumentStyle.CAUSAL_SERIALIZED
    target_scoring: ContactTargetScoring = ContactTargetScoring.ORDERED_TOKENS
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
                "think_tokens is only supported by relative-position styles; "
                "use GenerationConfig(think=True) for serialized causal documents"
            )
        if (
            self.style == ContactDocumentStyle.CAUSAL_SERIALIZED
            and self.target_scoring != ContactTargetScoring.ORDERED_TOKENS
        ):
            raise ValueError(
                "target_scoring is only supported by full-attention-relative"
            )


def build_contact_training_document(
    generation: GenerationResult,
    *,
    config: DocumentConstructionConfig = DocumentConstructionConfig(),
) -> Document:
    """Build the configured model-facing document from canonical generation."""
    if config.style == ContactDocumentStyle.CAUSAL_SERIALIZED:
        return causal_document_from_generation(generation)
    if config.style == ContactDocumentStyle.FULL_ATTENTION_RELATIVE:
        return full_attention_relative_document(
            generation,
            think_tokens=config.think_tokens,
            max_seq_len=config.max_seq_len,
            target_scoring=config.target_scoring,
        )
    if config.style == ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE:
        return block_causal_relative_document(
            generation,
            think_tokens=config.think_tokens,
            max_seq_len=config.max_seq_len,
            target_scoring=config.target_scoring,
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


def full_attention_relative_document(
    generation: GenerationResult,
    *,
    think_tokens: int = 0,
    max_seq_len: int = CONTEXT_LENGTH,
    target_scoring: ContactTargetScoring = ContactTargetScoring.ORDERED_TOKENS,
) -> Document:
    """Build a full-attention sequence context with hidden canonical targets.

    Amino acids occur once, in primary-sequence order, and carry their 0-based
    chain-relative position as a coordinate. Contact targets use true relative
    indices and canonical ``(i, j)`` ordering, rather than the serialized
    format's random wrap-around positions, shuffled statements, and pair flips.
    Target labels live only in the query span's scoring metadata; full
    attention therefore cannot read them from the model inputs.
    """
    if think_tokens < 0:
        raise ValueError(f"think_tokens must be non-negative, got {think_tokens}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    sequence_tokens = [
        VOCABULARY.token(f"<{residue.resname}>") for residue in generation.residues
    ]
    context_tokens: list[Token] = [DOC_TYPE, BEGIN_SEQUENCE]
    context_tokens.extend(sequence_tokens)
    context_tokens.append(BEGIN_STRUCTURE)
    context_tokens.extend([THINK] * think_tokens)

    contacts = sorted(generation.contacts, key=lambda item: (item.seq_i, item.seq_j))
    target_tokens: list[Token] = []
    for contact in contacts:
        target_tokens.extend(
            (
                CONTACT,
                POSITIONS[contact.seq_i],
                POSITIONS[contact.seq_j],
            )
        )
    target_tokens.append(END)

    document_length = len(context_tokens) + len(target_tokens)
    if document_length > max_seq_len:
        raise ValueError(
            f"Full-attention document needs {document_length} tokens, "
            f"exceeding max_seq_len={max_seq_len}"
        )

    framing_count = 2
    sequence_count = len(sequence_tokens)
    suffix_context_count = 1 + think_tokens
    context_relative_positions = (
        (RELATIVE_POSITION.missing,) * framing_count
        + tuple(range(sequence_count))
        + (RELATIVE_POSITION.missing,) * suffix_context_count
    )
    context_document = Document(
        context_tokens,
        {RELATIVE_POSITION: context_relative_positions},
        attention=AttentionLayout.FULL,
    ).unscored()
    query_document = Document(
        [THINK] * len(target_tokens),
        {
            RELATIVE_POSITION: tuple(range(len(target_tokens))),
            QUERY: (True,) * len(target_tokens),
        },
        attention=AttentionLayout.FULL,
    ).with_targets(target_tokens)
    if target_scoring == ContactTargetScoring.UNORDERED_CONTACTS:
        query_document = query_document.scored_by(unordered_contacts_score)
    elif target_scoring != ContactTargetScoring.ORDERED_TOKENS:
        raise ValueError(f"Unsupported contact target scoring: {target_scoring}")
    return context_document + query_document


def block_causal_relative_document(
    generation: GenerationResult,
    *,
    think_tokens: int = 0,
    max_seq_len: int = CONTEXT_LENGTH,
    target_scoring: ContactTargetScoring = ContactTargetScoring.UNORDERED_CONTACTS,
) -> Document:
    """Build a bidirectional sequence prefix and ordered causal query blocks.

    The sequence/framing tokens form one full-attention block. Each optional
    think token is its own block. Each three-position contact query is one
    bidirectional block, followed by a singleton ``<end>`` query block. A block
    may attend to itself and every earlier block, but never to a later block.
    Contact labels remain scoring-only metadata; every query input is neutral.
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
    context_blocks = (0,) * len(prefix_tokens) + tuple(range(1, think_tokens + 1))

    contacts = sorted(generation.contacts, key=lambda item: (item.seq_i, item.seq_j))
    target_tokens: list[Token] = []
    query_blocks: list[int] = []
    for block, contact in enumerate(contacts):
        target_tokens.extend(
            (CONTACT, POSITIONS[contact.seq_i], POSITIONS[contact.seq_j])
        )
        query_blocks.extend((block, block, block))
    target_tokens.append(END)
    query_blocks.append(len(contacts))

    document_length = len(context_tokens) + len(target_tokens)
    if document_length > max_seq_len:
        raise ValueError(
            f"Block-causal document needs {document_length} tokens, "
            f"exceeding max_seq_len={max_seq_len}"
        )

    context_relative_positions = (
        (RELATIVE_POSITION.missing,) * 2
        + tuple(range(len(sequence_tokens)))
        + (RELATIVE_POSITION.missing,) * (1 + think_tokens)
    )
    context_document = Document(
        context_tokens,
        {
            RELATIVE_POSITION: context_relative_positions,
            ATTENTION_BLOCK: context_blocks,
        },
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).unscored()
    query_document = Document(
        [THINK] * len(target_tokens),
        {
            RELATIVE_POSITION: tuple(range(len(target_tokens))),
            QUERY: (True,) * len(target_tokens),
            ATTENTION_BLOCK: query_blocks,
        },
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).with_targets(target_tokens)
    if target_scoring == ContactTargetScoring.UNORDERED_CONTACTS:
        query_document = query_document.scored_by(unordered_contacts_score)
    elif target_scoring != ContactTargetScoring.ORDERED_TOKENS:
        raise ValueError(f"Unsupported contact target scoring: {target_scoring}")
    return context_document + query_document
