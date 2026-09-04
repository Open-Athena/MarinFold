# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contacts-v1 dataset adapters for exp177."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from haliax import Axis

from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    RELATIVE_POSITION,
    causal_document_from_generation,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    DOC_TYPE,
    END,
    POSITIONS,
    VOCABULARY,
)
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    QUERY,
    AttentionLayout,
    Document,
    pack,
)
from marinfold_models.document_loss import (
    CompactContactDocumentBatch,
    LevanterDocumentBatch,
    compact_contact_document_batch,
    levanter_document_batch,
)
from marinfold_models.shard_documents import (
    FixedQuotaShardDocumentDataset,
    PackedDocuments,
    causal_lm_example_from_documents,
)


def _generation_from_row(row: Mapping[str, Any]):
    analyzed = analyzed_from_row(row)
    return build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        global_plddt=analyzed.global_plddt,
    )


def causal_contacts_v1_document_from_row(row: Mapping[str, Any]) -> Document | None:
    """Reconstruct the canonical serialized contacts-v1 training document."""
    generated = _generation_from_row(row)
    if generated is None:
        return None
    return causal_document_from_generation(generated)


def soft_target_contacts_v1_document_from_row(row: Mapping[str, Any]) -> Document | None:
    """Build a compact block-causal contacts-v1 training document."""
    generated = _generation_from_row(row)
    if generated is None:
        return None

    sequence_tokens = [
        VOCABULARY.token(f"<{residue.resname}>") for residue in generated.residues
    ]
    prefix_tokens = [DOC_TYPE, BEGIN_SEQUENCE, *sequence_tokens, BEGIN_STRUCTURE]
    suffix_tokens = []
    for contact in generated.contacts:
        first, second = POSITIONS[contact.seq_i], POSITIONS[contact.seq_j]
        if contact.flipped:
            first, second = second, first
        suffix_tokens.extend((CONTACT, first, second))
    suffix_tokens.append(END)

    token_ids = (*prefix_tokens, *suffix_tokens)
    if len(token_ids) > CONTEXT_LENGTH:
        raise ValueError(
            f"Block-causal document needs {len(token_ids)} tokens, "
            f"exceeding max_seq_len={CONTEXT_LENGTH}"
        )

    prediction_start = len(prefix_tokens) - 1
    query = np.zeros(len(token_ids), dtype=np.bool_)
    query[prediction_start : prediction_start + len(suffix_tokens)] = True
    attention_blocks = (0,) * len(prefix_tokens) + tuple(
        range(1, len(suffix_tokens) + 1)
    )
    relative_positions = (
        (RELATIVE_POSITION.missing,) * 2
        + tuple(range(len(sequence_tokens)))
        + (RELATIVE_POSITION.missing,)
        + tuple(range(len(suffix_tokens)))
    )
    return Document(
        token_ids,
        {
            RELATIVE_POSITION: relative_positions,
            QUERY: query,
            ATTENTION_BLOCK: attention_blocks,
        },
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).unscored()


def document_batch_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> LevanterDocumentBatch:
    """Convert packed structured documents to one Levanter document batch item."""
    del max_segments_per_example
    packed = pack(documents, max_seq_len=max_seq_len)
    if packed.token_ids.shape[0] != 1:
        raise AssertionError("Shard packing bin unexpectedly produced multiple rows")
    return levanter_document_batch(packed, Pos=Axis("position", max_seq_len))


def compact_contact_batch_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> CompactContactDocumentBatch:
    """Convert one compact contacts-v1 document to a Levanter batch item."""
    del max_segments_per_example
    if len(documents) != 1:
        raise ValueError(
            f"Compact soft-target batches require one document, got {len(documents)}"
        )
    packed = pack(documents, max_seq_len=max_seq_len)
    if packed.token_ids.shape[0] != 1:
        raise AssertionError("Shard packing bin unexpectedly produced multiple rows")
    return compact_contact_document_batch(packed, Pos=Axis("position", max_seq_len))


class FixedQuotaPremadeContactsDataset(FixedQuotaShardDocumentDataset):
    """Build fixed-quota canonical contacts-v1 examples from premade contacts."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        shard_cache_size: int = 2,
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=causal_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=causal_lm_example_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
        )


class FixedQuotaSoftTargetContactsDataset(FixedQuotaShardDocumentDataset):
    """Build fixed-quota soft-target contacts-v1 examples from premade contacts."""

    def _construct_shard(
        self, epoch: int, shard_index: int
    ) -> tuple[PackedDocuments | None, ...]:
        slots = super()._construct_shard(epoch, shard_index)
        if all(slot is not None for slot in slots):
            return slots

        real_slots = tuple(slot for slot in slots if slot is not None)
        if not real_slots:
            raise ValueError(f"Shard {shard_index} yielded no soft-target packs")

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, shard_index, 2])
        )
        return tuple(
            slot if slot is not None else real_slots[int(rng.integers(len(real_slots)))]
            for slot in slots
        )

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 1,
        shard_cache_size: int = 2,
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=soft_target_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=compact_contact_batch_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
        )


__all__ = [
    "FixedQuotaPremadeContactsDataset",
    "FixedQuotaSoftTargetContactsDataset",
    "causal_contacts_v1_document_from_row",
    "compact_contact_batch_from_documents",
    "soft_target_contacts_v1_document_from_row",
]
