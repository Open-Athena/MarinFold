# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contacts-v1 dataset adapters for exp177."""

from collections.abc import Mapping
from typing import Any

from haliax import Axis

from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    ContactDocumentStyle,
    DocumentConstructionConfig,
    build_contact_training_document,
    causal_document_from_generation,
)
from marinfold.document_structures.documents import Document, pack
from marinfold_models.document_loss import LevanterDocumentBatch, levanter_document_batch
from marinfold_models.shard_documents import (
    FixedQuotaShardDocumentDataset,
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
    """Build the block-causal soft-target contacts-v1 training document."""
    generated = _generation_from_row(row)
    if generated is None:
        return None
    return build_contact_training_document(
        generated,
        config=DocumentConstructionConfig(
            style=ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE,
            max_seq_len=CONTEXT_LENGTH,
        ),
    )


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
            generate_document=soft_target_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=document_batch_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
        )


__all__ = [
    "FixedQuotaPremadeContactsDataset",
    "FixedQuotaSoftTargetContactsDataset",
    "causal_contacts_v1_document_from_row",
    "soft_target_contacts_v1_document_from_row",
]
