# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contacts-v1 adapter for the reusable streaming document dataset."""

from collections.abc import Mapping
from typing import Any

from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    causal_document_from_generation,
)
from marinfold.document_structures.documents import Document
from marinfold_models.shard_documents import FixedQuotaShardDocumentDataset
from marinfold_models.shard_documents import (
    causal_lm_example_from_documents,
)


def contacts_v1_document_from_row(row: Mapping[str, Any]) -> Document | None:
    """Reconstruct the canonical serialized contacts-v1 training document."""
    analyzed = analyzed_from_row(row)
    generated = build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        global_plddt=analyzed.global_plddt,
    )
    if generated is None:
        return None
    return causal_document_from_generation(generated)


class FixedQuotaPremadeContactsDataset(FixedQuotaShardDocumentDataset):
    """Build a fixed number of canonical contacts-v1 examples per shard."""

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
            generate_document=contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=causal_lm_example_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
        )


__all__ = [
    "FixedQuotaPremadeContactsDataset",
    "contacts_v1_document_from_row",
]
