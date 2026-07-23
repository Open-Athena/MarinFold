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
from marinfold_models.streaming_documents import StreamingDocumentDataset
from marinfold_models.streaming_documents import (
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


class StreamingPremadeContactsDataset(StreamingDocumentDataset):
    """Stream canonical contacts-v1 documents from premade contact rows."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        min_fill_fraction: float = 0.99,
        max_open_packs: int = 256,
        row_block_size: int = 256,
        process_index: int | None = None,
        process_count: int | None = None,
        global_batch_size: int | None = None,
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=contacts_v1_document_from_row,
            # Change this identifier if construction semantics change. Loader
            # checkpoints reject mismatches instead of mixing document formats.
            generator_id="contacts-v1/causal-serialized/v1",
            num_shards=num_shards,
            total_shards=total_shards,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=causal_lm_example_from_documents,
            max_segments_per_example=max_segments_per_example,
            min_fill_fraction=min_fill_fraction,
            max_open_packs=max_open_packs,
            row_block_size=row_block_size,
            process_index=process_index,
            process_count=process_count,
            global_batch_size=global_batch_size,
        )


__all__ = [
    "StreamingPremadeContactsDataset",
    "contacts_v1_document_from_row",
]
