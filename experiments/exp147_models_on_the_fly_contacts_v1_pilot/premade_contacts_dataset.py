# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contacts-v1 adapter for the reusable streaming document dataset."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from any_permissible import GrugContactOracleExample, contact_edge_capacity
from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    causal_document_from_generation,
)
from marinfold.document_structures.contacts_v1.vocab import CONTACT
from marinfold.document_structures.documents import Document
from marinfold_models.shard_documents import (
    FixedQuotaShardDocumentDataset,
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


def contact_oracle_lm_example_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> GrugContactOracleExample:
    """Attach fixed-shape edge slots to the otherwise unchanged causal example."""
    base = causal_lm_example_from_documents(
        documents,
        max_seq_len,
        max_segments_per_example,
    )
    if base.attn_mask.segment_ids is None:
        raise AssertionError("Packed causal contacts example has no segment ids")

    tokens = np.asarray(base.tokens)
    segment_ids = np.asarray(base.attn_mask.segment_ids[0])
    contact_positions = np.flatnonzero(tokens == int(CONTACT)).astype(np.int32)
    max_edges = contact_edge_capacity(max_seq_len)
    if contact_positions.size > max_edges:
        raise AssertionError(
            f"Packed example contains {contact_positions.size} contacts, "
            f"exceeding fixed edge capacity {max_edges}"
        )
    if np.any(contact_positions + 2 >= max_seq_len):
        raise AssertionError("Contact statement extends beyond the packed example")
    if np.any(segment_ids[contact_positions] != segment_ids[contact_positions + 2]):
        raise AssertionError("Contact statement crosses a document segment")

    edge_positions = np.zeros(max_edges, dtype=np.int32)
    edge_segment_ids = np.full(max_edges, -1, dtype=np.int32)
    edge_valid = np.zeros(max_edges, dtype=np.bool_)
    edge_count = contact_positions.size
    edge_positions[:edge_count] = contact_positions
    edge_segment_ids[:edge_count] = segment_ids[contact_positions]
    edge_valid[:edge_count] = True
    return GrugContactOracleExample(
        tokens=base.tokens,
        loss_weight=base.loss_weight,
        attn_mask=base.attn_mask,
        edge_positions=jnp.asarray(edge_positions),
        edge_segment_ids=jnp.asarray(edge_segment_ids),
        edge_valid=jnp.asarray(edge_valid),
    )


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
        any_permissible_loss: bool = False,
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
            example_builder=(
                contact_oracle_lm_example_from_documents
                if any_permissible_loss
                else causal_lm_example_from_documents
            ),
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
        )


__all__ = [
    "FixedQuotaPremadeContactsDataset",
    "contact_oracle_lm_example_from_documents",
    "contacts_v1_document_from_row",
]
