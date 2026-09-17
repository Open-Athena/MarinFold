# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from compute_contacts_delta_stream_documents import (
    AA_TO_TOKEN,
    CONTACTS_BEGIN_TOKEN_ID,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    STOP_TOKEN_ID,
    delta_to_token,
    document_row_from_analyzed,
)
from validate_delta_stream_documents import validate_row


def test_document_places_full_sequence_before_contacts() -> None:
    row = {
        "entry_id": "example",
        "seq_len": 3,
        "global_plddt": 90.0,
        "num_contacts": 1,
        "residue_resname": ["ALA", "GLY", "LEU"],
        "contact_seq_i": [0],
        "contact_seq_j": [2],
        "contact_degree": [1.0],
    }

    result = document_row_from_analyzed(
        row,
        source_shard=0,
        min_seq_separation=2,
        min_contact_degree=0.001,
        max_abs_delta=1024,
    )

    assert result["token_ids"] == [
        DOC_START_TOKEN_ID,
        AA_TO_TOKEN["ALA"],
        AA_TO_TOKEN["GLY"],
        AA_TO_TOKEN["LEU"],
        CONTACTS_BEGIN_TOKEN_ID,
        delta_to_token(2, max_abs_delta=1024),
        STOP_TOKEN_ID,
        STOP_TOKEN_ID,
        delta_to_token(-2, max_abs_delta=1024),
        STOP_TOKEN_ID,
        DOC_END_TOKEN_ID,
    ]
    assert result["sequence_token_count"] == 3
    assert result["contact_token_count"] == 5
    validate_row(result)
