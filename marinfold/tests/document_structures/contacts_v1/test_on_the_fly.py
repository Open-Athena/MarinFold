# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1.generate import build_document
from marinfold.document_structures.contacts_v1.on_the_fly import (
    iter_afdb_records_from_parquet,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_v1.training_documents import (
    RELATIVE_POSITION,
    ContactDocumentStyle,
    DocumentConstructionConfig,
    build_contact_training_document,
    causal_document_from_generation,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    DOC_TYPE,
    END,
    POSITIONS,
    THINK,
    VOCABULARY,
)
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    QUERY,
    AttentionLayout,
)


def _residues(count: int) -> tuple[ResidueInfo, ...]:
    names = ("MET", "ALA", "GLY", "PHE", "SER", "THR", "LYS", "VAL")
    return tuple(
        ResidueInfo(index, names[index % len(names)], index + 1, "A")
        for index in range(count)
    )


def _explicit_range(document):
    return next(
        target_range
        for target_range in document.score_ranges
        if target_range.has_explicit_targets
    )


def test_causal_document_matches_existing_contacts_serialization() -> None:
    residues = _residues(8)
    generation = build_document(
        "ex-21",
        residues,
        (RawContact(0, 6, 0.91), RawContact(1, 7, 0.42)),
    )
    assert generation is not None
    tokenizer = build_tokenizer(VOCABULARY)

    document = causal_document_from_generation(generation)
    expected_ids = tuple(tokenizer.convert_tokens_to_ids(generation.document.split()))
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None

    assert tuple(document.token_ids) == (*expected_ids, eos_token_id)
    assert tuple(document[QUERY]) == (True,) * len(expected_ids) + (False,)
    assert len(document.score_ranges) == 1
    assert document.score_ranges[0].scored
    assert not document.score_ranges[0].has_explicit_targets


def test_block_causal_document_teacher_forces_contact_tokens() -> None:
    residues = _residues(8)
    generation = build_document(
        "ex-block-causal",
        residues,
        (RawContact(0, 6, 0.91), RawContact(1, 7, 0.42)),
    )
    assert generation is not None
    config = DocumentConstructionConfig(
        style=ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE,
        think_tokens=2,
    )

    document = build_contact_training_document(generation, config=config)

    sequence_tokens = tuple(
        VOCABULARY.token(f"<{residue.resname}>") for residue in residues
    )
    prefix = (DOC_TYPE, BEGIN_SEQUENCE, *sequence_tokens, BEGIN_STRUCTURE)
    suffix = []
    for contact in generation.contacts:
        endpoints = (POSITIONS[contact.seq_i], POSITIONS[contact.seq_j])
        if contact.flipped:
            endpoints = endpoints[::-1]
        suffix.extend((CONTACT, *endpoints))
    suffix.append(END)
    context = (*prefix, THINK, THINK)
    prediction_start = len(context) - 1

    assert document.attention == AttentionLayout.BLOCK_CAUSAL
    assert tuple(document.token_ids) == tuple(
        int(token) for token in (*context, *suffix)
    )
    assert tuple(document[ATTENTION_BLOCK]) == (
        (0,) * len(prefix) + (1, 2) + tuple(range(3, 3 + len(suffix)))
    )
    assert tuple(document[QUERY]) == (
        (False,) * prediction_start + (True,) * len(suffix) + (False,)
    )
    assert tuple(document[RELATIVE_POSITION]) == (
        (-1, -1)
        + tuple(range(len(residues)))
        + (-1, -1, -1)
        + tuple(range(len(suffix)))
    )

    target_range = _explicit_range(document)
    assert (target_range.start, target_range.stop) == (
        prediction_start,
        len(document) - 1,
    )
    assert target_range.target_weights is not None
    np.testing.assert_allclose(target_range.target_weights.sum(axis=1), 1.0)
    target_index = {
        target_id: index for index, target_id in enumerate(target_range.target_ids)
    }
    for relative_position, actual_target in enumerate(suffix):
        assert (
            target_range.target_weights[
                relative_position, target_index[int(actual_target)]
            ]
            > 0
        )


def test_first_endpoint_distribution_matches_remaining_contact_counts() -> None:
    residues = _residues(16)
    generation = build_document(
        "degree-weighted",
        residues,
        (
            RawContact(5, 11, 1.0),
            RawContact(5, 12, 0.9),
            RawContact(5, 13, 0.8),
            RawContact(5, 14, 0.7),
            RawContact(2, 10, 0.6),
        ),
    )
    assert generation is not None
    document = build_contact_training_document(
        generation,
        config=DocumentConstructionConfig(
            style=ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE
        ),
    )

    target_range = _explicit_range(document)
    assert target_range.target_weights is not None
    target_index = {
        target_id: index for index, target_id in enumerate(target_range.target_ids)
    }
    first_endpoint_weights = target_range.target_weights[1]
    pos5_weight = first_endpoint_weights[target_index[int(POSITIONS[5])]]
    pos2_weight = first_endpoint_weights[target_index[int(POSITIONS[2])]]

    np.testing.assert_allclose(pos5_weight, 4.0 * pos2_weight)


def test_later_distributions_remove_teacher_forced_contacts() -> None:
    residues = _residues(16)
    generation = build_document(
        "without-replacement",
        residues,
        (
            RawContact(5, 11, 1.0),
            RawContact(5, 12, 0.9),
            RawContact(5, 13, 0.8),
        ),
    )
    assert generation is not None
    document = build_contact_training_document(
        generation,
        config=DocumentConstructionConfig(
            style=ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE
        ),
    )
    target_range = _explicit_range(document)
    assert target_range.target_weights is not None
    target_index = {
        target_id: index for index, target_id in enumerate(target_range.target_ids)
    }

    remaining = list(generation.contacts)
    for slot, contact in enumerate(generation.contacts):
        first, second = (POSITIONS[contact.seq_i], POSITIONS[contact.seq_j])
        if contact.flipped:
            first, second = second, first
        second_endpoint_row = target_range.target_weights[3 * slot + 2]
        expected_neighbors = {
            other.seq_j if int(first) == int(POSITIONS[other.seq_i]) else other.seq_i
            for other in remaining
            if int(first) in (int(POSITIONS[other.seq_i]), int(POSITIONS[other.seq_j]))
        }
        positive_neighbors = {
            index
            for index in range(len(residues))
            if int(POSITIONS[index]) in target_index
            and second_endpoint_row[target_index[int(POSITIONS[index])]] > 0
        }
        assert positive_neighbors == expected_neighbors
        assert int(second) in target_index
        remaining.pop(0)


def test_no_contacts_predicts_end_from_last_context_token() -> None:
    generation = build_document("no-contacts", _residues(8), ())
    assert generation is not None
    document = build_contact_training_document(
        generation,
        config=DocumentConstructionConfig(
            style=ContactDocumentStyle.BLOCK_CAUSAL_RELATIVE
        ),
    )

    target_range = _explicit_range(document)
    assert target_range.target_ids == (int(CONTACT), int(END))
    assert target_range.target_weights is not None
    end_index = target_range.target_ids.index(int(END))
    assert target_range.target_weights.shape == (1, 2)
    assert target_range.target_weights[0, end_index] == 1.0
    assert int(document.token_ids[-1]) == int(END)


def test_afdb_manifest_projection_preserves_split_and_structure_pointer(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "entry_id": "AF-TRAIN-F1",
            "gcs_uri": "gs://example/AF-TRAIN-F1-model_v4.cif",
            "split": "train",
            "uniprot_accession": "TRAIN",
            "tax_id": 1,
            "organism_name": "training organism",
            "global_plddt": 91.5,
            "seq_len": 80,
            "seq_cluster_id": "TRAIN",
            "struct_cluster_id": "TRAIN",
        },
        {
            "entry_id": "AF-VAL-F1",
            "gcs_uri": "gs://example/AF-VAL-F1-model_v4.cif",
            "split": "val",
            "uniprot_accession": "VAL",
            "tax_id": 2,
            "organism_name": "validation organism",
            "global_plddt": 88.0,
            "seq_len": 120,
            "seq_cluster_id": "VAL",
            "struct_cluster_id": "VAL",
        },
    ]
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)

    records = tuple(iter_afdb_records_from_parquet(shard, split="val"))

    assert len(records) == 1
    assert records[0].entry_id == "AF-VAL-F1"
    assert records[0].gcs_uri == "gs://example/AF-VAL-F1-model_v4.cif"
    assert records[0].split == "val"
