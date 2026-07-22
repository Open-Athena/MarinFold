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
    ContactTargetScoring,
    DocumentConstructionConfig,
    build_contact_training_document,
    causal_document_from_generation,
    unordered_contacts_score,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    DOC_TYPE_TOKEN,
    END_TOKEN,
    THINK_TOKEN,
    VOCABULARY,
    position_token,
)
from marinfold.document_structures.documents import (
    POSITION_IDS,
    QUERY,
    AttentionLayout,
    ScoreContext,
    next_token_score,
)


def test_causal_document_matches_existing_contacts_serialization() -> None:
    residues = tuple(
        ResidueInfo(index, name, index + 1, "A")
        for index, name in enumerate(
            ("MET", "ALA", "GLY", "PHE", "SER", "THR", "LYS", "VAL")
        )
    )
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
    assert document.score_ranges[0].scorer is next_token_score
    assert document.score_ranges[0].target_ids is None


def test_full_attention_document_uses_natural_relative_sequence_positions() -> None:
    residues = tuple(
        ResidueInfo(index, name, index + 1, "A")
        for index, name in enumerate(
            ("MET", "ALA", "GLY", "PHE", "SER", "THR", "LYS", "VAL")
        )
    )
    generation = build_document(
        "ex-full-attention",
        residues,
        (RawContact(1, 7, 0.42), RawContact(0, 6, 0.91)),
    )
    assert generation is not None
    tokenizer = build_tokenizer(VOCABULARY)
    config = DocumentConstructionConfig(
        style=ContactDocumentStyle.FULL_ATTENTION_RELATIVE,
        think_tokens=2,
    )

    document = build_contact_training_document(generation, config=config)

    context_tokens = [
        DOC_TYPE_TOKEN,
        BEGIN_SEQUENCE_TOKEN,
        *(f"<{residue.resname}>" for residue in residues),
        BEGIN_STRUCTURE_TOKEN,
        THINK_TOKEN,
        THINK_TOKEN,
    ]
    target_tokens = [
        CONTACT_TOKEN,
        position_token(0),
        position_token(6),
        CONTACT_TOKEN,
        position_token(1),
        position_token(7),
        END_TOKEN,
    ]
    context_ids = tokenizer.encode(" ".join(context_tokens), add_special_tokens=False)
    target_ids = tokenizer.encode(" ".join(target_tokens), add_special_tokens=False)
    query_id = tokenizer.encode(THINK_TOKEN, add_special_tokens=False)[0]

    assert document.attention == AttentionLayout.FULL
    assert document.vocabulary == VOCABULARY.identity
    assert tuple(document.token_ids) == tuple(
        context_ids + [query_id] * len(target_ids)
    )
    assert tuple(document[POSITION_IDS]) == (0,) * len(document)
    assert tuple(document[RELATIVE_POSITION]) == (
        -1,
        -1,
        *range(len(residues)),
        -1,
        -1,
        -1,
        *range(len(target_ids)),
    )
    assert tuple(document[QUERY]) == (False,) * len(context_ids) + (True,) * len(
        target_ids
    )
    assert len(document.score_ranges) == 2
    context_range, target_range = document.score_ranges
    assert (context_range.start, context_range.stop, context_range.scorer) == (
        0,
        len(context_ids),
        None,
    )
    assert (target_range.start, target_range.stop) == (
        len(context_ids),
        len(document),
    )
    assert target_range.scorer is next_token_score
    assert target_range.target_ids == tuple(target_ids)


def test_full_attention_document_ignores_serialization_randomization() -> None:
    residues = tuple(
        ResidueInfo(index, name, index + 1, "A")
        for index, name in enumerate(
            ("MET", "ALA", "GLY", "PHE", "SER", "THR", "LYS", "VAL")
        )
    )
    contacts = (RawContact(1, 7, 0.42), RawContact(0, 6, 0.91))
    first = build_document("randomization-seed-a", residues, contacts)
    second = build_document("randomization-seed-b", residues, contacts)
    assert first is not None
    assert second is not None
    assert first.document != second.document
    config = DocumentConstructionConfig(
        style=ContactDocumentStyle.FULL_ATTENTION_RELATIVE
    )

    first_document = build_contact_training_document(first, config=config)
    second_document = build_contact_training_document(second, config=config)

    assert first_document == second_document


def test_full_attention_document_can_score_unordered_contacts() -> None:
    residues = tuple(
        ResidueInfo(index, name, index + 1, "A")
        for index, name in enumerate(
            ("MET", "ALA", "GLY", "PHE", "SER", "THR", "LYS", "VAL")
        )
    )
    generation = build_document(
        "ex-unordered",
        residues,
        (RawContact(0, 6, 0.91), RawContact(1, 7, 0.42)),
    )
    assert generation is not None
    config = DocumentConstructionConfig(
        style=ContactDocumentStyle.FULL_ATTENTION_RELATIVE,
        target_scoring=ContactTargetScoring.UNORDERED_CONTACTS,
    )

    document = build_contact_training_document(generation, config=config)

    assert len(document.score_ranges) == 2
    assert document.score_ranges[0].scorer is None
    assert document.score_ranges[1].scorer is unordered_contacts_score


def test_unordered_contacts_score_ignores_contact_order_and_orientation() -> None:
    logits = np.zeros((7, 16), dtype=np.float32)
    predicted_targets = (5, 9, 8, 5, 6, 7, 2)
    for position, target in enumerate(predicted_targets):
        logits[position, target] = 20.0

    canonical_targets = (5, 6, 7, 5, 8, 9, 2)
    reordered_reversed_targets = (5, 9, 8, 5, 7, 6, 2)
    canonical_loss = unordered_contacts_score(
        logits,
        ScoreContext(token_ids=np.zeros(7), target_ids=canonical_targets),
    )
    reordered_loss = unordered_contacts_score(
        logits,
        ScoreContext(token_ids=np.zeros(7), target_ids=reordered_reversed_targets),
    )
    reordered_logits = logits[[3, 4, 5, 0, 1, 2, 6]]
    reordered_prediction_loss = unordered_contacts_score(
        reordered_logits,
        ScoreContext(token_ids=np.zeros(7), target_ids=canonical_targets),
    )

    assert float(canonical_loss) < 1e-6
    np.testing.assert_allclose(canonical_loss, reordered_loss, atol=1e-7)
    np.testing.assert_allclose(canonical_loss, reordered_prediction_loss, atol=1e-7)


def test_unordered_contacts_score_uses_each_gold_contact_once() -> None:
    logits = np.zeros((7, 8), dtype=np.float32)
    predicted_targets = (5, 1, 2, 5, 1, 2, 6)
    for position, target in enumerate(predicted_targets):
        logits[position, target] = 10.0

    distinct_contact_loss = unordered_contacts_score(
        logits,
        ScoreContext(token_ids=np.zeros(7), target_ids=(5, 1, 2, 5, 3, 4, 6)),
    )
    duplicate_contact_loss = unordered_contacts_score(
        logits,
        ScoreContext(token_ids=np.zeros(7), target_ids=(5, 1, 2, 5, 1, 2, 6)),
    )

    assert float(distinct_contact_loss) > float(duplicate_contact_loss) + 2.0


def test_unordered_contacts_score_handles_no_contacts() -> None:
    logits = np.zeros((1, 8), dtype=np.float32)
    logits[0, 2] = 20.0

    loss = unordered_contacts_score(
        logits,
        ScoreContext(token_ids=np.zeros(1), target_ids=(2,)),
    )

    assert float(loss) < 1e-6


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
