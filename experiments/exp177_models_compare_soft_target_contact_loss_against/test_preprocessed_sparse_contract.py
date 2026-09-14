# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exp177 Zephyr-preprocessed soft-target row contract.

The previous exp177 bug lived at this boundary: the Zephyr producer emitted rows
whose token-space/position metadata did not match the ordinary contacts-v1
training format.  These tests start from analyzed rows, pass through the exact
producer helpers, and then read the row through the sparse precomputed dataset
used by training.
"""

import sys
from collections import Counter
from pathlib import Path

import haliax as hax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "marinfold"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from marinfold.document_structures.contacts_v1.generate import build_document  # noqa: E402
from marinfold.document_structures.contacts_v1.parse import (  # noqa: E402
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyzed_to_row,
)
from marinfold.document_structures.contacts_v1.vocab import (  # noqa: E402
    BEGIN_STRUCTURE,
    CONTACT,
    END,
    POSITIONS,
    VOCABULARY,
)
from premade_contacts_dataset import (  # noqa: E402
    SparsePrecomputedSoftTargetContactsDataset,
    _sparse_second_endpoint_targets,
    soft_target_contacts_v1_document_from_row,
)
from preprocess_soft_targets import _row_from_document  # noqa: E402
from test_sparse_loss_exact_oracle import (  # noqa: E402
    FixedActivationModel,
    batch_from_raw,
    dense_suffix_oracle_loss,
)


def _toy_residues():
    return tuple(
        ResidueInfo(seq_index=i, resname="ALA", resnum=i + 1, chain="A")
        for i in range(18)
    )


def _toy_contacts():
    # All pass contacts-v1's default min_seq_separation=6.  The builder may
    # still randomize order/orientation/position numbering from entry_id; the
    # test intentionally treats build_document as the source of truth.
    return (
        RawContact(seq_i=0, seq_j=7, degree=1.0),
        RawContact(seq_i=0, seq_j=8, degree=0.9),
        RawContact(seq_i=7, seq_j=15, degree=0.8),
        RawContact(seq_i=8, seq_j=16, degree=0.7),
    )


def _analyzed_row(entry_id: str, contacts: tuple[RawContact, ...]):
    residues = _toy_residues()
    analyzed = AnalyzedStructure(
        entry_id=entry_id,
        source_path="toy.cif",
        residues=residues,
        contacts=contacts,
        global_plddt=90.0,
    )
    return analyzed_to_row(analyzed), residues, contacts


def _toy_analyzed_row():
    return _analyzed_row("exp177-preprocess-contract-toy", _toy_contacts())


def _expected_serialized_edges(entry_id: str, residues, contacts) -> tuple[tuple[int, int], ...]:
    generated = build_document(entry_id, residues, contacts, global_plddt=90.0)
    assert generated is not None
    return tuple(
        (
            int(POSITIONS[contact.pos_j if contact.flipped else contact.pos_i]),
            int(POSITIONS[contact.pos_i if contact.flipped else contact.pos_j]),
        )
        for contact in generated.contacts
    )


def _preprocessed_row(max_seq_len: int = 128):
    row, residues, contacts = _toy_analyzed_row()
    document = soft_target_contacts_v1_document_from_row(row)
    assert document is not None
    return row, residues, contacts, document, _row_from_document(
        document,
        source_shard=17,
        slot_index=23,
        max_seq_len=max_seq_len,
    )


def test_preprocessed_row_round_trips_generated_contacts_v1_suffix():
    row, residues, contacts, document, out = _preprocessed_row()
    expected_edges = _expected_serialized_edges("exp177-preprocess-contract-toy", residues, contacts)
    expected_prefix_text = build_document(
        "exp177-preprocess-contract-toy",
        residues,
        contacts,
        global_plddt=90.0,
    ).document.split()
    expected_prefix_text = expected_prefix_text[: expected_prefix_text.index(BEGIN_STRUCTURE.text) + 1]
    expected_prefix_ids = [int(VOCABULARY.token(token)) for token in expected_prefix_text]

    token_ids = [int(token) for token in out["token_ids"]]
    prediction_start = int(out["prediction_start"])
    contact_count = int(out["contact_count"])
    end_position = prediction_start + 3 * contact_count + 1

    assert token_ids[: len(expected_prefix_ids)] == expected_prefix_ids
    assert prediction_start == len(expected_prefix_ids) - 1
    assert token_ids[prediction_start] == int(BEGIN_STRUCTURE)
    assert token_ids[end_position] == int(END)
    assert out["position_ids"] == list(range(len(out["position_ids"])))
    assert out["segment_ids"][: end_position + 1] == [0] * (end_position + 1)
    assert out["segment_ids"][end_position + 1 :] == [-1] * (len(out["segment_ids"]) - end_position - 1)
    assert out["attention_blocks"][: prediction_start + 1] == [0] * (prediction_start + 1)
    assert out["attention_blocks"][prediction_start + 1 : end_position + 1] == list(range(1, 3 * contact_count + 2))
    assert out["target_position_count"] == 3 * len(expected_edges) + 1

    actual_edges = tuple(
        (int(out["contact_first_ids"][i]), int(out["contact_second_ids"][i]))
        for i in range(contact_count)
    )
    assert actual_edges == expected_edges
    for contact_index, (first, second) in enumerate(expected_edges):
        contact_position = prediction_start + 1 + 3 * contact_index
        assert token_ids[contact_position] == int(CONTACT)
        assert token_ids[contact_position + 1] == first
        assert token_ids[contact_position + 2] == second


def test_preprocessed_row_reader_preserves_tokens_positions_and_sparse_neighbors():
    _row, residues, contacts, _document, out = _preprocessed_row()
    expected_edges = _expected_serialized_edges("exp177-preprocess-contract-toy", residues, contacts)
    dataset = SparsePrecomputedSoftTargetContactsDataset(
        data_prefix="memory://unused",
        num_shards=1,
        examples_per_shard=1,
        max_seq_len=128,
        max_sparse_contacts=16,
        max_sparse_degree=16,
    )
    raw = dataset._raw_example_from_row(out)
    batch = dataset._batch_from_raw(raw)

    np.testing.assert_array_equal(np.asarray(batch.tokens.array), np.asarray(out["token_ids"], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch.position_ids.array), np.arange(128, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch.segment_ids.array), np.asarray(out["segment_ids"], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch.attention_blocks.array), np.asarray(out["attention_blocks"], dtype=np.int32))
    assert int(batch.contact_count) == len(expected_edges)
    assert int(batch.target_position_count) == 3 * len(expected_edges) + 1

    neighbor_ids = np.asarray(batch.second_neighbor_ids)
    neighbor_counts = np.asarray(batch.second_neighbor_counts)
    neighbor_count = np.asarray(batch.second_neighbor_count)
    for contact_index, (observed_first, _observed_second) in enumerate(expected_edges):
        remaining = expected_edges[contact_index:]
        expected_neighbors: Counter[int] = Counter()
        for first, second in remaining:
            if first == observed_first:
                expected_neighbors[second] += 1
            if second == observed_first:
                expected_neighbors[first] += 1
        actual_neighbors = Counter(
            {
                int(token_id): int(count)
                for token_id, count in zip(neighbor_ids[contact_index], neighbor_counts[contact_index], strict=True)
                if count
            }
        )
        assert int(neighbor_count[contact_index]) == sum(expected_neighbors.values())
        assert actual_neighbors == expected_neighbors


def test_preprocessed_row_sparse_loss_matches_dense_oracle():
    _row, residues, contacts, _document, out = _preprocessed_row()
    expected_edges = _expected_serialized_edges("exp177-preprocess-contract-toy", residues, contacts)
    dataset = SparsePrecomputedSoftTargetContactsDataset(
        data_prefix="memory://unused",
        num_shards=1,
        examples_per_shard=1,
        max_seq_len=128,
        max_sparse_contacts=16,
        max_sparse_degree=16,
    )
    raw = dataset._raw_example_from_row(out)
    batch = batch_from_raw(raw, max_contacts=16, max_degree=16)

    rng = np.random.default_rng(177)
    embed_size = 13
    vocab_size = 2845
    activations = rng.normal(size=(1, 128, embed_size)).astype(np.float32)
    lm_head = rng.normal(size=(embed_size, vocab_size)).astype(np.float32) / np.sqrt(embed_size)
    model = FixedActivationModel(
        jnp.asarray(activations),
        jnp.asarray(lm_head),
        hax.Axis("embed", embed_size),
        hax.Axis("vocab", vocab_size),
    )

    from marinfold_models.document_loss import sparse_contact_document_loss

    actual = float(sparse_contact_document_loss(model, batch))
    expected = dense_suffix_oracle_loss(
        activations[0],
        lm_head,
        expected_edges,
        prediction_start=int(out["prediction_start"]),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def _row_for_entry(entry_id: str, contacts: tuple[RawContact, ...], *, max_seq_len: int = 128):
    row, residues, contacts = _analyzed_row(entry_id, contacts)
    document = soft_target_contacts_v1_document_from_row(row)
    assert document is not None
    out = _row_from_document(document, source_shard=17, slot_index=23, max_seq_len=max_seq_len)
    return residues, contacts, out, _expected_serialized_edges(entry_id, residues, contacts)


def _stack_raw_examples(raws, *, max_contacts: int, max_degree: int):
    from marinfold_models.document_loss import SparseContactDocumentBatch

    sparse = [
        _sparse_second_endpoint_targets(raw, max_contacts=max_contacts, max_degree=max_degree)
        for raw in raws
    ]
    Batch = hax.Axis("batch", len(raws))
    Pos = hax.Axis("position", raws[0].token_ids.shape[0])
    return SparseContactDocumentBatch(
        tokens=hax.named(jnp.asarray(np.stack([raw.token_ids for raw in raws])), (Batch, Pos)),
        contact_first_ids=jnp.asarray(np.stack([item[0] for item in sparse])),
        contact_second_ids=jnp.asarray(np.stack([item[1] for item in sparse])),
        second_neighbor_ids=jnp.asarray(np.stack([item[2] for item in sparse])),
        second_neighbor_counts=jnp.asarray(np.stack([item[3] for item in sparse])),
        second_neighbor_count=jnp.asarray(np.stack([item[4] for item in sparse])),
        contact_count=jnp.asarray([raw.contact_count for raw in raws], dtype=jnp.int32),
        prediction_start=jnp.asarray([raw.prediction_start for raw in raws], dtype=jnp.int32),
        position_ids=hax.named(jnp.asarray(np.stack([raw.position_ids for raw in raws])), (Batch, Pos)),
        segment_ids=hax.named(jnp.asarray(np.stack([raw.segment_ids for raw in raws])), (Batch, Pos)),
        attention_blocks=hax.named(jnp.asarray(np.stack([raw.attention_blocks for raw in raws])), (Batch, Pos)),
        target_position_count=jnp.asarray([raw.target_position_count for raw in raws], dtype=jnp.int32),
        vocabulary=None,
    )


def test_batched_preprocessed_sparse_loss_uses_target_position_denominator():
    contacts_many = _toy_contacts()
    contacts_one = (_toy_contacts()[0],)
    _residues_a, _contacts_a, out_a, edges_a = _row_for_entry("exp177-preprocess-batch-many", contacts_many)
    _residues_b, _contacts_b, out_b, edges_b = _row_for_entry("exp177-preprocess-batch-one", contacts_one)

    dataset = SparsePrecomputedSoftTargetContactsDataset(
        data_prefix="memory://unused",
        num_shards=1,
        examples_per_shard=1,
        max_seq_len=128,
        max_sparse_contacts=16,
        max_sparse_degree=16,
    )
    raw_a = dataset._raw_example_from_row(out_a)
    raw_b = dataset._raw_example_from_row(out_b)
    batch = _stack_raw_examples([raw_a, raw_b], max_contacts=16, max_degree=16)

    rng = np.random.default_rng(178)
    embed_size = 13
    vocab_size = 2845
    activations = rng.normal(size=(2, 128, embed_size)).astype(np.float32)
    lm_head = rng.normal(size=(embed_size, vocab_size)).astype(np.float32) / np.sqrt(embed_size)
    model = FixedActivationModel(
        jnp.asarray(activations),
        jnp.asarray(lm_head),
        hax.Axis("embed", embed_size),
        hax.Axis("vocab", vocab_size),
    )

    from marinfold_models.document_loss import sparse_contact_document_loss

    actual = float(sparse_contact_document_loss(model, batch))
    expected_sum_a = dense_suffix_oracle_loss(
        activations[0], lm_head, edges_a, prediction_start=int(out_a["prediction_start"])
    ) * int(out_a["target_position_count"])
    expected_sum_b = dense_suffix_oracle_loss(
        activations[1], lm_head, edges_b, prediction_start=int(out_b["prediction_start"])
    ) * int(out_b["target_position_count"])
    expected = (expected_sum_a + expected_sum_b) / (int(out_a["target_position_count"]) + int(out_b["target_position_count"]))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
