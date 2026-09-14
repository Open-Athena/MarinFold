# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense-oracle tests for exp177's production sparse contact loss path.

These are ported from exp279's exact-target tests, but intentionally exercise the
exp177 production stack: precomputed sparse target construction,
``SparseContactDocumentBatch``, and ``sparse_contact_document_loss``.  The oracle
is independent and enumerates the categorical next-token distribution implied by
an unordered remaining contact set for each serialized order/orientation.
"""

import sys
from collections import Counter
from itertools import permutations, product
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "marinfold"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from marinfold.document_structures.contacts_v1.vocab import (  # noqa: E402
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    DOC_TYPE,
    END,
)
from marinfold_models.document_loss import (  # noqa: E402
    SparseContactDocumentBatch,
    sparse_contact_document_loss,
)
from premade_contacts_dataset import (  # noqa: E402
    RawPrecomputedSoftTargetExample,
    _sparse_second_endpoint_targets,
)


class FixedActivationModel(eqx.Module):
    """Tiny LM-head-compatible model with fixed activations and head weights."""

    activations_array: jax.Array
    lm_head_array: jax.Array
    Embed: hax.Axis = eqx.field(static=True)
    Vocab: hax.Axis = eqx.field(static=True)

    def activations(self, tokens, attention_mask, *, key=None, pos_ids=None):
        del tokens, attention_mask, key, pos_ids
        Batch = hax.Axis("batch", self.activations_array.shape[0])
        Pos = hax.Axis("position", self.activations_array.shape[1])
        return hax.named(self.activations_array, (Batch, Pos, self.Embed))

    def get_lm_head(self):
        return hax.named(self.lm_head_array, (self.Embed, self.Vocab))


def serialized_documents(edges: tuple[tuple[int, int], ...]):
    """Yield every order/orientation of an undirected contact set."""
    for order in permutations(edges):
        for flips in product((False, True), repeat=len(order)):
            yield tuple(edge[::-1] if flip else edge for edge, flip in zip(order, flips, strict=True))


def make_raw_example(serialized_edges: tuple[tuple[int, int], ...], *, pos_size: int) -> RawPrecomputedSoftTargetExample:
    prefix = [int(DOC_TYPE), int(BEGIN_SEQUENCE)]
    residues = sorted({token for edge in serialized_edges for token in edge})
    for residue in residues:
        prefix.extend([residue, 86])
    prefix.append(int(BEGIN_STRUCTURE))
    prediction_start = len(prefix) - 1

    suffix: list[int] = []
    for first, second in serialized_edges:
        suffix.extend([int(CONTACT), first, second])
    suffix.append(int(END))

    token_ids = np.zeros(pos_size, dtype=np.int32)
    raw_tokens = np.asarray(prefix + suffix, dtype=np.int32)
    token_ids[: raw_tokens.shape[0]] = raw_tokens
    position_ids = np.arange(pos_size, dtype=np.int32)
    segment_ids = np.full(pos_size, -1, dtype=np.int32)
    segment_ids[: raw_tokens.shape[0]] = 0
    attention_blocks = np.zeros(pos_size, dtype=np.int32)
    attention_blocks[prediction_start + 1 : raw_tokens.shape[0]] = np.arange(
        1,
        raw_tokens.shape[0] - prediction_start,
        dtype=np.int32,
    )

    max_contacts = (pos_size - 2) // 3
    first_ids = np.zeros(max_contacts, dtype=np.int32)
    second_ids = np.zeros(max_contacts, dtype=np.int32)
    if serialized_edges:
        first_ids[: len(serialized_edges)] = [first for first, _ in serialized_edges]
        second_ids[: len(serialized_edges)] = [second for _, second in serialized_edges]

    return RawPrecomputedSoftTargetExample(
        token_ids=token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        attention_blocks=attention_blocks,
        first_ids=first_ids,
        second_ids=second_ids,
        contact_count=len(serialized_edges),
        prediction_start=prediction_start,
        target_position_count=prediction_start + 3 * len(serialized_edges) + 1,
    )


def batch_from_raw(raw: RawPrecomputedSoftTargetExample, *, max_contacts: int, max_degree: int) -> SparseContactDocumentBatch:
    first_ids, second_ids, neighbor_ids, neighbor_counts, neighbor_count = _sparse_second_endpoint_targets(
        raw,
        max_contacts=max_contacts,
        max_degree=max_degree,
    )
    Batch = hax.Axis("batch", 1)
    Pos = hax.Axis("position", raw.token_ids.shape[0])
    return SparseContactDocumentBatch(
        tokens=hax.named(jnp.asarray(raw.token_ids[None, :]), (Batch, Pos)),
        contact_first_ids=jnp.asarray(first_ids[None, :]),
        contact_second_ids=jnp.asarray(second_ids[None, :]),
        second_neighbor_ids=jnp.asarray(neighbor_ids[None, :, :]),
        second_neighbor_counts=jnp.asarray(neighbor_counts[None, :, :]),
        second_neighbor_count=jnp.asarray(neighbor_count[None, :]),
        contact_count=jnp.asarray([raw.contact_count], dtype=jnp.int32),
        prediction_start=jnp.asarray([raw.prediction_start], dtype=jnp.int32),
        position_ids=hax.named(jnp.asarray(raw.position_ids[None, :]), (Batch, Pos)),
        segment_ids=hax.named(jnp.asarray(raw.segment_ids[None, :]), (Batch, Pos)),
        attention_blocks=hax.named(jnp.asarray(raw.attention_blocks[None, :]), (Batch, Pos)),
        target_position_count=jnp.asarray([raw.target_position_count], dtype=jnp.int32),
        vocabulary=None,
    )


def dense_suffix_oracle_loss(
    activations: np.ndarray,
    lm_head: np.ndarray,
    serialized_edges: tuple[tuple[int, int], ...],
    *,
    prediction_start: int,
    token_ids: list[int] | np.ndarray | None = None,
) -> float:
    """Dense CE for exp177's prefix-CE plus soft suffix target distribution."""
    logits = activations @ lm_head
    log_probs = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    total = 0.0

    def ce(position: int, distribution: Counter[int]) -> float:
        denom = sum(distribution.values())
        return float(
            -sum(count * float(log_probs[position, token]) for token, count in distribution.items()) / denom
        )

    if token_ids is not None:
        for position in range(prediction_start):
            total += ce(position, Counter({int(token_ids[position + 1]): 1}))

    for contact_index, (observed_first, _observed_second) in enumerate(serialized_edges):
        contact_predict_position = prediction_start if contact_index == 0 else prediction_start + 3 * contact_index
        total += ce(contact_predict_position, Counter({int(CONTACT): 1}))

        remaining = serialized_edges[contact_index:]
        endpoint_distribution: Counter[int] = Counter()
        for first, second in remaining:
            endpoint_distribution[first] += 1
            endpoint_distribution[second] += 1
        total += ce(prediction_start + 1 + 3 * contact_index, endpoint_distribution)

        neighbor_distribution: Counter[int] = Counter()
        for first, second in remaining:
            if first == observed_first:
                neighbor_distribution[second] += 1
            if second == observed_first:
                neighbor_distribution[first] += 1
        assert neighbor_distribution
        total += ce(prediction_start + 2 + 3 * contact_index, neighbor_distribution)

    total += ce(prediction_start + 3 * len(serialized_edges), Counter({int(END): 1}))
    return total / (prediction_start + 3 * len(serialized_edges) + 1)


def test_sparse_contact_loss_matches_dense_serialization_oracle_for_all_orders_and_orientations():
    """Production sparse loss equals the independent exact suffix oracle."""
    rng = np.random.default_rng(279)
    pos_size = 32
    embed_size = 11
    vocab_size = 2845
    max_contacts = 8
    max_degree = 8
    activations = rng.normal(size=(1, pos_size, embed_size)).astype(np.float32)
    lm_head = rng.normal(size=(embed_size, vocab_size)).astype(np.float32) / np.sqrt(embed_size)
    model = FixedActivationModel(
        jnp.asarray(activations),
        jnp.asarray(lm_head),
        hax.Axis("embed", embed_size),
        hax.Axis("vocab", vocab_size),
    )

    # Includes repeated endpoint tokens and nontrivial degrees, the case most
    # likely to reveal set-vs-multiset and consumed-edge bugs.
    undirected_edges = ((143, 144), (143, 145), (144, 145))
    for serialized_edges in serialized_documents(undirected_edges):
        raw = make_raw_example(serialized_edges, pos_size=pos_size)
        batch = batch_from_raw(raw, max_contacts=max_contacts, max_degree=max_degree)
        actual = float(sparse_contact_document_loss(model, batch))
        expected = dense_suffix_oracle_loss(
            activations[0],
            lm_head,
            serialized_edges,
            prediction_start=raw.prediction_start,
            token_ids=raw.token_ids,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_sparse_second_endpoint_targets_include_current_edge_then_consume_it():
    """Neighbor rows should represent remaining edges before consuming current edge."""
    raw = make_raw_example(((143, 144), (143, 145), (145, 143)), pos_size=32)
    _first, _second, neighbor_ids, neighbor_counts, neighbor_count = _sparse_second_endpoint_targets(
        raw,
        max_contacts=8,
        max_degree=8,
    )

    def row_counts(index: int) -> Counter[int]:
        return Counter(
            {
                int(token_id): int(count)
                for token_id, count in zip(neighbor_ids[index], neighbor_counts[index], strict=True)
                if count
            }
        )

    assert neighbor_count[0] == 3
    assert row_counts(0) == Counter({144: 1, 145: 2})
    assert neighbor_count[1] == 2
    assert row_counts(1) == Counter({145: 2})
    assert neighbor_count[2] == 1
    assert row_counts(2) == Counter({143: 1})
