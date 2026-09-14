# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Probability oracle enumerates serializations, independent of degree formulas."""

import re
from collections import Counter, defaultdict
from itertools import permutations, product

import jax
import numpy as np
import pytest
from conftest import make_document
from marinfold.document_structures.contacts_v1.generate import build_document
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    residues_from_sequence,
)

from experiments.exp279_models_exact_soft_contact_targets.targets import (
    DocumentSlice,
    distributions,
    document_edges,
    pack_targets,
)


def serialization_oracle(edges):
    continuations = defaultdict(Counter)
    serializations = []
    for order in permutations(edges):
        for flips in product((False, True), repeat=len(edges)):
            oriented = [
                edge[::-1] if flip else edge
                for edge, flip in zip(order, flips, strict=True)
            ]
            tokens = tuple(x for edge in oriented for x in (5, *edge)) + (10,)
            for i, token in enumerate(tokens):
                continuations[tokens[:i]][token] += 1
            serializations.append(oriented)
    return continuations, serializations


q_kernel = jax.jit(distributions, static_argnums=(7,))


def target_q(targets, tokens):
    # jnp.roll is required inside jit, while the input oracle remains NumPy.
    return np.asarray(
        q_kernel(
            targets.first.array,
            targets.second.array,
            targets.start.array,
            targets.stop.array,
            targets.kind.array,
            tokens,
            np.roll(tokens, -1),
            2845,
        )
    )


@pytest.mark.parametrize(
    "edges",
    [
        [],
        [(143, 144)],
        [(143, 144), (143, 145)],
        [(143, 144), (145, 146)],
        [(143, 144), (144, 145), (143, 145)],
        [(143, 144), (143, 145), (143, 146), (143, 147)],
    ],
)
def test_every_permutation_orientation_and_prefix(vocab, edges):
    oracle, serializations = serialization_oracle(edges)
    for oriented in serializations:
        doc = make_document(oriented)
        padded = np.pad(doc, (0, 64 - len(doc)))
        targets = pack_targets([DocumentSlice(doc)], padded, vocab, edge_capacity=8)
        q = target_q(targets, padded)
        np.testing.assert_allclose(q.sum(-1), 1, atol=1e-7)
        structure = int(np.flatnonzero(doc == 9)[0])
        emitted = tuple(doc[structure + 1 : -1])
        for i, token in enumerate(emitted):
            counts = oracle[emitted[:i]]
            expected = np.zeros(vocab.size)
            for value, count in counts.items():
                expected[value] = count / sum(counts.values())
            np.testing.assert_allclose(q[structure + i], expected, atol=1e-7)
        np.testing.assert_array_equal(q[:structure].argmax(-1), doc[1 : structure + 1])


def test_fragment_keeps_unseen_future_edges_and_consumes_past(vocab):
    doc = make_document([(143, 144), (143, 145), (146, 143)])
    markers = np.flatnonzero(doc == 5)
    full = pack_targets([DocumentSlice(doc)], doc, vocab, edge_capacity=8)
    expected = target_q(full, doc)
    # Include left/right fragments and cuts inside both endpoints.
    for start in range(len(doc) - 1):
        for stop in range(start + 1, len(doc) + 1):
            fragment = np.pad(doc[start:stop], (0, 64 - (stop - start)))
            target = pack_targets(
                [DocumentSlice(doc, start, stop)], fragment, vocab, edge_capacity=8
            )
            q = target_q(target, fragment)
            np.testing.assert_allclose(
                q[: stop - start - 1], expected[start : stop - 1], atol=1e-7
            )
    # The first endpoint still includes residue 146, unseen beyond this fragment.
    left = doc[: markers[0] + 2]
    q = target_q(
        pack_targets([DocumentSlice(doc, 0, len(left))], left, vocab, edge_capacity=8),
        left,
    )
    assert q[markers[0], 146] == pytest.approx(1 / 6)


def test_packed_documents_with_shared_numbering_are_isolated(vocab):
    a, b = make_document([(143, 144)]), make_document([(143, 145), (143, 146)])
    packed = np.pad(np.concatenate([a, b]), (0, 5))
    target = pack_targets(
        [DocumentSlice(a), DocumentSlice(b)], packed, vocab, edge_capacity=8
    )
    q = target_q(target, packed)
    for offset, doc in ((0, a), (len(a), b)):
        individual = target_q(
            pack_targets([DocumentSlice(doc)], doc, vocab, edge_capacity=8), doc
        )
        np.testing.assert_allclose(q[offset : offset + len(doc) - 1], individual[:-1])


@pytest.mark.parametrize(
    "edges, message", [([(143, 143)], "Self"), ([(143, 144), (144, 143)], "Duplicate")]
)
def test_invalid_graphs_fail(vocab, edges, message):
    with pytest.raises(ValueError, match=message):
        document_edges(make_document(edges), vocab)


def test_corruption_and_capacity_fail_loudly(vocab):
    doc = make_document([(143, 144)])
    for malformed in (doc[:-2], doc[1:], np.append(doc, 2844), doc.astype(float)):
        with pytest.raises(ValueError):
            document_edges(malformed, vocab)
    with pytest.raises(ValueError, match="capacity"):
        pack_targets([DocumentSlice(doc)], doc, vocab, edge_capacity=0)
    with pytest.raises(ValueError, match="reproduce"):
        pack_targets([DocumentSlice(doc)], np.roll(doc, 1), vocab, edge_capacity=1)
    wrong = doc.copy()
    wrong[-3] = 155
    with pytest.raises(ValueError, match="absent"):
        document_edges(wrong, vocab)


def test_real_generator_filters_truncates_and_randomizes_numbering(vocab):
    contacts = [
        RawContact(0, 7, 0.5),
        RawContact(0, 8, 0.1),
        RawContact(1, 9, 0.0001),
        RawContact(1, 2, 1.0),
        RawContact(3, 12, 0.7),
        RawContact(4, 15, 0.6),
    ]
    mapping = {
        "<contacts-v1>": 2,
        "<begin_sequence>": 8,
        "<begin_statements>": 9,
        "<contact>": 5,
        "<end>": 10,
        "<n-term>": 3,
        "<c-term>": 4,
        "<ALA>": 86,
    }
    mapping.update({f"<p{i}>": 143 + i for i in range(2000)})
    numberings = set()
    for seed in range(10):
        result = build_document(
            str(seed), residues_from_sequence("A" * 16), contacts, context_length=46
        )
        assert result is not None
        assert sorted(c.degree for c in result.contacts) == [0.6, 0.7]
        tokens = np.asarray(
            [mapping[t] for t in re.findall(r"<[^>]+>", result.document)] + [1],
            np.int32,
        )
        edges, _ = document_edges(tokens, vocab)
        assert len(edges) == 2
        numberings.add(tuple(edges.reshape(-1)))
        target = pack_targets([DocumentSlice(tokens)], tokens, vocab, edge_capacity=8)
        q = target_q(target, tokens)
        first_marker = int(np.flatnonzero(tokens == 5)[0])
        # The selected contacts are disjoint, despite different contact degrees.
        np.testing.assert_allclose(q[first_marker, edges.reshape(-1)], 0.25)
    assert len(numberings) > 1
