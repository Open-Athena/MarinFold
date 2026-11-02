# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-chain (multimer) contacts-v1 documents.

The format extension is described in ``SPEC.md`` under *Residue indexing*:
several protein chains share one 2000-index wrap-around ring, laid out
disjointly with a gap between them, each contributing its own ``<n-term>``
and ``<c-term>`` statement. These tests pin the three properties that make
such a document readable -- chains occupy contiguous, non-overlapping index
runs; every chain is announced by exactly one terminus pair; and the
sequence-separation filter is intra-chain only -- plus the invariant that
matters most to every corpus built before this existed: a single-chain
document is byte-for-byte what it always was.
"""

import random
import re

import pytest

from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    ResidueInfo,
    chain_segments,
)


_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]

# min_seq_separation=1 keeps every synthetic pair (seq_i < seq_j always), so
# tests that are about layout rather than the separation filter aren't
# silently emptied by it. The filter has its own tests below.
_CFG = GenerationConfig(min_seq_separation=1)


def _residues(lengths: list[int]) -> list[ResidueInfo]:
    """Residues for chains of the given lengths, named A, B, C, ... in order."""
    out: list[ResidueInfo] = []
    for chain_index, length in enumerate(lengths):
        chain = chr(ord("A") + chain_index)
        for k in range(length):
            out.append(
                ResidueInfo(
                    seq_index=len(out),
                    resname=_AA_CYCLE[len(out) % len(_AA_CYCLE)],
                    resnum=k + 1,
                    chain=chain,
                )
            )
    return out


def _sequence_section(document: str) -> list[str]:
    tokens = document.split()
    start = tokens.index(vocab.BEGIN_SEQUENCE_TOKEN) + 1
    stop = tokens.index(vocab.BEGIN_STRUCTURE_TOKEN)
    return tokens[start:stop]


def _termini(document: str) -> tuple[list[int], list[int]]:
    """(n-terminal indices, c-terminal indices) read back out of a document."""
    tokens = _sequence_section(document)
    n_term: list[int] = []
    c_term: list[int] = []
    for i, token in enumerate(tokens):
        if token == vocab.N_TERM_TOKEN:
            n_term.append(int(re.fullmatch(r"<p(\d+)>", tokens[i + 1]).group(1)))
        elif token == vocab.C_TERM_TOKEN:
            c_term.append(int(re.fullmatch(r"<p(\d+)>", tokens[i + 1]).group(1)))
    return n_term, c_term


def test_chain_segments_splits_contiguous_runs():
    segments = chain_segments(_residues([3, 5, 2]))
    assert [(s.chain, s.start, s.stop) for s in segments] == [
        ("A", 0, 3), ("B", 3, 8), ("C", 8, 10),
    ]
    assert [s.length for s in segments] == [3, 5, 2]


def test_chain_segments_rejects_interleaved_chains():
    """A chain split across two runs is a bug upstream, not something to paper over."""
    residues = [
        ResidueInfo(seq_index=0, resname="ALA", resnum=1, chain="A"),
        ResidueInfo(seq_index=1, resname="GLY", resnum=1, chain="B"),
        ResidueInfo(seq_index=2, resname="SER", resnum=2, chain="A"),
    ]
    with pytest.raises(ValueError, match="more than one run"):
        chain_segments(residues)


def test_one_terminus_pair_per_chain():
    lengths = [40, 25, 60]
    result = build_document("multimer-1", _residues(lengths), [], config=_CFG)
    assert result is not None
    assert result.num_chains == 3
    assert result.chain_ids == ("A", "B", "C")
    assert result.chain_lengths == tuple(lengths)

    n_term, c_term = _termini(result.document)
    assert len(n_term) == 3
    assert len(c_term) == 3
    assert sorted(n_term) == sorted(result.n_term_indices)
    assert sorted(c_term) == sorted(result.c_term_indices)


def test_chains_occupy_disjoint_contiguous_index_runs():
    """Each chain gets one unbroken run of indices, and the runs don't overlap."""
    lengths = [40, 25, 60, 11]
    residues = _residues(lengths)
    result = build_document("multimer-2", residues, [], config=_CFG)
    assert result is not None

    # Rebuild each chain's index run from its announced n-terminus: a chain of
    # length L starting at n runs n, n+1, ... (mod 2000) for L positions.
    runs: list[set[int]] = []
    for length, n_term in zip(result.chain_lengths, result.n_term_indices):
        runs.append({(n_term + k) % vocab.NUM_POSITION_INDICES for k in range(length)})

    # Disjoint.
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            assert not runs[i] & runs[j], f"chains {i} and {j} overlap"

    # Exactly the residues the document assigns, and the c-terminus is the run's end.
    assigned = [
        int(m.group(1))
        for m in re.finditer(r"<p(\d+)> <(?:[A-Z]{3})>", result.document)
    ]
    assert sorted(assigned) == sorted(x for run in runs for x in run)
    for length, n_term, c_term in zip(
        result.chain_lengths, result.n_term_indices, result.c_term_indices
    ):
        assert c_term == (n_term + length - 1) % vocab.NUM_POSITION_INDICES


def test_minimum_gap_separates_chains():
    """No chain's n-terminus is adjacent to another chain's c-terminus."""
    lengths = [300, 300, 300, 300, 300, 300]  # 1800 residues, only 200 slack
    result = build_document("multimer-tight", _residues(lengths), [], config=_CFG)
    assert result is not None
    occupied: set[int] = set()
    for length, n_term in zip(result.chain_lengths, result.n_term_indices):
        occupied |= {(n_term + k) % vocab.NUM_POSITION_INDICES for k in range(length)}
    assert len(occupied) == sum(lengths)
    for n_term in result.n_term_indices:
        before = (n_term - 1) % vocab.NUM_POSITION_INDICES
        assert before not in occupied, "chains must be separated by at least one index"


def test_chains_that_do_not_fit_are_rejected():
    """Residues plus the mandatory gaps must fit the ring, not just the residues."""
    # 2000 residues exactly fills the ring, leaving no room for the two gaps.
    assert build_document("too-big", _residues([1000, 1000]), [], config=_CFG) is None
    # One residue less per chain and it fits.
    assert build_document("just-fits", _residues([999, 999]), [], config=_CFG) is not None


def test_sequence_separation_filter_is_intra_chain_only():
    """Interface contacts survive at any separation; intra-chain ones don't."""
    residues = _residues([20, 20])
    contacts = [
        RawContact(seq_i=0, seq_j=2, degree=0.9),    # chain A, separation 2
        RawContact(seq_i=0, seq_j=15, degree=0.8),   # chain A, separation 15
        RawContact(seq_i=19, seq_j=20, degree=0.7),  # A/B interface, adjacent indices
        RawContact(seq_i=5, seq_j=25, degree=0.6),   # A/B interface
    ]
    config = GenerationConfig(min_seq_separation=6)
    result = build_document("multimer-sep", residues, contacts, config=config)
    assert result is not None

    emitted = {(c.seq_i, c.seq_j) for c in result.contacts}
    # The separation-2 intra-chain pair is gone; everything else survives,
    # including the interface pair whose sequence indices are adjacent.
    assert emitted == {(0, 15), (19, 20), (5, 25)}
    assert result.contacts_emitted_inter_chain == 2
    assert result.contacts_pre_filter_inter_chain == 2


def test_interface_contacts_are_flagged():
    residues = _residues([10, 10])
    contacts = [
        RawContact(seq_i=0, seq_j=9, degree=0.9),
        RawContact(seq_i=3, seq_j=13, degree=0.8),
    ]
    result = build_document("multimer-flag", residues, contacts, config=_CFG)
    assert result is not None
    by_pair = {(c.seq_i, c.seq_j): c for c in result.contacts}
    assert by_pair[(0, 9)].inter_chain is False
    assert by_pair[(0, 9)].chain_i == "A" and by_pair[(0, 9)].chain_j == "A"
    assert by_pair[(3, 13)].inter_chain is True
    assert by_pair[(3, 13)].chain_i == "A" and by_pair[(3, 13)].chain_j == "B"


def test_token_budget_accounts_for_every_chain():
    """Each extra chain costs one more <n-term>/<c-term> pair (4 tokens)."""
    one = build_document("budget-1", _residues([100]), [], config=_CFG)
    three = build_document("budget-3", _residues([40, 30, 30]), [], config=_CFG)
    assert one is not None and three is not None
    # Same residue count, two extra chains => two extra terminus pairs.
    assert three.num_tokens - one.num_tokens == 2 * 4
    assert three.num_tokens == len(three.document.split())


def test_document_never_exceeds_context_length():
    rng = random.Random(0)
    for trial in range(40):
        k = rng.randint(2, 6)
        lengths = [rng.randint(2, 300) for _ in range(k)]
        residues = _residues(lengths)
        total = sum(lengths)
        contacts = []
        seen = set()
        for _ in range(4000):
            i, j = rng.randrange(total), rng.randrange(total)
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            contacts.append(RawContact(seq_i=i, seq_j=j, degree=rng.random()))
        result = build_document(f"budget-{trial}", residues, contacts, config=_CFG)
        assert result is not None
        assert result.num_tokens <= vocab.CONTEXT_LENGTH


def test_single_chain_layout_is_unchanged():
    """The k=1 path must draw the same RNG stream the old generator did.

    Byte-identity for monomers is what lets every pre-existing corpus and
    checkpoint carry over. The old code drew exactly one number to pick the
    n-terminal index; the generic ring layout must too, which means skipping
    the chain-order shuffle and the gap composition when there is one chain.
    """
    residues = _residues([137])
    contacts = [RawContact(seq_i=i, seq_j=i + 20, degree=0.5 + i / 1000)
                for i in range(0, 100)]
    result = build_document("mono", residues, contacts, config=_CFG)
    assert result is not None
    assert result.num_chains == 1

    rng = random.Random(int(__import__("hashlib").sha1(b"mono").hexdigest()[:8], 16))
    expected_start = rng.randrange(vocab.NUM_POSITION_INDICES)
    assert result.start_index == expected_start
    assert result.n_term_index == expected_start
    assert result.c_term_index == (expected_start + 136) % vocab.NUM_POSITION_INDICES


def test_multimer_documents_are_deterministic():
    lengths = [55, 90, 33]
    first = build_document("multimer-det", _residues(lengths), [], config=_CFG)
    second = build_document("multimer-det", _residues(lengths), [], config=_CFG)
    assert first is not None and second is not None
    assert first.document == second.document
    # A different entry id reshuffles the ring.
    other = build_document("multimer-det-2", _residues(lengths), [], config=_CFG)
    assert other is not None
    assert other.document != first.document


def test_chain_ring_order_is_randomized():
    """The ring order is not the structure order -- otherwise chain identity
    would be recoverable from index order alone, which is a shortcut we do
    not want the model to learn."""
    lengths = [30, 30, 30, 30]
    orders = set()
    for seed in range(40):
        result = build_document(f"ring-{seed}", _residues(lengths), [], config=_CFG)
        assert result is not None
        # Rank the structure-order chains by their n-terminal index.
        ranking = tuple(
            sorted(range(4), key=lambda i: result.n_term_indices[i])
        )
        orders.add(ranking)
    assert len(orders) > 1
