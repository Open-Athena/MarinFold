# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact conditional next-token targets of contacts-v1 (#201).

The load-bearing test here is :func:`test_hard_loss_equals_soft_loss_plus_entropy`
-- the Monte-Carlo identity that certifies the targets really are the conditional
marginals of the generator. Every later phase (the JAX loss, the training A/B)
rests on it: a silently wrong target looks exactly like a working run.
"""

import math
import random
from collections import Counter

import pytest

from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_v1.vocab import NUM_POSITION_INDICES
from marinfold.document_structures.contacts_v1.soft_targets import (
    FIRST_ENDPOINT,
    SECOND_ENDPOINT,
    STATEMENT_HEAD,
    hard_cross_entropy,
    parse_document,
    permutation_entropy,
    soft_cross_entropy,
    soft_targets,
)


_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]

# min_seq_separation=1 keeps every synthetic pair (seq_i < seq_j always), so the
# small hand-built contact sets below survive intact.
_CFG = GenerationConfig(min_seq_separation=1)


def _residues(n: int) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=1 + i, chain="A")
        for i in range(n)
    ]


def _contacts(pairs: list[tuple[int, int]]) -> list[RawContact]:
    # Descending degree so the selection order is deterministic and every pair
    # clears the min-degree filter.
    return [RawContact(i, j, 0.9 - 0.001 * k) for k, (i, j) in enumerate(pairs)]


_PAIRS = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 5), (3, 6), (0, 7), (2, 7), (4, 7)]


def _document(entry_id: str = "e0", *, n_res: int = 9) -> list[str]:
    res = build_document(entry_id, _residues(n_res), _contacts(_PAIRS), config=_CFG)
    return res.document.split()


# ---------------------------------------------------------------------------
# Shape and coverage
# ---------------------------------------------------------------------------


def test_targets_cover_every_predicted_token_exactly_once():
    tokens = _document()
    targets = soft_targets(tokens)
    assert [t.target_index for t in targets] == list(range(1, len(tokens)))


def test_realised_token_is_always_in_the_support():
    tokens = _document()
    for target in soft_targets(tokens):
        assert target.probability_of(tokens[target.target_index]) > 0.0


def test_probs_are_normalised():
    for target in soft_targets(_document()):
        assert math.isclose(sum(target.probs), 1.0, rel_tol=1e-12)


def test_amino_acid_and_frame_slots_stay_one_hot():
    tokens = _document()
    for target in soft_targets(tokens):
        if not target.is_soft:
            assert target.support == (tokens[target.target_index],)
            assert target.probs == (1.0,)


def test_parse_rejects_think_documents():
    res = build_document(
        "think", _residues(9), _contacts(_PAIRS),
        config=GenerationConfig(min_seq_separation=1, think=True),
    )
    tokens = res.document.split()
    if "<think>" not in tokens:  # the initial-run gate is probabilistic
        pytest.skip("this entry_id drew no think tokens")
    with pytest.raises(ValueError, match="think"):
        parse_document(tokens)


# ---------------------------------------------------------------------------
# The distributions themselves
# ---------------------------------------------------------------------------


def test_statement_head_target_is_uniform_over_remaining_statements():
    tokens = _document()
    parsed = parse_document(tokens)
    n_statements = len(parsed.statements)
    heads = [t for t in soft_targets(tokens) if t.kind == STATEMENT_HEAD]
    assert len(heads) == n_statements
    for k, target in enumerate(heads):
        assert len(target.support) == n_statements - k
        assert set(target.support) == {h for h, _ in parsed.statements[k:]}


def test_first_endpoint_target_is_degree_over_twice_remaining():
    tokens = _document()
    parsed = parse_document(tokens)
    firsts = [t for t in soft_targets(tokens) if t.kind == FIRST_ENDPOINT]
    for k, target in enumerate(firsts):
        remaining = parsed.contacts[k:]
        degree: dict[str, int] = {}
        for a, b in remaining:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        expected = {p: d / (2 * len(remaining)) for p, d in degree.items()}
        assert dict(zip(target.support, target.probs)) == pytest.approx(expected)


def test_second_endpoint_target_is_uniform_over_remaining_partners():
    tokens = _document()
    parsed = parse_document(tokens)
    seconds = [t for t in soft_targets(tokens) if t.kind == SECOND_ENDPOINT]
    for k, target in enumerate(seconds):
        first = parsed.contacts[k][0]
        remaining = parsed.contacts[k:]
        partners = {b if a == first else a for a, b in remaining if first in (a, b)}
        assert set(target.support) == partners
        assert target.probs == pytest.approx((1.0 / len(partners),) * len(partners))


def test_second_endpoint_support_includes_the_realised_partner():
    # The contact being emitted is itself in R_k, so its far end must be in the
    # support -- an off-by-one in the peeling would silently break this.
    tokens = _document()
    parsed = parse_document(tokens)
    seconds = [t for t in soft_targets(tokens) if t.kind == SECOND_ENDPOINT]
    for (_, second), target in zip(parsed.contacts, seconds):
        assert second in target.support


def test_sequence_only_documents_have_no_structure_targets():
    res = build_document(
        "seq", _residues(9), _contacts(_PAIRS),
        config=GenerationConfig(min_seq_separation=1, sequence_only=True),
    )
    tokens = res.document.split()
    kinds = {t.kind for t in soft_targets(tokens)}
    assert FIRST_ENDPOINT not in kinds and SECOND_ENDPOINT not in kinds
    assert STATEMENT_HEAD in kinds


# ---------------------------------------------------------------------------
# The Monte-Carlo identities (the ones that certify correctness)
# ---------------------------------------------------------------------------


def _random_log_probs(tokens: list[str], seed: int) -> list[dict[str, float]]:
    """A fixed, arbitrary 'model': one normalised distribution per slot.

    It is keyed on the *prefix* (via the slot index and the preceding token) so
    that the same prefix seen under different orderings gets the same
    distribution -- which is what makes the identity below meaningful.
    """
    vocabulary = sorted(set(tokens))
    rng = random.Random(seed)
    weights = {tok: rng.random() + 0.05 for tok in vocabulary}
    total = sum(weights.values())
    logs = {tok: math.log(w / total) for tok, w in weights.items()}
    return [dict(logs) for _ in range(len(tokens) - 1)]


def test_soft_loss_decomposes_into_kl_plus_entropy():
    # Pointwise: soft-CE = KL(q||p) + H(q). Note this makes H(q) the FLOOR of the
    # soft loss, not something it subtracts off -- the interpretable
    # zero-at-optimum quantity is KL, i.e. the loss minus the (computable) H(q).
    tokens = _document()
    log_probs = _random_log_probs(tokens, seed=0)
    targets = soft_targets(tokens)
    kl = 0.0
    for target, lp in zip(targets, log_probs):
        for token, p in zip(target.support, target.probs):
            kl += p * (math.log(p) - lp[token])
    entropy = sum(t.entropy for t in targets)
    assert soft_cross_entropy(tokens, log_probs) == pytest.approx(
        (kl + entropy) / len(targets)
    )


def test_expected_hard_loss_equals_expected_soft_loss():
    """E_ordering[hard CE] == E_ordering[soft CE] -- the identity #201 rests on.

    The soft target is the conditional mean of the one-hot target, so the two
    losses share an expectation (and therefore an expected gradient, and a
    floor of H(q)). The soft one is the lower-variance estimator of it; that,
    not a smaller loss number, is the whole benefit.
    """
    residues = _residues(9)
    contacts = _contacts(_PAIRS)
    hard_total = 0.0
    soft_total = 0.0
    n_draws = 400
    for draw in range(n_draws):
        # Each entry_id seeds a different ordering of the same contact set.
        tokens = build_document(
            f"draw-{draw}", residues, contacts, config=_CFG
        ).document.split()
        log_probs = _random_log_probs(tokens, seed=0)
        hard_total += hard_cross_entropy(tokens, log_probs)
        soft_total += soft_cross_entropy(tokens, log_probs)
    assert hard_total / n_draws == pytest.approx(soft_total / n_draws, abs=5e-3)


def test_realised_tokens_follow_the_soft_target_distribution():
    """The decisive correctness test: sample orderings, histogram what actually
    gets emitted at a fixed slot, and check it converges to the claimed target.

    A wrong normaliser or an off-by-one in the "remaining contacts" bookkeeping
    survives every structural test above but fails here.

    Each draw also re-rolls the random wrap-around start index, so tokens are
    compared in the document's own **sequence-index** frame (``(pos - start) %
    NUM_POSITION_INDICES``), which is what is actually shared across draws.
    """
    residues = _residues(9)
    contacts = _contacts(_PAIRS)
    n_draws = 6000
    first_head: Counter = Counter()
    first_endpoint: Counter = Counter()
    expected_head: dict[str, float] = {}
    expected_endpoint: dict[str, float] = {}

    for draw in range(n_draws):
        result = build_document(f"mc-{draw}", residues, contacts, config=_CFG)
        tokens = result.document.split()

        def canonical(token: str, start: int = result.start_index) -> str:
            if not token.startswith("<p"):
                return token
            return f"seq{(int(token[2:-1]) - start) % NUM_POSITION_INDICES}"

        targets = soft_targets(tokens)
        # Slot 0 of each section: in the sequence-index frame the prefix is the
        # same across draws, so the realised tokens are i.i.d. draws from that
        # slot's target.
        head = next(t for t in targets if t.kind == STATEMENT_HEAD)
        endpoint = next(t for t in targets if t.kind == FIRST_ENDPOINT)
        first_head[canonical(tokens[head.target_index])] += 1
        first_endpoint[canonical(tokens[endpoint.target_index])] += 1
        if not expected_head:
            expected_head = {canonical(t): p for t, p in zip(head.support, head.probs)}
            expected_endpoint = {
                canonical(t): p for t, p in zip(endpoint.support, endpoint.probs)
            }

    for observed, expected in ((first_head, expected_head),
                               (first_endpoint, expected_endpoint)):
        assert set(observed) <= set(expected)
        for token, want in expected.items():
            got = observed[token] / n_draws
            # 4 sigma of a binomial proportion, plus slack for the small counts.
            tol = 4.0 * math.sqrt(want * (1.0 - want) / n_draws) + 1e-3
            assert abs(got - want) < tol, f"{token}: got {got:.4f} want {want:.4f}"


def test_expected_permutation_entropy_matches_closed_form():
    """Mean realised entropy over orderings == log((L+2)!) + log(N!) + N log 2."""
    residues = _residues(9)
    contacts = _contacts(_PAIRS)
    n_draws = 400
    total = 0.0
    breakdown = None
    for draw in range(n_draws):
        tokens = build_document(
            f"draw-{draw}", residues, contacts, config=_CFG
        ).document.split()
        breakdown = permutation_entropy(tokens)
        total += breakdown.total_nats
    assert breakdown is not None
    assert total / n_draws == pytest.approx(breakdown.expected_total_nats, rel=2e-3)


def test_sequence_entropy_is_exactly_log_factorial():
    # The statement-head terms are uniform over a shrinking support, so their
    # sum is log((L+2)!) exactly -- no Monte Carlo needed.
    tokens = _document()
    breakdown = permutation_entropy(tokens)
    assert breakdown.sequence_nats == pytest.approx(
        math.lgamma(breakdown.seq_len + 2 + 1)
    )
