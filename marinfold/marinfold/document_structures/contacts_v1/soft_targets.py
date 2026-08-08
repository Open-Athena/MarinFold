# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact conditional next-token distributions for contacts-v1 documents.

A contacts-v1 document serialises two unordered **sets** — the sequence
statements and the selected contacts — in a *uniformly random order*
(``generate.build_document`` shuffles both). One-hot next-token supervision
therefore asks the model to predict which permutation was sampled, which is
pure nuisance: it carries no information about the structure and the model
cannot ever predict it.

This module computes, for every slot in a document, the **exact conditional
distribution of the next token** implied by that generation process, given the
document's own content. Training against it instead of the one-hot target is a
Rao-Blackwellization: the population objective is unchanged (the one-hot target
is a sample from exactly this distribution, so the expected gradient is
identical) while the target variance can only fall. See
`#201 <https://github.com/Open-Athena/MarinFold/issues/201>`_.

The distributions are a **pure function of the token stream** — the document is
the contact list in emission order — so no side-channel, corpus change or
tokenizer change is needed to derive them.

This is the plain-Python reference implementation: readable, dependency-free
(stdlib only), and the oracle the JAX training-time implementation in
``marinfold_models`` is tested against. It is deliberately not fast.

The four kinds of slot
----------------------

Writing the structure section as ``<contact> X_k Y_k`` for ``k = 0..N-1``, with
``R_k`` the contacts not yet emitted when slot ``k`` begins (so
``|R_k| = N - k``) and ``deg_R(p)`` the number of contacts in ``R`` incident to
position ``p``:

``FRAME`` / ``STATEMENT_BODY``
    One-hot. Section markers, the amino acid of a residue statement, the index
    of a terminus statement, ``<contact>`` vs ``<end>``. These carry real
    information and stay hard targets.

``STATEMENT_HEAD``
    Uniform over the heads of the statements not yet emitted (``<pX>`` for an
    undefined residue, plus ``<n-term>`` / ``<c-term>`` if not yet emitted). All
    heads are distinct, so the distribution is uniform over its support.

``FIRST_ENDPOINT``
    ``q(p) = deg_R_k(p) / (2 * |R_k|)``. The generator draws the next contact
    uniformly from ``R_k`` then flips a fair coin for endpoint order, so each
    remaining contact contributes 1/2 to each of its two endpoints — hence the
    ``2 * |R_k|`` normaliser.

``SECOND_ENDPOINT``
    Uniform over ``X_k``'s **remaining** partners. Conditioned on "first
    endpoint = ``X``", the posterior over which contact was drawn is uniform
    over the remaining contacts incident to ``X`` (each contributes the same 1/2
    coin factor), so the second endpoint is uniform over their far ends.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    C_TERM_TOKEN,
    DOC_TYPE_TOKEN,
    END_TOKEN,
    N_TERM_TOKEN,
    SEQUENCE_ONLY_DOC_TYPE_TOKEN,
    THINK_TOKEN,
)

# Slot kinds. FRAME and STATEMENT_BODY are one-hot; the other three are the
# genuinely soft ones (see the module docstring).
FRAME = "frame"
STATEMENT_HEAD = "statement_head"
STATEMENT_BODY = "statement_body"
FIRST_ENDPOINT = "first_endpoint"
SECOND_ENDPOINT = "second_endpoint"

#: Slot kinds whose target is a non-degenerate distribution over the permutation.
SOFT_KINDS = frozenset({STATEMENT_HEAD, FIRST_ENDPOINT, SECOND_ENDPOINT})

_STATEMENT_HEADS_NON_POSITION = frozenset({N_TERM_TOKEN, C_TERM_TOKEN})


@dataclass(frozen=True)
class SoftTarget:
    """The exact conditional distribution of one predicted token.

    Attributes:
        target_index: Index into the document's token list of the token being
            predicted. It is predicted from the prefix ending at
            ``target_index - 1``.
        support: Tokens with non-zero probability, in a deterministic (but not
            necessarily sorted) order.
        probs: Probabilities matching ``support``; sums to 1.
        kind: One of the module-level slot-kind constants.
    """

    target_index: int
    support: tuple[str, ...]
    probs: tuple[float, ...]
    kind: str

    @property
    def entropy(self) -> float:
        """Shannon entropy of the target, in nats."""
        return -sum(p * math.log(p) for p in self.probs if p > 0.0)

    @property
    def is_soft(self) -> bool:
        """Whether this slot's target is a permutation distribution."""
        return self.kind in SOFT_KINDS

    def probability_of(self, token: str) -> float:
        """Probability the target assigns to ``token`` (0 if outside the support)."""
        for candidate, p in zip(self.support, self.probs):
            if candidate == token:
                return p
        return 0.0


@dataclass(frozen=True)
class EntropyBreakdown:
    """Per-document accounting of how much of the loss is nuisance permutation.

    ``*_nats`` fields are sums of per-slot target entropies over the document,
    i.e. the cross-entropy an oracle that knows the structure exactly would
    still pay under one-hot supervision.

    Attributes:
        sequence_nats: Contribution of the sequence-section statement order.
        structure_nats: Contribution of the contact order and endpoint flips.
        num_tokens: Length of the document in tokens.
        num_predicted: Number of supervised next-token slots (``num_tokens - 1``).
        seq_len: Number of residues.
        num_contacts: Number of emitted contacts.
    """

    sequence_nats: float
    structure_nats: float
    num_tokens: int
    num_predicted: int
    seq_len: int
    num_contacts: int

    @property
    def total_nats(self) -> float:
        """Total nuisance entropy of the document, in nats."""
        return self.sequence_nats + self.structure_nats

    @property
    def nats_per_token(self) -> float:
        """Nuisance entropy per supervised token — comparable with a reported loss."""
        return self.total_nats / self.num_predicted

    @property
    def expected_total_nats(self) -> float:
        """Expectation of :attr:`total_nats` over orderings, in closed form.

        ``log((L+2)!) + log(N!) + N*log 2``. The realised
        :attr:`total_nats` is one sample of the chain-rule decomposition of this
        quantity — the second-endpoint terms are evaluated at the realised first
        endpoint rather than averaged — so the two agree only in expectation.
        """
        return (
            math.lgamma(self.seq_len + 2 + 1)
            + math.lgamma(self.num_contacts + 1)
            + self.num_contacts * math.log(2.0)
        )


@dataclass(frozen=True)
class ParsedDocument:
    """Structural decomposition of a contacts-v1 token stream.

    Attributes:
        doc_type: The leading doc-type token.
        statements: Sequence-section statements as ``(head, body)`` pairs, in
            emission order.
        contacts: Structure-section contacts as ``(first, second)`` position
            tokens, in emission order (already endpoint-flipped).
        statements_start: Token index of the first statement head.
        structure_start: Token index of the first ``<contact>``, or ``None`` for
            a sequence-only document (which has no structure section).
        end_index: Token index of the trailing ``<end>``.
    """

    doc_type: str
    statements: tuple[tuple[str, str], ...]
    contacts: tuple[tuple[str, str], ...]
    statements_start: int
    structure_start: int | None
    end_index: int


def parse_document(tokens: Sequence[str]) -> ParsedDocument:
    """Decompose a contacts-v1 token stream into statements and contacts.

    Args:
        tokens: The document's tokens, e.g. ``result.document.split()``.

    Returns:
        The structural decomposition.

    Raises:
        ValueError: If the stream is not a well-formed contacts-v1 (or
            contacts-v1.sequence_only) document, or contains ``<think>`` tokens
            (not supported — a think-augmented document's inter-statement slots
            need their own treatment; see the SPEC's *Think (pause) tokens*).
    """
    if THINK_TOKEN in tokens:
        raise ValueError(
            "soft targets are not defined for think-augmented documents "
            f"({THINK_TOKEN} present); see #201"
        )
    if len(tokens) < 3:
        raise ValueError(f"document too short to be contacts-v1: {len(tokens)} tokens")
    doc_type = tokens[0]
    if doc_type not in (DOC_TYPE_TOKEN, SEQUENCE_ONLY_DOC_TYPE_TOKEN):
        raise ValueError(f"unexpected doc-type token {doc_type!r}")
    if tokens[1] != BEGIN_SEQUENCE_TOKEN:
        raise ValueError(f"expected {BEGIN_SEQUENCE_TOKEN} at index 1, got {tokens[1]!r}")

    statements_start = 2
    statements: list[tuple[str, str]] = []
    i = statements_start
    while i < len(tokens) and tokens[i] not in (BEGIN_STRUCTURE_TOKEN, END_TOKEN):
        if i + 1 >= len(tokens):
            raise ValueError(f"truncated sequence statement at token index {i}")
        statements.append((tokens[i], tokens[i + 1]))
        i += 2

    if i >= len(tokens):
        raise ValueError("sequence section is not terminated")

    contacts: list[tuple[str, str]] = []
    if tokens[i] == BEGIN_STRUCTURE_TOKEN:
        structure_start: int | None = i + 1
        j = i + 1
        while j < len(tokens) and tokens[j] != END_TOKEN:
            if tokens[j] != CONTACT_TOKEN:
                raise ValueError(
                    f"expected {CONTACT_TOKEN} at token index {j}, got {tokens[j]!r}"
                )
            if j + 2 >= len(tokens):
                raise ValueError(f"truncated contact statement at token index {j}")
            contacts.append((tokens[j + 1], tokens[j + 2]))
            j += 3
        if j >= len(tokens):
            raise ValueError("structure section is not terminated")
        end_index = j
    else:
        structure_start = None
        end_index = i

    if end_index != len(tokens) - 1:
        raise ValueError(
            f"{END_TOKEN} at index {end_index} is not the last token "
            f"({len(tokens)} tokens total)"
        )
    return ParsedDocument(
        doc_type=doc_type,
        statements=tuple(statements),
        contacts=tuple(contacts),
        statements_start=statements_start,
        structure_start=structure_start,
        end_index=end_index,
    )


def _one_hot(target_index: int, token: str, kind: str) -> SoftTarget:
    return SoftTarget(target_index=target_index, support=(token,), probs=(1.0,), kind=kind)


def _uniform(target_index: int, tokens: Sequence[str], kind: str) -> SoftTarget:
    n = len(tokens)
    return SoftTarget(
        target_index=target_index,
        support=tuple(tokens),
        probs=(1.0 / n,) * n,
        kind=kind,
    )


def soft_targets(tokens: Sequence[str]) -> list[SoftTarget]:
    """Exact conditional distribution for every predicted token in a document.

    Args:
        tokens: The document's tokens.

    Returns:
        One :class:`SoftTarget` per supervised slot, ordered by
        ``target_index`` and covering ``tokens[1:]`` exactly once. The realised
        token always lies in its target's support.
    """
    parsed = parse_document(tokens)
    targets: list[SoftTarget] = [_one_hot(1, BEGIN_SEQUENCE_TOKEN, FRAME)]

    # --- Sequence section: a uniform shuffle of L + 2 statements -------------
    # At the slot for statement k the remaining statements are exactly
    # ``statements[k:]`` (the document order IS the sampled permutation), and
    # their heads are distinct, so the head target is uniform over them.
    heads = [head for head, _ in parsed.statements]
    for k, (head, body) in enumerate(parsed.statements):
        head_index = parsed.statements_start + 2 * k
        targets.append(_uniform(head_index, heads[k:], STATEMENT_HEAD))
        targets.append(_one_hot(head_index + 1, body, STATEMENT_BODY))

    # The token after the last statement: whichever marker actually closes the
    # sequence section. One-hot -- it encodes "no statements remain", which is
    # real information (the model does not know the chain length a priori).
    section_end_index = parsed.statements_start + 2 * len(parsed.statements)
    targets.append(_one_hot(section_end_index, tokens[section_end_index], FRAME))

    if parsed.structure_start is None:
        return targets

    # --- Structure section: a uniform shuffle of N contacts, each with a fair
    # endpoint coin flip ------------------------------------------------------
    n_contacts = len(parsed.contacts)
    # Degree of each position over the contacts still to be emitted, maintained
    # by peeling contacts off the front as we walk the document. ``incident``
    # is the adjacency with emission times, built once so the second-endpoint
    # support costs O(degree) per slot rather than a rescan of the tail.
    remaining_degree: Counter[str] = Counter()
    incident: dict[str, list[tuple[int, str]]] = {}
    for k, (first, second) in enumerate(parsed.contacts):
        remaining_degree[first] += 1
        remaining_degree[second] += 1
        incident.setdefault(first, []).append((k, second))
        incident.setdefault(second, []).append((k, first))

    for k, (first, second) in enumerate(parsed.contacts):
        contact_index = parsed.structure_start + 3 * k
        remaining = n_contacts - k

        # <contact> vs <end>: one-hot, and real information (it is how the
        # document says how many contacts the structure has).
        targets.append(_one_hot(contact_index, CONTACT_TOKEN, FRAME))

        # First endpoint: deg_R(p) / (2 |R|). ``remaining_degree`` IS the support
        # -- positions are deleted as they run out -- so this is one pass over
        # the dict with no membership tests. That matters: this is the hot loop
        # (support size ~ chain length, once per contact), and it dominates the
        # cost of walking a document.
        norm = 2.0 * remaining
        targets.append(SoftTarget(
            target_index=contact_index + 1,
            support=tuple(remaining_degree),
            probs=tuple(degree / norm for degree in remaining_degree.values()),
            kind=FIRST_ENDPOINT,
        ))

        # Second endpoint: uniform over ``first``'s remaining partners -- the far
        # ends of its incident contacts not yet emitted. A pair is emitted at
        # most once per document, so this is a genuine set.
        partners = sorted(
            other for time, other in incident[first] if time >= k
        )
        targets.append(_uniform(contact_index + 2, partners, SECOND_ENDPOINT))

        remaining_degree[first] -= 1
        if remaining_degree[first] == 0:
            del remaining_degree[first]
        remaining_degree[second] -= 1
        if remaining_degree[second] == 0:
            del remaining_degree[second]

    targets.append(_one_hot(parsed.end_index, END_TOKEN, FRAME))
    return targets


def permutation_entropy(
    tokens: Sequence[str], *, targets: Sequence[SoftTarget] | None = None
) -> EntropyBreakdown:
    """Nuisance (permutation) entropy of a document, split by section.

    This is the cross-entropy an oracle with perfect knowledge of the structure
    would still pay under one-hot supervision — the irreducible floor that both
    the one-hot and the order-marginalized loss sit on top of.

    Args:
        tokens: The document's tokens.
        targets: Precomputed :func:`soft_targets` output for ``tokens``. Pass it
            when you already have it; building the targets dominates the cost.

    Returns:
        The per-section breakdown.
    """
    parsed = parse_document(tokens)
    if targets is None:
        targets = soft_targets(tokens)
    sequence_nats = 0.0
    structure_nats = 0.0
    for target in targets:
        if target.kind == STATEMENT_HEAD:
            sequence_nats += target.entropy
        elif target.kind in (FIRST_ENDPOINT, SECOND_ENDPOINT):
            structure_nats += target.entropy
    return EntropyBreakdown(
        sequence_nats=sequence_nats,
        structure_nats=structure_nats,
        num_tokens=len(tokens),
        num_predicted=len(tokens) - 1,
        seq_len=len(parsed.statements) - 2,
        num_contacts=len(parsed.contacts),
    )


def hard_cross_entropy(tokens: Sequence[str], log_probs: Sequence[dict[str, float]]) -> float:
    """Mean one-hot next-token cross-entropy, in nats per predicted token.

    Args:
        tokens: The document's tokens.
        log_probs: One mapping per supervised slot (aligned with
            ``soft_targets(tokens)``), giving ``log p_model(token | prefix)`` for
            at least every token in that slot's support and the realised token.

    Returns:
        The mean over supervised slots.
    """
    targets = soft_targets(tokens)
    if len(log_probs) != len(targets):
        raise ValueError(f"expected {len(targets)} log-prob maps, got {len(log_probs)}")
    total = -sum(lp[tokens[t.target_index]] for t, lp in zip(targets, log_probs))
    return total / len(targets)


def soft_cross_entropy(tokens: Sequence[str], log_probs: Sequence[dict[str, float]]) -> float:
    """Mean order-marginalized next-token cross-entropy, in nats per predicted token.

    Equals :func:`hard_cross_entropy` minus the permutation entropy **in
    expectation over orderings**, and equals ``KL(q || p_model) + H(q)``
    pointwise.

    Args:
        tokens: The document's tokens.
        log_probs: As for :func:`hard_cross_entropy`.

    Returns:
        The mean over supervised slots.
    """
    targets = soft_targets(tokens)
    if len(log_probs) != len(targets):
        raise ValueError(f"expected {len(targets)} log-prob maps, got {len(log_probs)}")
    total = 0.0
    for target, lp in zip(targets, log_probs):
        total -= sum(p * lp[token] for token, p in zip(target.support, target.probs) if p > 0.0)
    return total / len(targets)


__all__ = [
    "FIRST_ENDPOINT",
    "FRAME",
    "SECOND_ENDPOINT",
    "SOFT_KINDS",
    "STATEMENT_BODY",
    "STATEMENT_HEAD",
    "EntropyBreakdown",
    "ParsedDocument",
    "SoftTarget",
    "hard_cross_entropy",
    "parse_document",
    "permutation_entropy",
    "soft_cross_entropy",
    "soft_targets",
]
