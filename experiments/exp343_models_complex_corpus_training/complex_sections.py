# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Label every token of a multi-chain contacts-v1 document by what it encodes.

A model can score well on a complex document by predicting its intra-chain
contacts alone -- those are the same problem it already solves on monomers. The
question this experiment asks is whether it learned the *interface*, so the loss
has to be split by token role, and inter-chain contacts separated from
intra-chain ones.

Chain membership is recoverable from the document itself. Each chain occupies one
unbroken run of the shared 2000-index ring, the runs are disjoint and separated
by at least one unused index, and each chain contributes an `<n-term> <pX>` and a
`<c-term> <pY>` statement ([#222](https://github.com/Open-Athena/MarinFold/issues/222),
`SPEC.md` "Multiple protein chains"). Walking forward from each `<n-term>`, the
first `<c-term>` reached closes that chain.

`chain_lengths` and `contacts_emitted_inter_chain` from the published corpus are
never used to *do* the labelling -- only to check it, which is what
`test_complex_sections.py` and `verify_sections.py` do over real documents.
"""

from dataclasses import dataclass
from enum import Enum

#: The ring the position tokens live on. `<p0>`..`<p1999>` are ids 5..2004 in the
#: contacts-v1 vocabulary, but this module works on token *text*, so only the
#: modulus matters.
RING = 2000
POSITION_PREFIX = "<p"
BEGIN_SEQUENCE = "<begin_sequence>"
BEGIN_STATEMENTS = "<begin_statements>"
END = "<end>"
CONTACT = "<contact>"
N_TERM = "<n-term>"
C_TERM = "<c-term>"


class Role(Enum):
    """What a token position in the document encodes."""

    HEADER = "header"
    SEQUENCE = "sequence"
    TERMINUS = "terminus"
    CONTACT_INTRA = "contact_intra"
    CONTACT_INTER = "contact_inter"
    CONTACT_UNRESOLVED = "contact_unresolved"
    END = "end"


@dataclass(frozen=True)
class Sections:
    """One document's token roles and the chain layout they were derived from."""

    tokens: tuple[str, ...]
    roles: tuple[Role, ...]
    chain_of_index: dict[int, int]
    num_chains: int
    contacts_intra: int
    contacts_inter: int
    contacts_unresolved: int


def position(token: str) -> int | None:
    """The ring index a `<pN>` token names, or None for any other token."""
    if not token.startswith(POSITION_PREFIX) or not token.endswith(">"):
        return None
    body = token[len(POSITION_PREFIX) : -1]
    return int(body) if body.isdigit() else None


def chain_runs(termini: list[tuple[str, int]]) -> list[tuple[int, int]]:
    """Pair each `<n-term>` index with the `<c-term>` that closes its run.

    Walking the ring forward from an N-terminus, the first C-terminus reached
    belongs to that chain: the runs are disjoint, so no other chain's residues
    can lie between a chain's two ends.

    Args:
        termini: `(kind, index)` pairs in document order, kind `<n-term>` or
            `<c-term>`.

    Returns:
        One `(n_term_index, c_term_index)` pair per chain, N-terminus-ordered.

    Raises:
        ValueError: the counts differ, an index repeats, or some N-terminus
            reaches no C-terminus before the next N-terminus.
    """
    starts = sorted(index for kind, index in termini if kind == N_TERM)
    ends = {index for kind, index in termini if kind == C_TERM}
    if len(starts) != len(ends) or len(starts) + len(ends) != len(termini):
        raise ValueError(
            f"{len(starts)} N-termini and {len(ends)} C-termini over "
            f"{len(termini)} terminus statements"
        )
    if len(set(starts)) != len(starts):
        raise ValueError(f"repeated N-terminus index in {starts}")
    runs = []
    for rank, start in enumerate(starts):
        limit = starts[(rank + 1) % len(starts)] if len(starts) > 1 else start
        for step in range(RING):
            index = (start + step) % RING
            if index in ends:
                runs.append((start, index))
                break
            if step and index == limit:
                raise ValueError(
                    f"chain starting at {start} reaches the next N-terminus "
                    f"{limit} before any C-terminus"
                )
        else:
            raise ValueError(f"chain starting at {start} has no C-terminus")
    if len({end for _, end in runs}) != len(runs):
        raise ValueError(f"two chains claim the same C-terminus in {runs}")
    return runs


def index_to_chain(runs: list[tuple[int, int]]) -> dict[int, int]:
    """Map every occupied ring index to its chain's rank.

    Raises:
        ValueError: two chains overlap, which would make the runs non-disjoint
            and every inter-chain judgement downstream unsound.
    """
    owner: dict[int, int] = {}
    for rank, (start, end) in enumerate(runs):
        index = start
        while True:
            if index in owner:
                raise ValueError(
                    f"ring index {index} is claimed by chains {owner[index]} "
                    f"and {rank}"
                )
            owner[index] = rank
            if index == end:
                break
            index = (index + 1) % RING
    return owner


def label(document: str) -> Sections:
    """Label a whole document, deriving its chain layout on the way.

    A contact naming an index no chain claims is `CONTACT_UNRESOLVED` rather than
    an error: a truncated document can retain a contact whose terminus statement
    was dropped, and silently folding those into either class would bias the
    split.
    """
    tokens = tuple(document.split())
    if not tokens or BEGIN_SEQUENCE not in tokens or BEGIN_STATEMENTS not in tokens:
        raise ValueError("Document is not a contacts-v1 document")
    begin_sequence = tokens.index(BEGIN_SEQUENCE)
    begin_statements = tokens.index(BEGIN_STATEMENTS)
    termini = [
        (token, position(tokens[cursor + 1]))
        for cursor, token in enumerate(tokens)
        if token in (N_TERM, C_TERM) and cursor + 1 < len(tokens)
    ]
    if any(index is None for _, index in termini):
        raise ValueError("A terminus statement is not followed by a position token")
    runs = chain_runs([(kind, index) for kind, index in termini if index is not None])
    owner = index_to_chain(runs)

    roles = [Role.HEADER] * len(tokens)
    for cursor in range(begin_sequence + 1, begin_statements):
        roles[cursor] = Role.SEQUENCE
    counts = {Role.CONTACT_INTRA: 0, Role.CONTACT_INTER: 0, Role.CONTACT_UNRESOLVED: 0}
    cursor = begin_statements + 1
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == END:
            roles[cursor] = Role.END
            cursor += 1
            continue
        if token in (N_TERM, C_TERM):
            roles[cursor] = roles[cursor + 1] = Role.TERMINUS
            cursor += 2
            continue
        if token != CONTACT:
            raise ValueError(f"Unexpected statement token {token!r} at {cursor}")
        left, right = position(tokens[cursor + 1]), position(tokens[cursor + 2])
        if left is None or right is None:
            raise ValueError(f"Malformed contact statement at {cursor}")
        if left not in owner or right not in owner:
            role = Role.CONTACT_UNRESOLVED
        elif owner[left] == owner[right]:
            role = Role.CONTACT_INTRA
        else:
            role = Role.CONTACT_INTER
        roles[cursor] = roles[cursor + 1] = roles[cursor + 2] = role
        counts[role] += 1
        cursor += 3
    return Sections(
        tokens=tokens,
        roles=tuple(roles),
        chain_of_index=owner,
        num_chains=len(runs),
        contacts_intra=counts[Role.CONTACT_INTRA],
        contacts_inter=counts[Role.CONTACT_INTER],
        contacts_unresolved=counts[Role.CONTACT_UNRESOLVED],
    )
