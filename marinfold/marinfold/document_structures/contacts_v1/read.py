# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a contacts-v1 structure section back into a contact set.

Pure (regex + a set fold) — no pyconfind, no torch, no tokenizer. This is
the inverse of the ``<contact>`` / ``<retract>`` statements a document's
structure section carries, and the **semantic definition of retraction**
(issue #158):

    the structure section is an ordered *edit list*, not an unordered set.
    ``<contact> <pX> <pY>`` asserts a pair; ``<retract> <pX> <pY>`` takes
    back a currently-live pair; the final contact set is whatever is still
    live when the section ends (``<end>``).

A retract may reference a contact emitted arbitrarily far back — the fold
handles distance for free. A document with **no** ``<retract>`` statements
folds to exactly the set of pairs it emitted, so this is a no-op for every
pre-retraction contacts-v1 document and model (the current models never emit
``<retract>``); the rollout readout in ``inference.py`` can adopt it without
changing any existing score.

Pairs are canonicalised to a sorted ``(min, max)`` of the two **position**
indices, matching the coin-flipped orientation the generator emits — so a
retract matches its contact regardless of which order either was written in.
Callers map positions back to sequence indices (and drop the near-diagonal
band); this module stays in position space and does not editorialise
(degenerate ``<contact> <pX> <pX>`` self-pairs are kept as ``(X, X)`` and
left for the caller to filter, exactly as the old regex scan did).

Malformed edit lists are tolerated so inference-time parsing is robust to
model noise, but every anomaly is counted (see :class:`FoldResult`) so the
corpus job (issue #159) can *assert* they never occur in authored data.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# One structure-section statement: <contact>/<retract> then two <pN> tokens.
# Whitespace-tolerant (documents are single-space-joined, but decoded
# rollouts may vary), mirroring exp82's CONTACT_RE with <retract> added.
_STATEMENT_RE = re.compile(r"<(contact|retract)>\s+<p(\d+)>\s+<p(\d+)>")

CONTACT = "contact"
RETRACT = "retract"

# A canonicalised structure-section pair: sorted (min, max) position indices.
Pair = tuple[int, int]


def iter_structure_statements(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield ``(kind, pos_a, pos_b)`` for each statement, in stream order.

    ``kind`` is ``"contact"`` or ``"retract"``; ``pos_a`` / ``pos_b`` are the
    two position indices as written (orientation not yet canonicalised).
    ``re.findall`` scans left-to-right, so order is preserved — which is what
    makes the fold well defined.
    """
    for kind, a, b in _STATEMENT_RE.findall(text):
        yield kind, int(a), int(b)


def _canonical(a: int, b: int) -> Pair:
    """Sorted ``(min, max)`` position pair (orientation-independent)."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class FoldResult:
    """The live contact set plus counts of every anomaly seen while folding.

    ``live`` is the set of canonical position pairs live at the end of the
    stream. The counters exist so authored corpora can assert cleanliness
    (all zero) while inference stays tolerant:

    - ``n_contact`` / ``n_retract``: total statements of each kind.
    - ``n_retract_absent``: a ``<retract>`` of a pair that was not live
      (never emitted, or already retracted) — a no-op.
    - ``n_reemit``: a ``<contact>`` re-asserting a pair that had been
      retracted earlier (a legitimate move — the pair comes back live).
    - ``n_redundant_contact``: a ``<contact>`` for a pair already live — a
      no-op restatement.
    """

    live: frozenset[Pair]
    n_contact: int
    n_retract: int
    n_retract_absent: int
    n_reemit: int
    n_redundant_contact: int


def fold_statements(statements: Iterable[tuple[str, int, int]]) -> FoldResult:
    """Fold an ordered statement stream into the live contact set + anomalies.

    Start from the empty set; ``<contact>`` adds the canonical pair,
    ``<retract>`` removes it. A retract of a non-live pair is ignored (and
    counted); a contact of a previously retracted pair brings it back (and is
    counted as a re-emit). See :class:`FoldResult`.
    """
    live: set[Pair] = set()
    retracted_ever: set[Pair] = set()
    n_contact = n_retract = 0
    n_retract_absent = n_reemit = n_redundant_contact = 0

    for kind, a, b in statements:
        pair = _canonical(a, b)
        if kind == CONTACT:
            n_contact += 1
            if pair in live:
                n_redundant_contact += 1
            elif pair in retracted_ever:
                n_reemit += 1
            live.add(pair)
            retracted_ever.discard(pair)
        else:  # RETRACT
            n_retract += 1
            if pair in live:
                live.discard(pair)
                retracted_ever.add(pair)
            else:
                n_retract_absent += 1

    return FoldResult(
        live=frozenset(live),
        n_contact=n_contact,
        n_retract=n_retract,
        n_retract_absent=n_retract_absent,
        n_reemit=n_reemit,
        n_redundant_contact=n_redundant_contact,
    )


def live_contacts(text: str) -> frozenset[Pair]:
    """Canonical position pairs live at ``<end>`` — the common-case entry point.

    Equivalent to ``fold_statements(iter_structure_statements(text)).live``.
    For a document with no ``<retract>`` statements this is exactly the set of
    emitted pairs.
    """
    return fold_statements(iter_structure_statements(text)).live
