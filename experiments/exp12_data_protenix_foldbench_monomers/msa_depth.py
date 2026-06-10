# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""MSA depth metrics from a Protenix/ColabFold a3m file.

Computes two flavours of "how deep is this MSA" per protein, both
derived from the merged unpaired MSA (``non_pairing.a3m``) that Protenix
actually feeds to the model:

- ``n_seqs`` — the raw number of sequences in the a3m (including the
  query as the first record). The simple "total entries" count.

- ``n_eff`` — the redundancy-reweighted effective sequence count
  (a.k.a. Meff), computed at two sequence-identity thresholds (0.8 and
  0.62). This down-weights near-duplicate sequences so that a thousand
  copies of the same homolog count for far less than a thousand diverse
  ones.

### Definitions (pinned)

a3m match-state extraction: in a3m, uppercase letters and ``-`` are
*match* columns (aligned to the query); lowercase letters are
*insertions* relative to the query and are dropped. After stripping
insertions every sequence has the same length ``L`` (the query length).

Pairwise identity between two aligned sequences i, j::

    identity(i, j) = (# columns where both non-gap AND equal)
                     / (# columns where both non-gap)

i.e. identity over the overlapping (both-resolved) region. If the
overlap is empty the identity is 0. This is robust to partial-coverage
hits (a short fragment that perfectly matches its overlap clusters with
the query rather than looking distant because of its gaps).

Effective count (Meff) at threshold ``t``::

    w_i  = 1 / |{ j : identity(i, j) >= t }|      (the set includes i)
    n_eff = sum_i w_i

The query always contributes (identity(i, i) = 1 >= t for any t <= 1).
A set of fully diverse sequences gives ``n_eff == n_seqs``; a set of
identical sequences gives ``n_eff == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Default sequence-identity thresholds. 0.8 is the AlphaFold2 / HHblits
# convention; 0.62 is common in the coevolution / contact-prediction
# literature (e.g. plmc's default theta=0.2 → 80%, but EVcouplings and
# CCMpred analyses often quote 62%). We keep both so depth can be read
# either way.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.8, 0.62)

# Byte value of '-' (the gap character) once sequences are uppercased.
_GAP_BYTE = ord("-")


@dataclass(frozen=True)
class MSADepth:
    """Per-protein MSA depth metrics."""

    n_seqs: int                       # total records in the a3m (incl. query)
    query_len: int                    # L = number of match columns
    n_eff: dict[float, float]         # threshold -> effective sequence count


def parse_a3m_sequences(text: str) -> list[str]:
    """Return the raw sequence strings from a3m ``text`` (one per record).

    Sequence lines are concatenated until the next ``>`` header. Empty
    input yields an empty list. Insertion (lowercase) characters are
    *kept* here; :func:`a3m_to_match_array` strips them.
    """
    seqs: list[str] = []
    current: list[str] = []
    have_record = False
    for line in text.splitlines():
        if line.startswith(">"):
            if have_record:
                seqs.append("".join(current))
            current = []
            have_record = True
        elif line:
            current.append(line.strip())
    if have_record:
        seqs.append("".join(current))
    return seqs


def a3m_to_match_array(seqs: list[str]) -> np.ndarray:
    """Convert a3m sequences to an ``[N, L]`` uint8 match-state array.

    Insertion columns (lowercase letters) are dropped; the remaining
    uppercase letters and ``-`` form the per-sequence match state. All
    sequences must reduce to the same length ``L`` (true for a
    well-formed a3m aligned to a single query). Returns an empty
    ``(0, 0)`` array if ``seqs`` is empty.
    """
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint8)
    rows: list[np.ndarray] = []
    for s in seqs:
        # Keep only match columns: uppercase letters and gaps. Lowercase
        # letters are insertions relative to the query and are removed.
        match = bytes(c for c in s.encode("ascii", "ignore") if c == _GAP_BYTE or 65 <= c <= 90)
        rows.append(np.frombuffer(match, dtype=np.uint8))
    lengths = {r.shape[0] for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"a3m sequences have inconsistent match-state lengths: {sorted(lengths)}"
        )
    return np.vstack(rows)


def compute_neff(
    arr: np.ndarray,
    *,
    threshold: float,
    gap_byte: int = _GAP_BYTE,
    max_block_elems: int = 50_000_000,
) -> float:
    """Effective sequence count (Meff) for match-state array ``arr``.

    ``arr`` is ``[N, L]`` uint8 (see :func:`a3m_to_match_array`).
    Pairwise identity uses the both-non-gap overlap (see module docstring).
    Computed in row blocks so peak memory stays near ``max_block_elems``
    booleans regardless of MSA depth.
    """
    n, length = arr.shape
    if n == 0:
        return 0.0
    if length == 0:
        # No match columns at all; every sequence is its own (empty)
        # cluster -> n_eff == n. Degenerate, shouldn't happen in practice.
        return float(n)
    nongap = arr != gap_byte                       # [N, L] bool
    neighbor_counts = np.zeros(n, dtype=np.int64)
    block = max(1, max_block_elems // (n * length))
    for start in range(0, n, block):
        end = min(start + block, n)
        a = arr[start:end]                          # [B, L]
        a_ng = nongap[start:end]                     # [B, L]
        both = a_ng[:, None, :] & nongap[None, :, :]  # [B, N, L] both resolved
        eq = (a[:, None, :] == arr[None, :, :]) & both
        matches = eq.sum(axis=2)                      # [B, N]
        overlap = both.sum(axis=2)                    # [B, N]
        with np.errstate(invalid="ignore", divide="ignore"):
            ident = np.where(overlap > 0, matches / overlap, 0.0)
        neighbor_counts[start:end] = (ident >= threshold).sum(axis=1)
    return float(np.sum(1.0 / neighbor_counts))


def msa_depth(
    a3m_text: str,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> MSADepth:
    """Compute :class:`MSADepth` from the text of an a3m file."""
    seqs = parse_a3m_sequences(a3m_text)
    arr = a3m_to_match_array(seqs)
    n_eff = {t: compute_neff(arr, threshold=t) for t in thresholds}
    return MSADepth(
        n_seqs=arr.shape[0],
        query_len=arr.shape[1],
        n_eff=n_eff,
    )
