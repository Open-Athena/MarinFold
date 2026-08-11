# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The seven contact-set arms compared in issue #211.

Every arm produces a contact set on the same protein, and — except for the
unrestricted ground-truth arm — with the **same number of contacts** as the
rollout it is matched to. Size matching is not cosmetic: #142 measured that
rollouts emit ~0.70x the ground-truth contact count, and a sparser set is
trivially easier to embed, so an unmatched comparison would read
under-generation as consistency.

| arm | constructor | role |
|-----|-------------|------|
| 1 GT                     | ``ground_truth``        | calibration ceiling; must score ~0 |
| 2 GT subsampled          | ``subsample``           | removes the contact-count confound |
| 3 within-rollout         | (the rollout itself)    | the treatment |
| 4 marginal-matched chimera | ``marginal_chimera``  | **the key null** |
| 5 splice chimera         | ``splice_chimera``      | the literal "two different rollouts" null |
| 6 separation-matched random | ``separation_matched_random`` | floor |
| 7 decoy protein          | ``decoy_protein``       | sequence-blindness ceiling (see below) |

**Why arm 4 is the sharp test.** Arms 3 and 4 come from the same model, the same
protein, the same per-pair marginals and the same set size. The *only* thing that
differs is whether the contacts were drawn jointly (one autoregressive pass) or
independently (sampled from the pooled vote distribution). Any gap between them is
joint structure the model put there at generation time, and cannot be explained by
the model's marginal accuracy — which is what every existing contacts-v1 eval
already measures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from consistency import MIN_SEP


Pair = tuple[int, int]


def _norm(pairs) -> set[Pair]:
    """Canonicalize to a set of ``(min, max)`` tuples, dropping self-pairs."""
    return {(min(i, j), max(i, j)) for i, j in pairs if i != j}


def ground_truth(gt_pairs: Sequence[Pair]) -> list[Pair]:
    """Arm 1 — the pyconfind ground-truth contact set, unmodified."""
    return sorted(_norm(gt_pairs))


def subsample(pairs: Sequence[Pair], size: int, rng: np.random.Generator) -> list[Pair]:
    """Arm 2 — a uniformly random ``size``-subset of ``pairs``.

    Used to cut the ground truth down to a rollout's contact count. If ``size``
    exceeds what is available the whole set is returned (the caller is comparing a
    rollout that over-generated, which is its own signal).
    """
    p = sorted(_norm(pairs))
    if size >= len(p):
        return p
    idx = rng.choice(len(p), size=size, replace=False)
    return sorted(p[i] for i in idx)


def marginal_chimera(
    vote_counts: dict[Pair, int], size: int, rng: np.random.Generator
) -> list[Pair]:
    """Arm 4 — ``size`` pairs drawn without replacement, probability ∝ vote count.

    ``vote_counts`` maps each pair to how many of the protein's rollouts emitted
    it — i.e. the model's own per-pair marginal, which is exactly the quantity
    every existing eval (#82/#89/#180) scores. Sampling from it reproduces those
    marginals in expectation while destroying any joint structure, so a
    within-rollout set that beats this one is carrying information the marginals
    do not.

    Weighted sampling without replacement uses the exponential-race (Efraimidis-
    Spirakis) trick: draw ``E_p / w_p`` with ``E_p ~ Exp(1)`` and take the ``size``
    smallest keys. That is a genuine weighted sample without replacement, unlike
    repeated normalize-and-draw.
    """
    pairs = sorted(vote_counts)
    if size >= len(pairs):
        return pairs
    w = np.array([vote_counts[p] for p in pairs], dtype=float)
    if not (w > 0).any():
        raise ValueError("all vote counts are zero")
    keys = rng.exponential(size=len(pairs)) / np.maximum(w, 1e-12)
    idx = np.argpartition(keys, size - 1)[:size]
    return sorted(pairs[i] for i in idx)


def splice_chimera(
    a: Sequence[Pair],
    b: Sequence[Pair],
    size: int,
    rng: np.random.Generator,
    pool: Sequence[Pair] | None = None,
) -> list[Pair]:
    """Arm 5 — half of rollout ``a`` spliced with half of rollout ``b``.

    The literal reading of "sets sampled from different rollouts": two internally
    coherent halves that were generated under different structural hypotheses.

    Rollouts overlap heavily (their high-vote pairs recur), so the naive union
    deduplicates down to fewer than ``size`` pairs. Deduplication is followed by a
    top-up from ``pool`` (default: the union of ``a`` and ``b``) so the arm stays
    size-matched. The number of pairs that had to be topped up is worth logging —
    it is a direct measure of how much two rollouts of the same protein agree.
    """
    sa, sb = sorted(_norm(a)), sorted(_norm(b))
    half = size // 2
    take_a = [sa[i] for i in rng.choice(len(sa), size=min(half, len(sa)), replace=False)]
    take_b = [sb[i] for i in rng.choice(len(sb), size=min(size - half, len(sb)), replace=False)]
    out = set(take_a) | set(take_b)

    candidates = sorted((set(_norm(pool)) if pool is not None else set(sa) | set(sb)) - out)
    if candidates and len(out) < size:
        need = min(size - len(out), len(candidates))
        out |= {candidates[i] for i in rng.choice(len(candidates), size=need, replace=False)}
    return sorted(out)


def separation_matched_random(
    pairs: Sequence[Pair], length: int, rng: np.random.Generator, *, min_sep: int = MIN_SEP
) -> list[Pair]:
    """Arm 6 — random pairs with the *same* ``|i - j|`` separation profile.

    The strong floor. Preserving the separation profile preserves contact order
    and the short/medium/long range mix, so what separates this arm from a real
    contact set is joint geometry alone — not the trivial fact that real contact
    maps are dominated by short-range contacts. Phase 0 used this null and the
    ground truth still separated cleanly from it.

    Placement is rejection-sampled per contact; a separation with no free slot
    left is skipped, so the result can be marginally smaller than the input.
    """
    src = sorted(_norm(pairs))
    seps = [j - i for i, j in src]
    taken: set[Pair] = set()
    for s in seps:
        if s >= length:
            continue
        for _ in range(200):
            i = int(rng.integers(0, length - s))
            if (i, i + s) not in taken and s >= min_sep:
                taken.add((i, i + s))
                break
    return sorted(taken)


def decoy_protein(
    other_pairs: Sequence[Pair], length: int, size: int, rng: np.random.Generator
) -> list[Pair]:
    """Arm 7 — a *different* protein's contact set, clipped to this one's length.

    **Not a floor — a second ceiling, and the sequence-blindness control.** The
    issue called this a "hard floor"; the GT gate showed that is wrong, and
    instructively so. A different real protein's contact map scores *the same as
    the true one* (median 0.0384 vs 0.0337 per contact; the true set wins on
    49.6% of proteins — a coin flip). That is correct behaviour, not a failure:
    the score sees only the contact graph and never the sequence, and a real
    contact map is a realizable 3D structure no matter which protein it came
    from. So this arm measures geometric plausibility with the *structural*
    answer swapped out, and it bounds what the metric can ever detect: it cannot
    tell "wrong fold, copied from a real one" from "right fold". What it can tell
    is "not a fold at all" — which is what separation-matched random is (median
    0.1886, 5.6x worse than the truth).

    Pairs falling outside ``[0, length)`` are dropped, then
    the set is subsampled to ``size``. Callers should pick the donor from proteins
    of similar length so that clipping removes little.
    """
    kept = [(i, j) for i, j in _norm(other_pairs) if i < length and j < length]
    return subsample(kept, size, rng)
