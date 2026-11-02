# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Consensus scoring and leave-one-out marginals for exp208 — issue #208.

THE POINT. #208's document-level reward is a rollout's **marginal contribution to
the group's consensus**, not its own F1. That is deliberate: the number MarinFold
reports is a consensus over 100 rollouts, and #82 recorded that "over-sharpening
collapses the vote" — so a precision-only reward can raise per-rollout quality and
*lower* the reported metric by making every rollout agree. Scoring the marginal
makes the objective literally the deployed metric, and pays a rollout for
contributing something its siblings did not.

MATCHING THE DEPLOYED METRIC EXACTLY. The eval path is
``score_rollout_worker.py`` -> ``fetch_cw_scores.py`` -> ``build_rollout_rows.py``
(which carries exp89's ``compute_metrics`` verbatim). Reading those three end to
end, the ranking is:

1. candidates are ordered pairs ``(i, j)``, ``i < j``, both **resolved** in the GT
   structure, taken in ``np.triu_indices`` order over the resolved index array;
2. the score is the **integer vote count** and nothing else — ``fetch_cw_scores``
   writes ``M[i, j] = votes`` into a float16 matrix, so there is no pairwise
   probability term anywhere on this path despite "pairwise tie-break" appearing
   in some prose;
3. ranking is ``np.argsort(-score, kind="mergesort")`` — a **stable** sort, so
   equal vote counts are broken by ascending candidate index;
4. R-precision takes the top ``R`` where ``R`` is the number of true contacts
   *within the range*, and divides the hits by ``min(R, n_candidates)``.

Point 3 is why this module can be exact rather than approximate: the tie-break is
positional, so it costs nothing to reproduce in the training loop.

Point 4 has a consequence worth stating because it is the quantitative form of
"vote collapse": when fewer than ``R`` candidate pairs receive any vote at all,
the remaining slots are filled by **zero-vote pairs in index order**, which are
essentially arbitrary and almost never true. So a policy that emits fewer, more
confident contacts can shrink the union below ``R`` and lose R-precision even
though every pair it did emit got better. ``group_diagnostics`` reports
``union_over_r`` for exactly this reason.

COST. One ``argsort`` over the candidate universe per leave-one-out evaluation:
``G`` sorts of ``P`` candidates per protein. At the eval shape (``G`` = 100,
median ``L`` = 161 so ``P`` ~ 13k) that is ~0.1 s/protein; at the training shape
(``G`` = 16, ``L`` <= 512 so ``P`` <= 131k) it is ~0.2 s/protein against a ~37 s
sampling call. Not worth optimizing, and the naive version is the one that is
obviously the same function as the eval path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

MIN_SEP = 6

Pair = tuple[int, int]


def candidate_index(
    length: int,
    *,
    resolved: Sequence[int] | None = None,
    min_sep: int = MIN_SEP,
    max_sep: int | None = None,
) -> tuple[np.ndarray, dict[Pair, int]]:
    """The ranked candidate universe, in the deployed metric's canonical order.

    Args:
        length: Number of residues in the input sequence.
        resolved: Input-sequence indices resolved in the GT structure. ``None``
            means every residue is a candidate — which is the training case
            (AFDB documents carry every residue) and NOT the eval case, where the
            universe is restricted so that every predictor is scored on the same
            pairs.
        min_sep: Inclusive lower bound on ``j - i``.
        max_sep: Inclusive upper bound, or ``None`` for the ``all`` band.

    Returns:
        ``(pairs, position)`` where ``pairs`` is an ``[P, 2]`` int array in
        ``np.triu_indices`` order over ``resolved`` (the order the metric's stable
        sort falls back on for ties), and ``position`` maps a pair to its row.
    """
    index = np.arange(length, dtype=np.int64) if resolved is None else np.asarray(resolved, dtype=np.int64)
    a, b = np.triu_indices(len(index), k=1)
    i, j = index[a], index[b]
    sep = j - i
    keep = sep >= min_sep
    if max_sep is not None:
        keep &= sep <= max_sep
    pairs = np.stack([i[keep], j[keep]], axis=1)
    position = {(int(p), int(q)): k for k, (p, q) in enumerate(pairs)}
    return pairs, position


def vote_counts(
    pair_sets: Sequence[Iterable[Pair]],
    position: dict[Pair, int],
    n_candidates: int,
) -> np.ndarray:
    """Per-candidate vote counts, ``[G, P]``, one row per rollout.

    Pairs outside the candidate universe are dropped — that is the metric's own
    behaviour (an unresolved residue simply is not a candidate), not a silent
    correction.
    """
    votes = np.zeros((len(pair_sets), n_candidates), dtype=np.int32)
    for r, pairs in enumerate(pair_sets):
        for pair in pairs:
            slot = position.get(pair)
            if slot is not None:
                votes[r, slot] = 1
    return votes


def rprecision(votes: np.ndarray, is_true: np.ndarray, n_true: int) -> float:
    """R-precision of one vote vector, exactly as ``build_rollout_rows`` computes it.

    Args:
        votes: ``[P]`` summed vote counts over the candidate universe.
        is_true: ``[P]`` boolean, whether each candidate is a true contact.
        n_true: ``R`` — the number of true contacts in this range. Passed in
            rather than derived from ``is_true`` so a caller can score a band
            whose GT it computed elsewhere; they must agree.

    Returns:
        Precision at top-R, or NaN when there is nothing to score.
    """
    n_candidates = int(votes.size)
    if n_candidates == 0 or n_true <= 0:
        return math.nan
    order = np.argsort(-votes, kind="mergesort")
    top = min(int(n_true), n_candidates)
    return float(is_true[order[:top]].sum()) / top


def loo_marginals(
    votes: np.ndarray, is_true: np.ndarray, n_true: int
) -> tuple[float, np.ndarray]:
    """Consensus score and each rollout's leave-one-out marginal contribution.

    ``A_i = C(all) - C(all \\ {i})``. An empty rollout changes no votes and so
    scores exactly 0 — which, once the group is centred, is a *negative*
    advantage against siblings that contributed. That is the property that makes
    this term oppose collapse-to-silence rather than merely tolerate it.

    Args:
        votes: ``[G, P]`` per-rollout indicator matrix from :func:`vote_counts`.
        is_true: ``[P]`` boolean truth mask over the candidate universe.
        n_true: ``R``.

    Returns:
        ``(consensus, marginals)`` — the group's R-precision, and a ``[G]``
        float64 array of marginals. Both are NaN / all-NaN when unscoreable.
    """
    total = votes.sum(axis=0)
    consensus = rprecision(total, is_true, n_true)
    if math.isnan(consensus):
        return consensus, np.full(len(votes), math.nan)
    marginals = np.empty(len(votes), dtype=np.float64)
    for i in range(len(votes)):
        marginals[i] = consensus - rprecision(total - votes[i], is_true, n_true)
    return consensus, marginals


def group_diagnostics(votes: np.ndarray, is_true: np.ndarray, n_true: int) -> dict[str, float]:
    """The collapse detectors, computed on one protein's group of rollouts.

    ``union_over_r`` below 1.0 means the top-R selection is being padded with
    zero-vote pairs, which is vote collapse in its most direct form.
    ``mean_jaccard`` here is **between rollouts**, unlike #200's same-named metric
    which was between candidate sections inside one multi-draft rollout. In plain
    mode that one is NaN by construction, which is why this exists.
    """
    n_rollouts = int(votes.shape[0])
    total = votes.sum(axis=0)
    voted = total > 0
    sizes = votes.sum(axis=1)

    jaccards: list[float] = []
    for i in range(n_rollouts):
        for j in range(i + 1, n_rollouts):
            union = int(np.count_nonzero(votes[i] | votes[j]))
            if union:
                jaccards.append(float(np.count_nonzero(votes[i] & votes[j])) / union)

    # Vote-mass entropy over voted pairs: falls as the group concentrates on a
    # shrinking set, which is the early warning the issue asks for.
    mass = total[voted].astype(np.float64)
    entropy = math.nan
    if mass.size:
        prob = mass / mass.sum()
        entropy = float(-(prob * np.log(prob)).sum())

    order = np.argsort(-total, kind="mergesort")
    top = min(int(n_true), int(total.size)) if n_true > 0 else 0
    return {
        "n_rollouts": float(n_rollouts),
        "n_candidates": float(total.size),
        "n_true": float(n_true),
        "union": float(np.count_nonzero(voted)),
        "union_over_r": (float(np.count_nonzero(voted)) / n_true) if n_true > 0 else math.nan,
        "mean_pairs_per_rollout": float(sizes.mean()) if n_rollouts else math.nan,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else math.nan,
        "vote_entropy": entropy,
        "mean_vote_top_r": float(total[order[:top]].mean()) if top else math.nan,
        "consensus_rprec": rprecision(total, is_true, n_true),
    }


def rollout_precision_recall(votes_row: np.ndarray, is_true: np.ndarray, n_true: int) -> dict[str, float]:
    """Single-rollout precision / recall / F1 over the candidate universe.

    Reported separately from precision on purpose: the stepwise reward has an
    explicit incentive to stop early, so a precision gain paid for with recall is
    a null result rather than an improvement, and only separate reporting catches
    it.
    """
    n_pred = int(votes_row.sum())
    hits = int(np.count_nonzero(votes_row & is_true))
    precision = (hits / n_pred) if n_pred else math.nan
    recall = (hits / n_true) if n_true > 0 else math.nan
    if not n_pred or n_true <= 0 or (hits == 0):
        f1 = 0.0 if (n_pred or n_true > 0) else math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"n_pred": float(n_pred), "n_hit": float(hits),
            "precision": precision, "recall": recall, "f1": f1}


def truth_mask(pairs: np.ndarray, gt: set[Pair]) -> np.ndarray:
    """Boolean mask over the candidate array for a ground-truth pair set."""
    return np.fromiter(
        ((int(i), int(j)) in gt for i, j in pairs), dtype=bool, count=len(pairs)
    )


__all__ = [
    "MIN_SEP",
    "candidate_index",
    "group_diagnostics",
    "loo_marginals",
    "rollout_precision_recall",
    "rprecision",
    "truth_mask",
    "vote_counts",
]
