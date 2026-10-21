"""Evaluate ranked contact maps against the entire eligible pair universe."""

import hashlib
from collections import Counter, defaultdict

import numpy as np


def aggregate(rollouts: list[dict], method: str) -> dict[tuple[int, int], float]:
    """Combine unique per-rollout observations, with optional trajectory discount."""
    if method == 'frequency':
        counts = Counter()
        for row in rollouts:
            counts.update({(a, b) for a, b in row['pairs'] if 0 <= a < b < row['L'] and b - a >= 6})
        return {pair: float(count) for pair, count in counts.items()}
    observations = []
    for row in rollouts:
        pairs = {}
        for pair, probability in zip(row['pairs'], row['probabilities'], strict=True):
            a, b = pair
            if 0 <= a < b < row['L'] and b - a >= 6:
                pairs.setdefault((a, b), probability)
        observations.append(pairs)
    weights = np.ones(len(observations))
    if method == 'diversity_sumq':
        sets = [set(row) for row in observations]
        for i, a in enumerate(sets):
            neighbors = sum(bool(a | b) and len(a & b) / len(a | b) >= .8 for b in sets)
            weights[i] = 1 / max(1, neighbors)
    if method == 'f1_weighted_frequency':
        for i, row in enumerate(rollouts):
            unique = {}
            for pair, probability in zip(row['pairs'], row['probabilities'], strict=True):
                unique.setdefault(tuple(pair), probability)
            precision = float(np.mean(list(unique.values()))) if unique else 0.
            recall = row['predicted_recall']
            weights[i] = max(1e-8, 2 * precision * recall / max(1e-8, precision + recall))
    count, total = defaultdict(float), defaultdict(float)
    for pairs, weight in zip(observations, weights, strict=True):
        for pair, probability in pairs.items():
            count[pair] += weight
            total[pair] += weight * probability
    if method == 'f1_weighted_frequency':
        return dict(count)
    if method == 'frequency_then_meanq':
        # Integer vote counts remain strictly ordered. Confidence can resolve a
        # frequency tie but cannot outweigh even one additional independent vote.
        return {pair: count[pair] + .5 * total[pair] / count[pair] for pair in count}
    if method in {'sumq', 'diversity_sumq'}:
        return dict(total)
    if method == 'meanq':
        return {pair: total[pair] / count[pair] for pair in count}
    if method == 'sqrt_frequency_meanq':
        return {pair: total[pair] / count[pair] ** .5 for pair in count}
    raise ValueError(method)


def rank_key(pair: tuple[int, int]) -> bytes:
    """Break score ties reproducibly without looking at labels or residue order."""
    return hashlib.sha256(f'{pair[0]}:{pair[1]}'.encode()).digest()


def map_metrics(scores: dict[tuple[int, int], float], truth: set[tuple[int, int]],
                length: int, separation: int = 6) -> dict[str, float]:
    """Score sparse maps, assigning every unproposed eligible pair score zero.

    AP uses grouped score ties, including the complete zero-score tail. For
    fixed-budget precision, unfilled prediction slots count as misses; these
    methods cannot gain credit by silently guessing pairs they never proposed.
    """
    eligible = {pair: score for pair, score in scores.items()
                if 0 <= pair[0] < pair[1] < length and pair[1] - pair[0] >= separation}
    native = {pair for pair in truth if pair[1] - pair[0] >= separation}
    if any(not np.isfinite(score) or score <= 0 for score in eligible.values()):
        raise ValueError('Sparse candidate scores must be finite and positive')
    universe = max(0, (length - separation) * (length - separation + 1) // 2)
    if not native:
        return dict(r_precision=float('nan'), average_precision=float('nan'),
                    oracle_coverage=float('nan'), candidates=len(eligible), gt_count=0,
                    precision_L=float('nan'), precision_L2=float('nan'), precision_L5=float('nan'))
    ranked = sorted(eligible, key=lambda pair: (-eligible[pair], rank_key(pair)))
    labels = np.array([pair in native for pair in ranked], dtype=np.float64)
    cumulative = labels.cumsum()
    groups = defaultdict(lambda: [0, 0])
    for pair, score in eligible.items():
        groups[score][0] += 1
        groups[score][1] += int(pair in native)
    tp = predicted = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        n, positives = groups[score]
        tp += positives
        predicted += n
        ap += positives / len(native) * tp / predicted
    if universe > predicted:
        ap += (len(native) - tp) / len(native) * len(native) / universe

    def precision(k: int) -> float:
        return float(cumulative[min(k, len(cumulative)) - 1] / k) if len(cumulative) else 0.0

    return dict(r_precision=precision(len(native)), average_precision=ap,
        oracle_coverage=tp / len(native), candidates=len(eligible), gt_count=len(native),
        precision_L=precision(length), precision_L2=precision(max(1, length // 2)),
        precision_L5=precision(max(1, length // 5)))


def bootstrap(values: list[float], groups: list[str], seed: int = 368) -> tuple[float, float]:
    """Resample related groups while retaining the per-structure mean estimand."""
    bins = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        if np.isfinite(value):
            bins[group].append(value)
    total = np.array([sum(rows) for rows in bins.values()])
    count = np.array([len(rows) for rows in bins.values()])
    if not len(total):
        return float('nan'), float('nan')
    indices = np.random.default_rng(seed).integers(len(total), size=(2000, len(total)))
    means = total[indices].sum(-1) / count[indices].sum(-1)
    return tuple(np.quantile(means, [.025, .975]).tolist())
