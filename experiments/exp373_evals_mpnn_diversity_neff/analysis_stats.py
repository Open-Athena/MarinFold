"""Protein-level diversity statistics and frequency-preserving randomization."""

import numpy as np
from scipy import sparse


def incidence(sets: list[set[int]], universe_size: int) -> sparse.csr_matrix:
    """Build a binary sample-by-pair matrix without dense candidate storage."""
    rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
    cols = np.array([v for s in sets for v in sorted(s)], dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(len(cols), dtype=np.float64), (rows, cols)),
        shape=(len(sets), universe_size),
    )


def diversity(sets: list[set[int]], universe_size: int) -> dict[str, float]:
    """Return all-pairs Jaccard distance, entropy and overlap-spread statistics."""
    matrix = incidence(sets, universe_size)
    counts = np.array([len(s) for s in sets])
    intersection = (matrix @ matrix.T).toarray()
    union = counts[:, None] + counts[None, :] - intersection
    distance = np.divide(
        union - intersection, union, out=np.zeros_like(union), where=union > 0
    )
    ix = np.triu_indices(len(sets), 1)
    _, occurrences = np.unique(matrix.indices, return_counts=True)
    frequencies = occurrences / len(sets)
    p = frequencies[(frequencies > 0) & (frequencies < 1)]
    entropy = float((-p * np.log2(p) - (1 - p) * np.log2(1 - p)).sum())
    return {
        "jaccard": float(distance[ix].mean()),
        "jaccard_sd": float(distance[ix].std()),
        "overlap_sd": float(intersection[ix].std()),
        "entropy_sum": entropy,
        "mean_contacts": float(counts.mean()),
        "empty_fraction": float((counts == 0).mean()),
        "unique_contacts": int((frequencies > 0).sum()),
    }


def thin_matched(
    left: list[set[int]], right: list[set[int]], seed: int
) -> tuple[list[set[int]], list[set[int]]]:
    """Match both contact counts at every paired sample index by random thinning."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for x, y in zip(left, right, strict=True):
        n = min(len(x), len(y))
        a.append(set(rng.choice(sorted(x), n, replace=False).tolist()))
        b.append(set(rng.choice(sorted(y), n, replace=False).tolist()))
    return a, b


def curveball(sets: list[set[int]], seed: int, trades: int) -> list[set[int]]:
    """Randomize binary rows with exact row and column marginal preservation."""
    rng = np.random.default_rng(seed)
    result = [set(row) for row in sets]
    for _ in range(trades):
        a, b = rng.choice(len(result), 2, replace=False)
        common = result[a] & result[b]
        unique = sorted(result[a] ^ result[b])
        if not unique:
            continue
        n = len(result[a]) - len(common)
        rng.shuffle(unique)
        result[a] = common | set(unique[:n])
        result[b] = common | set(unique[n:])
    return result


def bootstrap_mean(
    values: np.ndarray, draws: int = 10000, seed: int = 373
) -> tuple[float, float, float, int]:
    """Mean and percentile CI across finite independent protein observations."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(values[0]), float("nan"), float("nan"), 1
    rng = np.random.default_rng(seed)
    means = values[rng.integers(n, size=(draws, n))].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        n,
    )


def bootstrap_slope(
    y: np.ndarray, x: np.ndarray, covariates: np.ndarray | None = None
) -> tuple[float, float, float, int]:
    """Bootstrap paired-difference OLS slopes, retaining the protein pairing."""
    design = np.column_stack(
        [np.ones(len(x)), x] + ([] if covariates is None else [covariates])
    )
    keep = np.isfinite(y) & np.isfinite(design).all(axis=1)
    y, design = y[keep], design[keep]
    n = len(y)
    if n <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan"), float("nan"), float("nan"), n
    point = np.linalg.lstsq(design, y, rcond=None)[0][1]
    rng = np.random.default_rng(373)
    slopes = []
    for _ in range(10000):
        idx = rng.integers(n, size=n)
        if np.linalg.matrix_rank(design[idx]) == design.shape[1]:
            slopes.append(np.linalg.lstsq(design[idx], y[idx], rcond=None)[0][1])
    return (
        float(point),
        float(np.quantile(slopes, 0.025)),
        float(np.quantile(slopes, 0.975)),
        n,
    )
