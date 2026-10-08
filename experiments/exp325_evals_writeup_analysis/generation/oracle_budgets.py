"""Frozen random positive-contact budgets for the exp325 structural diagnostic."""

import hashlib

import numpy as np

BUDGETS = ("random_5", "random_10", "random_L5", "random_L2")
REPLICATES = 2
CONTROLS = ("top_0", "positive_all", "oracle")


def requested_count(arm: str, length: int) -> int:
    """Use the frozen FoldBench input sequence length for relative budgets."""
    return {"random_5": 5, "random_10": 10,
            "random_L5": max(1, length // 5), "random_L2": max(1, length // 2)}[arm]


def map_keys() -> list[tuple[str, int]]:
    """Return the complete, prespecified per-protein map inventory."""
    return [(arm, replicate) for replicate in range(REPLICATES) for arm in BUDGETS] + [(arm, 0) for arm in CONTROLS]


def random_seed(stem: str, replicate: int) -> int:
    """Give each protein/replicate a stable seed independent of processing order."""
    value = hashlib.sha256(f"exp325-oracle-budget-v1:{stem}:{replicate}".encode()).digest()
    return int.from_bytes(value[:8], "little")


def build_maps(oracle: np.ndarray, stem: str, length: int, *,
               budgets: tuple[str, ...] = BUDGETS,
               controls: tuple[str, ...] = CONTROLS) -> dict[str, np.ndarray]:
    """Sample true contacts without replacement; all other pairs stay unknown.

    A single random permutation per protein/replicate supplies nested prefixes
    for all four budgets. Independent permutations provide the two replicates.
    Fixed-count arms fail rather than silently returning fewer than 5/10 pairs.
    Relative budgets cap at the available true-contact count, recorded separately
    from the requested count in the input inventory.
    """
    if oracle.ndim != 2 or oracle.shape[0] != oracle.shape[1] or not np.array_equal(oracle, oracle.T):
        raise ValueError(f"{stem}: oracle must be a symmetric square matrix")
    if not np.isin(oracle, [0, 1, 2]).all() or np.any(np.diag(oracle) != 0):
        raise ValueError(f"{stem}: invalid oracle contact states")
    pairs = np.column_stack(np.where(np.triu(oracle == 2, 1)))
    output = {}
    for replicate in range(REPLICATES):
        shuffled = np.random.default_rng(random_seed(stem, replicate)).permutation(pairs)
        for arm in budgets:
            k = requested_count(arm, length)
            if k > len(pairs) and arm in ("random_5", "random_10"):
                raise ValueError(f"{stem}: {arm} requests {k} contacts but only {len(pairs)} exist")
            k = min(k, len(pairs))
            chosen = shuffled[:k]
            state = np.zeros_like(oracle)
            state[chosen[:, 0], chosen[:, 1]] = 2
            state[chosen[:, 1], chosen[:, 0]] = 2
            output[f"{arm}-{replicate}"] = state
    if "top_0" in controls:
        output["top_0-0"] = np.zeros_like(oracle)
    if "positive_all" in controls:
        output["positive_all-0"] = np.where(oracle == 2, 2, 0).astype(oracle.dtype)
    if "oracle" in controls:
        output["oracle-0"] = oracle.copy()
    return output
