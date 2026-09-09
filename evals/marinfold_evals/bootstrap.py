"""Poisson bootstrap helpers for weighted eval-loss estimates."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    """Summary of a Poisson bootstrap estimate for a weighted mean."""

    estimate: float
    stderr: float
    n_units: int
    n_bootstrap: int
    seed: int


def poisson_bootstrap_weighted_mean(
    numerators: Sequence[float] | np.ndarray,
    denominators: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    batch_size: int = 1_024,
) -> BootstrapSummary:
    """Estimate stderr of ``sum(numerators) / sum(denominators)``.

    This is the right shape for document-level eval loss: pass per-document
    negative-log-likelihood sums as ``numerators`` and per-document token counts
    as ``denominators``. The point estimate remains the token-weighted eval loss,
    while the bootstrap resamples documents with independent Poisson(1) weights.

    Args:
        numerators: Per-unit numerator sums.
        denominators: Per-unit positive weights/counts.
        n_bootstrap: Number of bootstrap replicates.
        seed: Random seed for the Poisson draws.
        batch_size: Replicates to draw per batch to bound memory use.

    Returns:
        BootstrapSummary containing the weighted mean and bootstrap stderr.

    Raises:
        ValueError: If arrays have incompatible shapes, no units, non-finite
            values, non-positive denominators, or invalid bootstrap settings.
    """
    nums = np.asarray(numerators, dtype=np.float64)
    dens = np.asarray(denominators, dtype=np.float64)
    if nums.shape != dens.shape:
        raise ValueError(f"numerators and denominators must have the same shape, got {nums.shape} and {dens.shape}")
    if nums.ndim != 1:
        raise ValueError(f"numerators and denominators must be one-dimensional, got ndim={nums.ndim}")
    if nums.size == 0:
        raise ValueError("at least one unit is required")
    if not np.all(np.isfinite(nums)) or not np.all(np.isfinite(dens)):
        raise ValueError("numerators and denominators must be finite")
    if np.any(dens <= 0):
        raise ValueError("denominators must be strictly positive")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    total_den = float(np.sum(dens))
    estimate = float(np.sum(nums) / total_den)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    filled = 0
    while filled < n_bootstrap:
        this_batch = min(batch_size, n_bootstrap - filled)
        weights = rng.poisson(1.0, size=(this_batch, nums.size)).astype(np.float64)
        sample_dens = weights @ dens
        while np.any(sample_dens == 0):
            # With many validation documents this is astronomically unlikely,
            # but small unit tests can draw an all-zero replicate. Redraw only
            # those rows so every bootstrap sample has a defined ratio.
            zero_rows = sample_dens == 0
            weights[zero_rows] = rng.poisson(1.0, size=(int(np.sum(zero_rows)), nums.size))
            sample_dens = weights @ dens
        sample_nums = weights @ nums
        samples[filled : filled + this_batch] = sample_nums / sample_dens
        filled += this_batch

    return BootstrapSummary(
        estimate=estimate,
        stderr=float(np.std(samples, ddof=1)),
        n_units=int(nums.size),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )
