import math

import numpy as np
import pytest

from marinfold_evals import poisson_bootstrap_weighted_mean


def test_poisson_bootstrap_weighted_mean_reports_token_weighted_estimate() -> None:
    summary = poisson_bootstrap_weighted_mean(
        np.array([10.0, 20.0, 45.0]),
        np.array([5.0, 10.0, 15.0]),
        n_bootstrap=200,
        seed=123,
        batch_size=17,
    )

    assert summary.estimate == pytest.approx(75.0 / 30.0)
    assert summary.n_units == 3
    assert summary.n_bootstrap == 200
    assert summary.seed == 123
    assert math.isfinite(summary.stderr)
    assert summary.stderr > 0.0


def test_poisson_bootstrap_weighted_mean_is_deterministic_for_seed() -> None:
    kwargs = dict(n_bootstrap=500, seed=7, batch_size=31)
    a = poisson_bootstrap_weighted_mean([1.0, 4.0, 9.0, 16.0], [1.0, 2.0, 3.0, 4.0], **kwargs)
    b = poisson_bootstrap_weighted_mean([1.0, 4.0, 9.0, 16.0], [1.0, 2.0, 3.0, 4.0], **kwargs)

    assert a == b


def test_poisson_bootstrap_weighted_mean_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        poisson_bootstrap_weighted_mean([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        poisson_bootstrap_weighted_mean([1.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="at least 2"):
        poisson_bootstrap_weighted_mean([1.0, 2.0], [1.0, 1.0], n_bootstrap=1)
