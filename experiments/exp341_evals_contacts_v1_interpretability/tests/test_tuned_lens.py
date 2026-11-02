# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tuned-lens tests on the tiny random Qwen3 from conftest."""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import b1_tuned_lens as lens  # noqa: E402
import harness  # noqa: E402


def test_query_residuals_keep_one_row_per_query_without_changing_the_readout(backend, protein):
    clean = harness.pcontact_per_seed(backend, protein, seeds=2)[0]
    residuals, fwd, pos_ids = lens.query_pass(backend, protein, seed=0)
    np.testing.assert_array_equal(lens.pcontact_from_fwd(fwd), clean)
    assert len(residuals) == 3  # embedding + 2 layers
    assert all(r.shape == (protein.length + 1, 32) for r in residuals)
    assert len(pos_ids) == protein.length


def test_final_boundary_of_both_lenses_is_the_production_readout(backend, protein):
    residuals, fwd, pos_ids = lens.query_pass(backend, protein, seed=0)
    weights = torch.zeros(2, 33, 32)  # tuned maps for boundaries 0 and 1; boundary 2 is the identity
    pcontact = lens.lens_pcontact(backend.model, residuals, weights, pos_ids)
    assert pcontact.shape == (2, 3, protein.length, protein.length)
    np.testing.assert_allclose(pcontact[0, -1], lens.pcontact_from_fwd(fwd), rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(pcontact[1, -1], pcontact[0, -1])
    # An all-zero tuned map gives the same distribution at every boundary; the logit lens does not.
    np.testing.assert_allclose(pcontact[1, 0], pcontact[1, 1])
    assert not np.allclose(pcontact[0, 0], pcontact[0, 1])


def test_ridge_recovers_an_affine_map_and_mse_matches_its_direct_value():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(500, 6)))
    x = torch.cat([x, torch.ones(500, 1, dtype=x.dtype)], dim=1)
    true = torch.from_numpy(rng.normal(size=(7, 6)))
    y = x @ true + 0.01 * torch.from_numpy(rng.normal(size=(500, 6)))
    xtx, xty, yty = x.T @ x, x.T @ y, float((y ** 2).sum())
    weights = lens.ridge_solve(xtx, xty, 1e-8)
    torch.testing.assert_close(weights, true, atol=5e-3, rtol=0)
    direct = float(((x @ weights - y) ** 2).mean())
    assert abs(lens.mse(weights, xtx, xty, yty, 500) - direct) < 1e-10
    # A strong penalty shrinks the feature weights but leaves the bias row free.
    shrunk = lens.ridge_solve(xtx, xty, 1e3)
    assert shrunk[:-1].abs().max() < 0.05 * true[:-1].abs().max()
    torch.testing.assert_close(shrunk[-1], y.mean(0), atol=0.05, rtol=0)


def test_fit_lens_returns_one_map_per_boundary_and_a_fit_table(backend, protein):
    args = SimpleNamespace(seeds=2, device="cpu")
    weights, fit = lens.fit_lens(backend, [protein] * 5, args, hidden=32, n_boundaries=2)
    assert weights.shape == (2, 33, 32) and weights.dtype == torch.float32
    assert fit.chosen.sum() == 2
    assert set(fit.lens) == {"logit", "tuned"}
    # The chosen map fits the held-out queries at least as well as every other strength.
    tuned = fit[fit.lens == "tuned"]
    for _, group in tuned.groupby("boundary"):
        assert group[group.chosen].heldout_mse.iloc[0] == group.heldout_mse.min()
