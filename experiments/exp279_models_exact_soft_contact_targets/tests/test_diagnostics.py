# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation denominators and the CE = entropy + KL readout."""

from dataclasses import replace

import haliax as hax
import jax
import numpy as np
from test_loss import example_and_oracle
from test_trainer import tiny_config

from experiments.exp279_models_exact_soft_contact_targets.diagnostics import (
    diagnostic_sums,
)


def test_same_validation_readout_independent_of_training_arm(vocab):
    example, q = example_and_oracle(vocab)
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(279)
    )
    sums = {k: float(v) for k, v in diagnostic_sums(model, example).items()}
    control = replace(
        model,
        transformer=replace(
            model.transformer, config=replace(model.config, soft_targets=False)
        ),
    )
    assert sums == {k: float(v) for k, v in diagnostic_sums(control, example).items()}
    mask = np.asarray(example.loss_weight.array)
    assert sums["scored_tokens"] == mask.sum()
    expected_entropy = (-(q * np.log(np.maximum(q, 1e-30))).sum(-1) * mask).sum()
    np.testing.assert_allclose(sums["entropy_sum"], expected_entropy, atol=2e-6)
    np.testing.assert_allclose(
        sums["soft_ce_sum"], sums["entropy_sum"] + sums["kl_sum"], atol=2e-5
    )
    assert sums["kl_sum"] >= 0
