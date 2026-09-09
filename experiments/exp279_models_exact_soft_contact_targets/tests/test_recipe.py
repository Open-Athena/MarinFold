# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration identity and schedule boundaries are experiment invariants."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.optim.config import AdamConfig
from levanter.optim.skipstep import SkipStepConfig
from levanter.tracker import current_tracker
from levanter.tracker.tracker import NoopTracker

from experiments.exp279_models_exact_soft_contact_targets.recipe import (
    PHASES,
    RecipeAdamConfig,
    learning_rate,
    optimizer_for_phase,
)
from experiments.exp279_models_exact_soft_contact_targets.train import build_config


def test_full_effective_schedule():
    original = AdamConfig(
        learning_rate=1e-3,
        warmup=0.1,
        decay=0.2,
        min_lr_ratio=0.1,
        lr_schedule="linear",
    ).lr_scheduler(145200)
    steps = jnp.array([0, 1, 14519, 14520, 116159, 116160])
    np.testing.assert_allclose(learning_rate(steps), original(steps), rtol=1e-6)
    steps = jnp.array([116161, 145200, 217801, 333960, 333961, 344850, 344851, 363000])
    expected = np.array([1e-3] * 5 + [5e-5] * 3)
    np.testing.assert_allclose(learning_rate(steps), expected, rtol=1e-6)
    for phase in PHASES.values():
        np.testing.assert_array_equal(
            optimizer_for_phase(phase).lr_scheduler(phase.stop)(steps),
            learning_rate(steps),
        )


def test_skipped_update_still_advances_outer_lr_schedule():
    config = RecipeAdamConfig(
        skip_bad_steps=SkipStepConfig(rolling_interval_length=4, sigma_factor=0.5),
        weight_decay=0,
    )
    with current_tracker(NoopTracker()):
        optimizer = config.build(10)
        parameter = jnp.array(1.0)
        state = optimizer.init(parameter)
        for _ in range(2):
            _, state = optimizer.update(
                jnp.array(1.0), state, parameter, loss=jnp.array(1.0)
            )
        update, skipped = optimizer.update(
            jnp.array(100.0), state, parameter, loss=jnp.array(100.0)
        )
        assert float(update) == 0
        assert int(skipped.count) == 3
        for a, b in zip(
            jax.tree.leaves(skipped.inner_state.inner_opt_state),
            jax.tree.leaves(state.inner_state.inner_opt_state),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        _, next_state = optimizer.update(
            jnp.array(1.0), skipped, parameter, loss=jnp.array(1.0)
        )
        np.testing.assert_allclose(
            next_state.hyperparams["learning_rate"], learning_rate(3), rtol=1e-6
        )


def test_arms_change_only_loss_switch_and_run_identity(tmp_path):
    manifest = {
        "inputs": {
            name: {"cache_dir": f"/frozen/{name}"} for name in ("afdb", "esm", "val")
        }
    }
    ce = build_config(
        manifest,
        arm="ce",
        phase_name="base",
        run_name="exp279-ce-s0",
        output=str(tmp_path),
        resume=None,
    )
    soft = build_config(
        manifest,
        arm="soft",
        phase_name="base",
        run_name="exp279-soft-s0",
        output=str(tmp_path),
        resume=None,
    )
    assert ce.data == soft.data
    assert ce.optimizer == soft.optimizer
    a, b = asdict(ce.model), asdict(soft.model)
    assert a.pop("soft_targets") is False
    assert b.pop("soft_targets") is True
    assert a == b
    assert ce.trainer.mp == soft.trainer.mp
    assert ce.trainer.num_train_steps == 217801
    assert ce.model.use_qk_norm and ce.model.hidden_dim == 2048
    assert ce.data.augmentation_num_train_steps == 145200
    assert ce.hf_generation_eos_token_ids == [1, 10]
    with pytest.raises(ValueError, match="full-state"):
        build_config(
            manifest,
            arm="soft",
            phase_name="final",
            run_name="exp279-soft-s0",
            output=str(tmp_path),
            resume=None,
        )
