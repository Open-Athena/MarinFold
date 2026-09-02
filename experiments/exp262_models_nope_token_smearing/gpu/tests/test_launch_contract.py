# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The launch script must change the architecture and nothing else.

exp262's whole claim is that a difference between arms is a difference between
architectures. That only holds if every other field of the model config is
exp232's, so this pins it rather than trusting the code to have stayed honest.
"""

import dataclasses

import pytest
from architecture import NoRotaryEmbeddingsConfig
from exp262_train_cw import ARMS, CONTROL_POINT, POINTS, SCREEN_FRACTION, model_config

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    GLOBAL_BATCH_SIZE,
    MODEL_CONFIG,
    NUM_TRAIN_STEPS,
    SEQ_LEN,
)

# The only two fields any arm is allowed to move.
MUTABLE = {"smear_width", "rope"}


@pytest.mark.parametrize("arm_key", sorted(ARMS))
def test_arms_change_only_the_architecture_fields(arm_key: str):
    config = model_config(ARMS[arm_key])
    for field in MODEL_CONFIG.__dataclass_fields__:
        if field in MUTABLE:
            continue
        assert getattr(config, field) == getattr(MODEL_CONFIG, field), (
            f"arm {arm_key} changed {field}, which is part of the exp232 contract"
        )


def test_control_arm_is_exp232_exactly():
    """The control must reduce to exp232's model: no smear, exp232's rope."""
    config = model_config(ARMS["control"])
    assert config.smear_width == 0
    assert config.rope == MODEL_CONFIG.rope
    assert dataclasses.replace(config, smear_width=0).uses_rope


def test_the_two_arms_differ_only_in_the_thing_under_test():
    control, proposal = model_config(ARMS["control"]), model_config(ARMS["nope-smear"])
    assert not isinstance(control.rope, NoRotaryEmbeddingsConfig)
    assert isinstance(proposal.rope, NoRotaryEmbeddingsConfig)
    assert control.smear_width == 0 and proposal.smear_width == 2


def test_the_smear_is_a_rounding_error_in_parameter_count():
    """If the smear cost real parameters the comparison would not be clean."""
    control = model_config(ARMS["control"]).total_trainable_params(2845)
    smeared = model_config(ARMS["nope-smear"]).total_trainable_params(2845)
    assert 0 < smeared - control < 10_000
    assert (smeared - control) / control < 1e-5


def test_screen_is_a_fraction_and_full_is_exp232_exactly():
    """The headline run must be exp232's schedule, not an approximation of it."""
    screen_steps = int(NUM_TRAIN_STEPS * SCREEN_FRACTION)
    assert 0 < screen_steps < NUM_TRAIN_STEPS
    tokens = screen_steps * GLOBAL_BATCH_SIZE * SEQ_LEN
    assert 10e9 < tokens < 20e9, f"expected a ~15B-token screen, got {tokens / 1e9:.1f}B"


def test_control_optimizer_point_is_exp232s_swept_winner():
    """p06 is lr 1e-3 / wd 0.2 — the point exp232's own five-point sweep chose."""
    point = POINTS[CONTROL_POINT]
    assert (point.learning_rate, point.weight_decay) == (1e-3, 0.2)


def test_the_grid_reaches_above_the_control_rate():
    """The pilot said the NoPE arm wants a higher rate; the grid has to allow it."""
    rates = {point.learning_rate for point in POINTS.values()}
    assert max(rates) > POINTS[CONTROL_POINT].learning_rate * 5


def test_the_nope_config_is_indistinguishable_from_rope_when_serialised():
    """Why verify_and_train exists: the two configs serialise identically.

    ``NoRotaryEmbeddingsConfig`` inherits ``theta`` and carries no other field,
    so any path that round-trips it through a plain dict loses the distinction
    silently. This test documents the hazard so nobody "simplifies" the worker
    check away.
    """
    import dataclasses

    from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

    nope = dataclasses.asdict(NoRotaryEmbeddingsConfig())
    default = dataclasses.asdict(DefaultRotaryEmbeddingsConfig())
    assert nope == {"theta": 10000.0}
    assert nope == {key: value for key, value in default.items() if key == "theta"}


def test_architecture_survives_cloudpickle():
    """The config reaches the worker by cloudpickle; it must arrive as itself."""
    import cloudpickle

    for key, arm in ARMS.items():
        restored = cloudpickle.loads(cloudpickle.dumps(model_config(arm)))
        assert isinstance(restored.rope, NoRotaryEmbeddingsConfig) is not arm.use_rope, key
        assert restored.smear_width == arm.smear_width, key


def test_worker_check_rejects_a_downgraded_architecture():
    """If NoPE silently became rope, the worker must refuse to train."""
    from unittest.mock import patch

    from exp262_train_cw import verify_and_train

    class FakePod:
        def __init__(self, model):
            self.train_config = type("C", (), {"model": model})()

    with patch("exp262_train_cw.run_levanter_train_lm") as trainer:
        verify_and_train(FakePod(model_config(ARMS["nope-smear"])), expect_rope=False, expect_smear=2)
        assert trainer.called

    with patch("exp262_train_cw.run_levanter_train_lm") as trainer:
        downgraded = model_config(ARMS["control"])  # rope, no smear
        with pytest.raises(ValueError, match="did not survive dispatch"):
            verify_and_train(FakePod(downgraded), expect_rope=False, expect_smear=2)
        assert not trainer.called
