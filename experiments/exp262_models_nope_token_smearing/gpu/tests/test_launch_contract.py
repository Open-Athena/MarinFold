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
from exp262_train_cw import ARMS, TOKEN_FRACTION, model_config

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
    config = model_config(ARMS["a-rope"])
    assert config.smear_width == 0
    assert config.rope == MODEL_CONFIG.rope
    assert dataclasses.replace(config, smear_width=0).uses_rope


def test_the_2x2_is_actually_a_2x2():
    grid = {(arm.use_rope, arm.smear_width > 0) for arm in ARMS.values()}
    assert grid == {(True, False), (True, True), (False, True), (False, False)}
    for arm in ARMS.values():
        config = model_config(arm)
        assert isinstance(config.rope, NoRotaryEmbeddingsConfig) is not arm.use_rope
        assert (config.smear_width > 0) is (arm.smear_width > 0)


def test_the_smear_is_a_rounding_error_in_parameter_count():
    """If the smear cost real parameters the comparison would not be clean."""
    control = model_config(ARMS["a-rope"]).total_trainable_params(2845)
    smeared = model_config(ARMS["b-rope-smear"]).total_trainable_params(2845)
    assert 0 < smeared - control < 10_000
    assert (smeared - control) / control < 1e-5


def test_budget_is_a_screen_not_a_production_run():
    steps = int(NUM_TRAIN_STEPS * TOKEN_FRACTION)
    assert steps < NUM_TRAIN_STEPS
    tokens = steps * GLOBAL_BATCH_SIZE * SEQ_LEN
    assert 10e9 < tokens < 20e9, f"expected a ~15B-token screen, got {tokens / 1e9:.1f}B"
