# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolved exp232 m2/p06 recipe through checkpoint step-363000.

The restart at 217801 adds skip-step bookkeeping; the restart at 333961
changes the data seed and lowers LR. These are recipe phases, not fresh model
initializations. Failed/replayed source attempts are not extra training data.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import optax
from levanter.optim.config import AdamConfig, OptimizerConfig

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AFDB_DOCUMENTS,
    AFDB_TOKENS,
    ESM_DOCUMENTS,
    ESM_TOKENS,
)

TOKENIZER = (
    "eczech/contacts-v1-tokenizer-5d68a24a899f@80f4e411b957641e8c350b07bbe1e2c832697518"
)
REFERENCE_SHA = "7a18dd5158e71b0db8d782be77cc4b9a6d59c8df"
CORPORA = {
    "afdb": ("contacts_v1_decontam/train", AFDB_DOCUMENTS, AFDB_TOKENS),
    "esm": ("contacts_v1_esm_atlas_decontam/train", ESM_DOCUMENTS, ESM_TOKENS),
    "val": ("contacts_v1/val", None, None),
}


@dataclass(frozen=True)
class Phase:
    start: int
    stop: int  # exclusive: checkpoint step-(stop-1) contains state.step == stop
    data_seed: int
    skip_bad_steps: bool


PHASES = {
    "base": Phase(0, 217801, 0, False),
    "recovery": Phase(217801, 333961, 0, True),
    "final": Phase(333961, 363001, 232, True),
}


def learning_rate(count):
    """Source schedule, evaluated at Optax's outer schedule count.

    Levanter injects scheduled hyperparameters OUTSIDE SkipStep. A rejected
    update freezes Adam moments/count but still advances this schedule, keeping
    LR aligned with the trainer step. Tests exercise an actual rejected update.
    """
    warm = optax.linear_schedule(0.0, 1e-3, 14520)(count)
    lowered = optax.linear_schedule(1e-3, 5e-5, 10889)(count - 333961)
    return jnp.where(count < 14520, warm, jnp.where(count < 333961, 1e-3, lowered))


@OptimizerConfig.register_subclass("exp279_adam")
@dataclass(frozen=True)
class RecipeAdamConfig(AdamConfig):
    """Fix the schedule independently of a phase's stop or a pilot's length."""

    def lr_scheduler(self, num_train_steps, override_lr=None):
        if override_lr is not None:
            raise ValueError("The paired experiment fixes the reference learning rate")
        return learning_rate


def optimizer_for_phase(phase: Phase) -> RecipeAdamConfig:
    return RecipeAdamConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.2,
        max_grad_norm=1.0,
        min_lr_ratio=0.0,
        warmup=0.0,
        decay=1.0,
        skip_bad_steps=phase.skip_bad_steps,
    )
