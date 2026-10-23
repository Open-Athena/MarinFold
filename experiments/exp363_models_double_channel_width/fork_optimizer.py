"""Importable LR-fork optimizer configuration for remote serialization."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from levanter.optim.config import AdamConfig, OptimizerConfig

from experiments.exp277_models_single_mpnn_pilot.config import EPOCH_TRAIN_STEPS


SOURCE_CHECKPOINT_STEP = 74_178
RESUME_STEP = SOURCE_CHECKPOINT_STEP + 1
TRANSITION_STEPS = 2_114
TARGET_LR = 0.000707
FINAL_COOLDOWN_START = 213_076


@OptimizerConfig.register_subclass("exp363-fork-adam")
@dataclass(frozen=True)
class ForkAdamConfig(AdamConfig):
    """Keep Adam's checkpoint structure, replacing only its scheduled LR.

    Optax's injected hyperparameters reevaluate the schedule at the restored
    schedule count before each update. Keeping AdamConfig.build intact preserves
    the first/second moments and both update counters from the source checkpoint.
    The source schedule uses its original absolute horizon, even in a short smoke.
    """

    source_resume_step: int = RESUME_STEP
    transition_steps: int = TRANSITION_STEPS
    target_learning_rate: float = TARGET_LR

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transition_steps <= 0:
            raise ValueError("LR transition must have positive duration")
        if not 0 < self.target_learning_rate < self.learning_rate:
            raise ValueError("Fork target LR must be positive and below the source peak")
        if not 26_634 <= self.source_resume_step < FINAL_COOLDOWN_START - self.transition_steps:
            raise ValueError("Fork transition must fit in the source plateau")

    def lr_scheduler(
        self, num_train_steps: int, override_lr: float | None = None,
    ) -> Callable[[jax.Array], jax.Array]:
        """Reduce the plateau smoothly, retaining the original cooldown timing."""
        base_lr = self.learning_rate if override_lr is None else override_lr
        source_schedule = super().lr_scheduler(EPOCH_TRAIN_STEPS, override_lr=base_lr)
        target_ratio = self.target_learning_rate / self.learning_rate

        def schedule(step: jax.Array) -> jax.Array:
            progress = jnp.clip((step - self.source_resume_step) / self.transition_steps, 0, 1)
            return source_schedule(step) * (1 - (1 - target_ratio) * progress)

        return schedule
