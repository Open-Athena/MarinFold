# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the exp277-scale V2 TPU benchmark and cooldown sweep."""

import pytest
from levanter.layers.attention import AttentionBackend
from train_exp277_scale_full import TRAIN_STEPS
from train_exp277_scale_tpu_benchmark import benchmark_config
from train_exp277_scale_tpu_cooldown import SOURCE_STEP, cooldown_config

SOURCE = "gs://test/checkpoints/step-126294"


def test_topology_benchmark_changes_only_smoke_duration() -> None:
    config = benchmark_config("v6e-32", "us-east5-b", 50)
    assert config.resources.device.variant == "v6e-32"
    assert config.train_config.trainer.num_train_steps == 50
    assert config.train_config.trainer.per_device_parallelism == 4
    assert config.train_config.trainer.checkpointer.save_interval is None
    assert config.train_config.model.attn_backend == AttentionBackend.SPLASH


@pytest.mark.parametrize(
    ("decay_fraction", "decay_start"),
    ((0.2, 168_394), (0.3, 147_344), (0.4, 126_295)),
)
def test_cooldown_arms_restore_identical_state_and_change_only_schedule(
    decay_fraction: float,
    decay_start: int,
) -> None:
    pod = cooldown_config(
        tpu_type="v6e-64",
        zone="us-east5-b",
        decay_fraction=decay_fraction,
        source_checkpoint=SOURCE,
    )
    config = pod.train_config
    assert config.trainer.initialize_from == SOURCE
    assert config.trainer.num_train_steps == TRAIN_STEPS == 210_492
    assert config.trainer.per_device_parallelism == 2
    assert config.trainer.max_eval_batches is None
    assert config.model.attn_backend == AttentionBackend.SPLASH
    assert config.optimizer.decay == decay_fraction

    schedule = config.optimizer.lr_scheduler(TRAIN_STEPS)
    assert float(schedule(SOURCE_STEP)) == pytest.approx(1e-3)
    assert float(schedule(decay_start - 1)) == pytest.approx(1e-3)
    assert float(schedule(TRAIN_STEPS - 1)) == pytest.approx(1e-4, rel=3e-4)


def test_restore_smoke_runs_only_past_source_step() -> None:
    pod = cooldown_config(
        tpu_type="v6e-8",
        zone="us-east5-b",
        decay_fraction=0.4,
        source_checkpoint=SOURCE,
        end_step=SOURCE_STEP + 3,
    )
    trainer = pod.train_config.trainer
    assert trainer.num_train_steps == SOURCE_STEP + 3
    assert trainer.max_eval_batches == 2
    assert trainer.checkpointer.keep == []


def test_unknown_cooldown_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="decay_fraction"):
        cooldown_config(
            tpu_type="v6e-8",
            zone="us-east5-b",
            decay_fraction=0.25,
            source_checkpoint=SOURCE,
        )
