# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the exp277-scale V2 checkpoint with a selected cooldown length."""

import dataclasses
import logging
import os
from datetime import timedelta

from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from train_exp277_scale_full import GLOBAL_BATCH_SIZE, TRAIN_STEPS, pod_config
from train_exp277_scale_tpu_smoke import tpu_chip_count, tpu_resources

SOURCE_STEP = 126_294
DEFAULT_SOURCE_CHECKPOINT = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/checkpoints/"
    "delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03/checkpoints/step-126294"
)
SUPPORTED_DECAY_FRACTIONS = (0.2, 0.3, 0.4)


def cooldown_config(
    *,
    tpu_type: str,
    zone: str,
    decay_fraction: float,
    source_checkpoint: str,
    end_step: int = TRAIN_STEPS,
) -> TrainLmOnPodConfig:
    """Build one exact full-state cooldown continuation arm."""
    if decay_fraction not in SUPPORTED_DECAY_FRACTIONS:
        raise ValueError(
            f"decay_fraction must be one of {SUPPORTED_DECAY_FRACTIONS}, got {decay_fraction}"
        )
    if not SOURCE_STEP < end_step <= TRAIN_STEPS:
        raise ValueError(f"end_step must be in ({SOURCE_STEP}, {TRAIN_STEPS}], got {end_step}")

    chips = tpu_chip_count(tpu_type)
    if GLOBAL_BATCH_SIZE % chips:
        raise ValueError(
            f"global batch {GLOBAL_BATCH_SIZE} must divide evenly across {chips} chips"
        )
    resources = tpu_resources(tpu_type, zone)
    config = pod_config(resources, nodes=2)
    trainer = config.train_config.trainer
    tracker = dataclasses.replace(
        trainer.tracker,
        group="exp299-delta-v2-exp277-cooldown-sweep",
        tags=[
            "exp299",
            "delta-v2",
            "exp277-corpus",
            "cooldown-sweep",
            "full-state-continuation",
            "qwen3",
            "1_5b",
            f"source_step={SOURCE_STEP}",
            f"decay={decay_fraction:g}",
            f"tpu={tpu_type}",
        ],
        save_code=False,
    )
    smoke = end_step < TRAIN_STEPS
    trainer = dataclasses.replace(
        trainer,
        tracker=tracker,
        initialize_from=source_checkpoint,
        num_train_steps=end_step,
        steps_per_eval=max(1, end_step - SOURCE_STEP) if smoke else trainer.steps_per_eval,
        max_eval_batches=2 if smoke else None,
        per_device_parallelism=GLOBAL_BATCH_SIZE // chips,
        per_device_eval_parallelism=GLOBAL_BATCH_SIZE // chips,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(minutes=15),
            keep=[] if smoke else [{"every": TRAIN_STEPS // 10}],
        ),
    )
    train_config = dataclasses.replace(
        config.train_config,
        trainer=trainer,
        model=dataclasses.replace(
            config.train_config.model,
            attn_backend=AttentionBackend.SPLASH,
        ),
        optimizer=dataclasses.replace(
            config.train_config.optimizer,
            decay=decay_fraction,
        ),
    )
    return dataclasses.replace(config, train_config=train_config)


def main() -> None:
    """Run one cooldown arm directly in the Iris-managed TPU task."""
    tpu_type = os.environ["EXP299_TPU_TYPE"]
    zone = os.environ["EXP299_TPU_ZONE"]
    decay_fraction = float(os.environ["EXP299_DECAY_FRACTION"])
    source_checkpoint = os.environ.get(
        "EXP299_SOURCE_CHECKPOINT",
        DEFAULT_SOURCE_CHECKPOINT,
    )
    end_step = int(os.environ.get("EXP299_END_STEP", str(TRAIN_STEPS)))
    run_levanter_train_lm(
        cooldown_config(
            tpu_type=tpu_type,
            zone=zone,
            decay_fraction=decay_fraction,
            source_checkpoint=source_checkpoint,
            end_step=end_step,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
