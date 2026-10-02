# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the exp277-scale delta-stream V2 training smoke on a Marin TPU."""

import dataclasses
import logging
import os

from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from train_exp277_scale_smoke import GLOBAL_BATCH_SIZE, pod_config

DEFAULT_TPU_TYPE = "v5p-128"
DEFAULT_TPU_ZONE = "us-east5-a"


def tpu_resources(tpu_type: str, zone: str) -> ResourceConfig:
    """Construct the TPU slice used by the smoke."""
    return ResourceConfig.with_tpu(
        tpu_type,
        slice_count=1,
        cpu=200,
        ram="400GB",
        disk="100GB",
        zone=zone,
    )


def tpu_pod_config(
    tpu_type: str = DEFAULT_TPU_TYPE,
    zone: str = DEFAULT_TPU_ZONE,
) -> TrainLmOnPodConfig:
    """Build the smoke config with TPU attention and per-chip microbatching."""
    try:
        tpu_cores = int(tpu_type.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"TPU type must end in its core count: {tpu_type}") from error
    chips = tpu_cores // 2
    if tpu_cores % 2 or GLOBAL_BATCH_SIZE % chips:
        raise ValueError(
            f"global batch {GLOBAL_BATCH_SIZE} must divide evenly across {chips} chips"
        )

    resources = tpu_resources(tpu_type, zone)
    config = pod_config(resources)
    trainer = dataclasses.replace(
        config.train_config.trainer,
        per_device_parallelism=GLOBAL_BATCH_SIZE // chips,
        per_device_eval_parallelism=GLOBAL_BATCH_SIZE // chips,
    )
    model = dataclasses.replace(
        config.train_config.model,
        attn_backend=AttentionBackend.SPLASH,
    )
    return dataclasses.replace(
        config,
        train_config=dataclasses.replace(
            config.train_config,
            trainer=trainer,
            model=model,
        ),
    )


def main() -> None:
    """Run training directly in the Iris-managed TPU task."""
    tpu_type = os.environ.get("EXP299_TPU_TYPE", DEFAULT_TPU_TYPE)
    zone = os.environ.get("EXP299_TPU_ZONE", DEFAULT_TPU_ZONE)
    run_levanter_train_lm(tpu_pod_config(tpu_type, zone))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
