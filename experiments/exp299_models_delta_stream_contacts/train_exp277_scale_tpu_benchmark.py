# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exp277-scale delta-stream V2 training on one TPU topology."""

import dataclasses
import logging
import os

from levanter.checkpoint import CheckpointerConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from train_exp277_scale_tpu_smoke import tpu_pod_config

DEFAULT_STEPS = 50


def benchmark_config(
    tpu_type: str,
    zone: str,
    steps: int = DEFAULT_STEPS,
) -> TrainLmOnPodConfig:
    """Build a short from-scratch throughput benchmark."""
    if steps < 10:
        raise ValueError("benchmark must run at least ten updates")
    config = tpu_pod_config(tpu_type, zone)
    trainer = config.train_config.trainer
    tracker = dataclasses.replace(
        trainer.tracker,
        group="exp299-delta-v2-tpu-topology-benchmark",
        tags=[
            "exp299",
            "delta-v2",
            "exp277-corpus",
            "tpu-topology-benchmark",
            "qwen3",
            "1_5b",
            f"tpu={tpu_type}",
            f"steps={steps}",
        ],
        save_code=False,
    )
    trainer = dataclasses.replace(
        trainer,
        tracker=tracker,
        num_train_steps=steps,
        steps_per_eval=steps,
        checkpointer=CheckpointerConfig(save_interval=None, keep=[]),
    )
    return dataclasses.replace(
        config,
        train_config=dataclasses.replace(config.train_config, trainer=trainer),
    )


def main() -> None:
    """Run the benchmark directly in the Iris-managed TPU task."""
    tpu_type = os.environ["EXP299_TPU_TYPE"]
    zone = os.environ["EXP299_TPU_ZONE"]
    steps = int(os.environ.get("EXP299_BENCHMARK_STEPS", str(DEFAULT_STEPS)))
    run_levanter_train_lm(benchmark_config(tpu_type, zone, steps))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
