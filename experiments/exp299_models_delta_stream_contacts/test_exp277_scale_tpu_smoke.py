# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the exp277-scale V2 TPU smoke."""

import pytest
from levanter.layers.attention import AttentionBackend
from train_exp277_scale_tpu_smoke import tpu_pod_config, tpu_resources


def test_v5p128_smoke_uses_splash_and_two_examples_per_chip() -> None:
    resources = tpu_resources("v5p-128", "us-east5-a")
    assert resources.device.variant == "v5p-128"
    assert resources.zone == "us-east5-a"
    assert resources.replicas == 16

    config = tpu_pod_config().train_config
    assert config.model.attn_backend == AttentionBackend.SPLASH
    assert config.trainer.train_batch_size == 128
    assert config.trainer.per_device_parallelism == 2
    assert config.trainer.per_device_eval_parallelism == 2
    assert config.trainer.num_train_steps == 10
    assert config.train_seq_len == 8192


def test_v6e8_smoke_uses_eight_chips() -> None:
    resources = tpu_resources("v6e-8", "us-east5-b")
    assert resources.cpu == 160
    assert resources.replicas == 1
    assert tpu_pod_config("v6e-8", "us-east5-b").train_config.trainer.per_device_parallelism == 16


def test_tpu_shape_must_divide_global_batch() -> None:
    with pytest.raises(ValueError, match="must divide evenly"):
        tpu_pod_config("v5p-512", "us-east5-a")
