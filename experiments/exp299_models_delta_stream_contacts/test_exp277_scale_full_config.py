# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the exp277-scale finite V2 training configuration."""

from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent

from finite_delta_data import FULL_CORPUS, FiniteOneEpochDataConfig
from train_exp277_scale_full import (
    EXPECTED_PACKED_EXAMPLES,
    TRAIN_CORPORA,
    TRAIN_STEPS,
    data_config,
    pod_config,
    resources,
)


def test_full_data_config_is_finite_and_document_blocked() -> None:
    config = data_config()
    assert isinstance(config, FiniteOneEpochDataConfig)
    assert config.expected_packed_examples == EXPECTED_PACKED_EXAMPLES == 26_942_937
    assert config.block_cross_document_attention
    assert config.train_weights == {FULL_CORPUS: 1.0, "delta-v2/validation": 0.0}

    full = config.components[FULL_CORPUS]
    assert isinstance(full, ConcatDatasetComponent)
    assert tuple(full.children) == TRAIN_CORPORA
    assert all(child.flat_cache and child.pack is True for child in full.children.values())

    validation = config.components["delta-v2/validation"]
    assert isinstance(validation, DatasetComponent)
    assert not validation.flat_cache
    assert validation.split == "validation"
    assert validation.pack is True


def test_full_optimizer_and_parallelism_match_exp277_recipe() -> None:
    config = pod_config(resources(4), nodes=4).train_config
    assert config.trainer.num_train_steps == TRAIN_STEPS == 210_492
    assert config.trainer.train_batch_size == 128
    assert config.trainer.per_device_parallelism == 8
    assert config.train_seq_len == 8192
    assert config.data_seed == 0
    assert config.hf_save_steps is None
    assert config.optimizer.learning_rate == 1e-3
    assert config.optimizer.weight_decay == 0.2
    assert config.optimizer.warmup == 0.1
    assert config.optimizer.decay == 0.2
    assert config.optimizer.lr_schedule == "linear"
    assert config.optimizer.min_lr_ratio == 0.1

    fallback = pod_config(resources(2), nodes=2).train_config
    assert fallback.trainer.per_device_parallelism == 8
