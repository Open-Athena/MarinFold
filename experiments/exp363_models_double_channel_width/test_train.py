"""Audit the actual architecture and the lowered single-epoch training plan."""

from dataclasses import fields

import pytest
from levanter.data.text.datasets import ConcatDatasetComponent
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import TrainLmOnPodConfig, apply_output_path

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
)
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    OneEpochDataConfig,
)
from experiments.exp363_models_double_channel_width.config import (
    BASELINE_MODEL_CONFIG,
    EXPECTED_PARAMETERS,
    MODEL_CONFIG,
    PREFIX,
    RUN_ID,
    TOKENIZER,
    VERSION,
)
from experiments.exp363_models_double_channel_width.count_parameters import count_parameters
from experiments.exp363_models_double_channel_width.train import build_run


def resolved(*, smoke: bool, nodes: int, smoke_tag: str = "") -> TrainLmOnPodConfig:
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_run(smoke=smoke, nodes=nodes, smoke_tag=smoke_tag)
        step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        return step.build_config(context)


def test_only_width_changes_and_actual_parameters_match() -> None:
    changed = {
        field.name for field in fields(MODEL_CONFIG)
        if getattr(MODEL_CONFIG, field.name) != getattr(BASELINE_MODEL_CONFIG, field.name)
    }
    assert changed == {"hidden_dim", "intermediate_dim"}
    assert MODEL_CONFIG.actual_head_size == 128
    assert count_parameters(MODEL_CONFIG) == EXPECTED_PARAMETERS
    assert count_parameters(BASELINE_MODEL_CONFIG) == 1_471_374_336


def test_production_preserves_full_epoch_after_marin_lowering() -> None:
    pod = resolved(smoke=False, nodes=16)
    config = apply_output_path(pod.train_config, pod.output_path)
    assert config.model == MODEL_CONFIG
    assert config.trainer.num_train_steps == EPOCH_TRAIN_STEPS
    assert config.trainer.train_batch_size * config.train_seq_len == 1_048_576
    assert config.trainer.per_device_parallelism == 1
    assert config.trainer.max_eval_batches is None
    assert config.trainer.seed == config.data_seed == 0
    assert config.optimizer.learning_rate == 1e-3
    assert config.optimizer.weight_decay == 0.2
    assert config.optimizer.warmup == 0.1
    assert config.optimizer.decay == 0.2
    assert config.optimizer.min_lr_ratio == 0.1
    assert config.optimizer.lr_schedule == "linear"
    assert config.initialize_from_checkpoint_path is None
    assert config.initialize_model_from_checkpoint_path is None
    assert isinstance(config.data, OneEpochDataConfig)
    assert config.data.expected_packed_examples == EPOCH_PACKED_EXAMPLES
    assert config.data.augmentation_num_train_steps == EPOCH_TRAIN_STEPS
    assert config.data.block_cross_document_attention
    assert config.data.tokenizer == TOKENIZER
    combined = config.data.components[FULL_CORPUS]
    assert isinstance(combined, ConcatDatasetComponent)
    assert set(combined.children) == {f"input/full-epoch/{c.name}" for c in CORPORA}
    assert all(c.pack for c in combined.children.values())
    assert config.data.train_weights == {FULL_CORPUS: 1.0, "input/validation": 0.0}
    assert config.data.components["input/validation"].pack
    assert config.trainer.checkpointer.base_path == f"{PREFIX}/runs/{RUN_ID}/checkpoints"
    assert config.trainer.checkpointer.keep == [{"every": EPOCH_TRAIN_STEPS // 10}]
    assert config.trainer.tracker.project == "MarinFold"
    assert not {"WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"}.intersection(pod.env_vars)


def test_smoke_uses_distinct_identity_and_bounded_validation() -> None:
    smoke = resolved(smoke=True, nodes=1, smoke_tag="-n1-a01")
    repeat = resolved(smoke=True, nodes=1, smoke_tag="-n1-a02")
    production = resolved(smoke=False, nodes=16)
    assert len({p.output_path for p in (smoke, repeat, production)}) == 3
    assert smoke.train_config.trainer.num_train_steps == 10
    assert smoke.train_config.trainer.max_eval_batches == 2
    assert smoke.train_config.trainer.per_device_parallelism == 1
    assert smoke.train_config.data.expected_packed_examples == EPOCH_PACKED_EXAMPLES


def test_unsupported_gang_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported H100"):
        build_run(smoke=False, nodes=3)
