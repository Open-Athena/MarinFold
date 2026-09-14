"""Validate the full-state, reshuffled exp277 continuation contract."""

import pytest
from levanter.data.text.datasets import ConcatDatasetComponent
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    PREFIX,
    RUN_ID,
    VERSION,
)
from experiments.exp277_models_single_mpnn_pilot.continue_train import (
    CONTINUATION_DATA_SEED,
    CONTINUATION_RUN_ID,
    SOURCE_CHECKPOINT_STEP,
    SOURCE_RESUME_STEP,
    build_continuation_run,
)
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    ContinuationEpochDataConfig,
)


def test_continuation_restores_full_state_and_adds_exactly_one_epoch() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_continuation_run(smoke=False, nodes=16)
        step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(context)
    config = apply_output_path(pod.train_config, pod.output_path)
    assert config.trainer.num_train_steps == SOURCE_RESUME_STEP + EPOCH_TRAIN_STEPS
    assert config.trainer.initialize_from == (
        f"{PREFIX}/runs/{RUN_ID}/checkpoints/step-{SOURCE_CHECKPOINT_STEP}"
    )
    assert not config.trainer.allow_partial_checkpoint
    assert config.data_seed == CONTINUATION_DATA_SEED == 1
    assert config.optimizer.cycle_length == [SOURCE_RESUME_STEP, EPOCH_TRAIN_STEPS]
    assert config.optimizer.rewarmup == 0.0
    assert config.optimizer.decay == 0.2
    assert config.initialize_from_checkpoint_path is None
    assert config.initialize_model_from_checkpoint_path is None
    assert isinstance(config.data, ContinuationEpochDataConfig)
    assert config.data.start_step == SOURCE_RESUME_STEP
    assert config.data.source_augmentation_step == SOURCE_RESUME_STEP
    assert config.data.augmentation_num_train_steps == EPOCH_TRAIN_STEPS
    assert config.data.expected_packed_examples == EPOCH_PACKED_EXAMPLES
    combined = config.data.components[FULL_CORPUS]
    assert isinstance(combined, ConcatDatasetComponent)
    assert set(combined.children) == {
        f"input/full-epoch2/{corpus.name}" for corpus in CORPORA
    }
    assert config.trainer.checkpointer.base_path == (
        f"{PREFIX}/runs/{CONTINUATION_RUN_ID}/checkpoints"
    )
    assert config.trainer.checkpointer.keep == [{"every": EPOCH_TRAIN_STEPS // 10}]
    schedule = config.optimizer.lr_scheduler(config.trainer.num_train_steps)
    decay_start = SOURCE_RESUME_STEP + int(0.8 * EPOCH_TRAIN_STEPS)
    assert float(schedule(SOURCE_RESUME_STEP)) == pytest.approx(1e-3)
    assert float(schedule(decay_start - 1)) == pytest.approx(1e-3)
    assert float(schedule(config.trainer.num_train_steps - 1)) == pytest.approx(
        1e-4, rel=3e-4
    )


def test_continuation_smoke_has_separate_output_and_ten_added_steps() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        smoke = build_continuation_run(smoke=True, nodes=1)
        production = build_continuation_run(smoke=False, nodes=16)
    assert smoke.path(PREFIX) != production.path(PREFIX)
    context = StepContext.for_run(
        output_path=smoke.path(PREFIX),
        prefix=PREFIX,
        runtime_args=smoke.runtime_args,
        deps=smoke.deps,
    )
    pod = smoke.build_config(context)
    assert pod.train_config.trainer.num_train_steps == SOURCE_RESUME_STEP + 10
    assert pod.train_config.optimizer.cycle_length == [SOURCE_RESUME_STEP, 10]
