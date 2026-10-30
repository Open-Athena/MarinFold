"""Compare fully resolved configurations against the actual default builder."""

from dataclasses import replace

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint, TrainLmConfig, apply_output_path

from experiments.exp277_models_single_mpnn_pilot.train import build_run as baseline_run
from experiments.exp362_models_smaller_model_scaling.train import (
    PREFIX, PROFILES, TRIALS, VERSION, build_run, parameter_count, run_identity,
)


def resolve(step: ArtifactStep[LevanterCheckpoint]) -> TrainLmConfig:
    """Lower output and checkpoint paths exactly as the worker will."""
    step.fingerprint()
    ctx = StepContext.for_run(
        output_path=step.path(PREFIX), prefix=PREFIX,
        runtime_args=step.runtime_args, deps=step.deps,
    )
    pod = step.build_config(ctx)
    return apply_output_path(pod.train_config, pod.output_path)


@pytest.mark.parametrize("trial", TRIALS)
def test_only_depth_and_operational_identity_differ(trial: str) -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        baseline = resolve(baseline_run(smoke=False, nodes=4))
        candidate = resolve(build_run(trial=trial, smoke=False, cluster="cw-us-east-02a", nodes=4))
    assert candidate.model.num_layers == TRIALS[trial]
    assert 0 < parameter_count(TRIALS[trial]) < parameter_count(24)
    # Compare the complete configuration, including data, optimizer and schedule;
    # normalize only identity-derived paths and the deliberately varied depth.
    normalized = replace(
        candidate,
        model=replace(candidate.model, num_layers=24),
        trainer=replace(
            candidate.trainer, id=baseline.trainer.id,
            tracker=baseline.trainer.tracker,
            checkpointer=baseline.trainer.checkpointer,
        ),
        hf_save_path=baseline.hf_save_path,
    )
    assert normalized == baseline
    assert candidate.trainer.checkpointer.keep == baseline.trainer.checkpointer.keep
    assert candidate.trainer.checkpointer.save_interval == baseline.trainer.checkpointer.save_interval
    assert candidate.trainer.num_train_steps == 266345
    assert candidate.data.expected_packed_examples == 34092146


@pytest.mark.parametrize("cluster", PROFILES)
@pytest.mark.parametrize("nodes", (1, 2, 4, 8, 16))
def test_placement_preserves_identity_batch_and_checkpoint(cluster: str, nodes: int) -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        config = resolve(build_run(trial="depth16", smoke=False, cluster=cluster, nodes=nodes, attempt=7))
    identity = run_identity("depth16", smoke=False, cluster=cluster, nodes=nodes, attempt=7)
    assert identity == "contacts-v1-exp362-depth16-full-epoch"
    assert config.trainer.checkpointer.base_path == f"{PREFIX}/runs/{identity}/checkpoints"
    assert config.trainer.train_batch_size == 128
    microbatch = config.trainer.per_device_parallelism * nodes * PROFILES[cluster][1]
    assert 128 % microbatch == 0


def test_smoke_cannot_write_production_checkpoint() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        smoke = resolve(build_run(trial="depth4", smoke=True, cluster="cw-us-east-02a", nodes=1))
        production = resolve(build_run(trial="depth4", smoke=False, cluster="cw-us-east-02a", nodes=1))
    assert smoke.trainer.id != production.trainer.id
    assert smoke.trainer.checkpointer.base_path != production.trainer.checkpointer.base_path
    assert smoke.trainer.num_train_steps == 10
    assert production.trainer.num_train_steps == 266345
