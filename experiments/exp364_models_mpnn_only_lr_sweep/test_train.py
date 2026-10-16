"""Compare resolved production semantics directly with the current default."""

from dataclasses import asdict

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp277_models_single_mpnn_pilot.train import build_run as baseline_run
from experiments.exp364_models_mpnn_only_lr_sweep.config import (
    CORPORA,
    LEARNING_RATES,
    PREFIX,
    TRAIN_STEPS,
    VERSION,
    run_id_for,
)
from experiments.exp364_models_mpnn_only_lr_sweep.epoch_data import FULL_CORPUS
from experiments.exp364_models_mpnn_only_lr_sweep.train import build_run


def resolve(builder, **kwargs):
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = builder(**kwargs)
        fingerprint = step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(context)
    return apply_output_path(pod.train_config, pod.output_path), fingerprint


@pytest.mark.parametrize("trial", LEARNING_RATES)
def test_only_requested_training_semantics_change(trial: str) -> None:
    baseline, _ = resolve(baseline_run, smoke=False, nodes=8)
    config, _ = resolve(build_run, trial=trial, smoke=False, nodes=8)
    assert asdict(config.model) == asdict(baseline.model)
    optimizer = asdict(config.optimizer)
    reference = asdict(baseline.optimizer)
    assert optimizer.pop("learning_rate") == LEARNING_RATES[trial]
    reference.pop("learning_rate")
    assert optimizer == reference
    for field in (
        "train_batch_size",
        "seed",
        "mp",
        "steps_per_eval",
        "max_eval_batches",
        "per_device_parallelism",
        "per_device_eval_parallelism",
        "watch",
        "mesh",
    ):
        assert getattr(config.trainer, field) == getattr(baseline.trainer, field)
    for field in (
        "train_seq_len",
        "data_seed",
        "z_loss_weight",
        "initialize_from_checkpoint_path",
        "initialize_model_from_checkpoint_path",
        "initialize_from_hf",
    ):
        assert getattr(config, field) == getattr(baseline, field)
    for field in (
        "tokenizer",
        "shuffle",
        "block_cross_document_attention",
        "augmentation_seed",
    ):
        assert getattr(config.data, field) == getattr(baseline.data, field)
    assert config.trainer.num_train_steps == TRAIN_STEPS == 373_769
    assert config.data.augmentation_num_train_steps == TRAIN_STEPS
    assert {corpus.name for corpus in CORPORA} == {"mpnn-afdb", "mpnn-esm"}
    assert sum(corpus.documents for corpus in CORPORA) == 162_574_724
    children = config.data.components[FULL_CORPUS].children
    assert set(children) == {f"input/two-epoch/{corpus.name}" for corpus in CORPORA}
    assert all(component.pack for component in children.values())
    assert (
        config.data.components["input/validation"]
        == baseline.data.components["input/validation"]
    )
    assert config.data.train_weights == {FULL_CORPUS: 1.0, "input/validation": 0.0}
    assert (
        config.trainer.checkpointer.base_path
        == f"{PREFIX}/runs/{run_id_for(trial)}/checkpoints"
    )


def test_placement_and_attempt_do_not_change_production_identity() -> None:
    first, fingerprint = resolve(
        build_run, trial="lr1e-3", smoke=False, nodes=4, attempt=1
    )
    second, other = resolve(build_run, trial="lr1e-3", smoke=False, nodes=8, attempt=2)
    assert fingerprint == other
    assert first.trainer.id == second.trainer.id
    assert first.trainer.checkpointer.base_path == second.trainer.checkpointer.base_path
    smoke, _ = resolve(build_run, trial="lr1e-3", smoke=True, nodes=8)
    assert smoke.trainer.id != first.trainer.id
    assert smoke.trainer.checkpointer.base_path != first.trainer.checkpointer.base_path
