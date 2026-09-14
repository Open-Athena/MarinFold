"""Check the resolved Marin configuration across the required runtime upgrade."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path
from levanter.data.text.datasets import ConcatDatasetComponent

from experiments.exp288_models_chinchilla_size_sweep.config import (
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    PREFIX,
    TRIALS,
    VERSION,
    trainable_params,
)
from experiments.exp288_models_chinchilla_size_sweep.epoch_data import (
    FULL_CORPUS,
    OneEpochDataConfig,
)
from experiments.exp288_models_chinchilla_size_sweep.train import build_run


def test_trial_catalog_pins_chinchilla_size_points() -> None:
    params = {trial_id: trainable_params(trial.model) for trial_id, trial in TRIALS.items()}
    assert 650_000_000 <= params["0_7b"] <= 750_000_000
    assert 1_400_000_000 <= params["1_5b"] <= 1_600_000_000
    assert 2_800_000_000 <= params["3b"] <= 3_200_000_000
    token_budget = EPOCH_TRAIN_STEPS * 128 * 8192
    assert 390 <= token_budget / params["0_7b"] <= 405
    assert 185 <= token_budget / params["1_5b"] <= 195
    assert 90 <= token_budget / params["3b"] <= 95


def test_production_contract_survives_marin_lowering() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        trial = TRIALS["1_5b"]
        step = build_run(smoke=False, trial=trial, nodes=16)
        step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(context)
    config = apply_output_path(pod.train_config, pod.output_path)
    assert config.trainer.num_train_steps == (EPOCH_PACKED_EXAMPLES + 127) // 128
    assert config.trainer.train_batch_size * config.train_seq_len == 1_048_576
    assert config.trainer.per_device_parallelism == 1
    assert config.optimizer.learning_rate == 1e-3
    assert config.optimizer.weight_decay == 0.2
    assert config.initialize_from_checkpoint_path is None
    assert config.initialize_model_from_checkpoint_path is None
    assert isinstance(config.data, OneEpochDataConfig)
    assert config.data.augmentation_num_train_steps == EPOCH_TRAIN_STEPS
    assert config.data.block_cross_document_attention
    combined = config.data.components[FULL_CORPUS]
    assert isinstance(combined, ConcatDatasetComponent)
    assert set(combined.children) == {
        f"input/full-epoch/{corpus.name}" for corpus in CORPORA
    }
    assert all(component.pack for component in combined.children.values())
    assert config.data.components["input/validation"].pack
    weights = config.data.train_weights
    assert weights == {FULL_CORPUS: 1.0, "input/validation": 0.0}
    assert (
        config.trainer.checkpointer.base_path == f"{PREFIX}/runs/{trial.run_id}/checkpoints"
    )
    assert config.trainer.checkpointer.keep == [{"every": EPOCH_TRAIN_STEPS // 10}]
    assert config.trainer.max_eval_batches is None
    assert not {"WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"}.intersection(pod.env_vars)


def test_smoke_and_production_caches_have_distinct_artifact_identities() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        trial = TRIALS["1_5b"]
        smoke = build_run(smoke=True, trial=trial, nodes=16)
        production = build_run(smoke=False, trial=trial, nodes=16)
    shared = {dep.name for dep in smoke.deps} & {dep.name for dep in production.deps}
    assert shared == {"input/validation"}
    assert smoke.path(PREFIX) != production.path(PREFIX)
