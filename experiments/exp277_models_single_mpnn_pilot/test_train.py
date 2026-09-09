"""Check the resolved Marin configuration across the required runtime upgrade."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AminoAcidAugmentedDataConfig,
)
from experiments.exp277_models_single_mpnn_pilot.config import PREFIX, RUN_ID, VERSION
from experiments.exp277_models_single_mpnn_pilot.train import build_run


def test_production_contract_survives_marin_lowering() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_run(smoke=False, nodes=16)
        step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(context)
    config = apply_output_path(pod.train_config, pod.output_path)
    assert config.trainer.num_train_steps == 145_200
    assert config.trainer.train_batch_size * config.train_seq_len == 1_048_576
    assert config.trainer.per_device_parallelism == 1
    assert config.optimizer.learning_rate == 1e-3
    assert config.optimizer.weight_decay == 0.2
    assert config.initialize_from_checkpoint_path is None
    assert config.initialize_model_from_checkpoint_path is None
    assert isinstance(config.data, AminoAcidAugmentedDataConfig)
    assert config.data.augmentation_num_train_steps == 145_200
    assert config.data.block_cross_document_attention
    assert all(component.pack for component in config.data.components.values())
    weights = config.data.train_weights
    assert weights["input/full/native-afdb"] == weights["input/full/mpnn-afdb"]
    assert weights["input/full/native-esm"] == weights["input/full/mpnn-esm"]
    assert weights["input/validation"] == 0
    assert sum(weights.values()) == 1
    assert (
        config.trainer.checkpointer.base_path == f"{PREFIX}/runs/{RUN_ID}/checkpoints"
    )
    assert config.trainer.checkpointer.keep == [{"every": 14520}]
    assert config.trainer.max_eval_batches is None
    assert not {"WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"}.intersection(pod.env_vars)


def test_smoke_and_production_caches_have_distinct_artifact_identities() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        smoke = build_run(smoke=True, nodes=16)
        production = build_run(smoke=False, nodes=16)
    shared = {dep.name for dep in smoke.deps} & {dep.name for dep in production.deps}
    assert shared == {"input/validation"}
    assert smoke.path(PREFIX) != production.path(PREFIX)
