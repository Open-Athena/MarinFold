"""Check the resolved Marin configuration against exp277's training contract."""

from levanter.data.text.datasets import ConcatDatasetComponent
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.training.training import apply_output_path

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    DATA_SEED,
    MODEL_SEED,
    SEQ_LEN,
    STEPS_PER_EVAL,
)
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    OneEpochDataConfig,
)
from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_TRAIN,
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    PREFIX,
    RUN_ID,
    VERSION,
)
from experiments.exp343_models_complex_corpus_training.train import build_run

VALIDATIONS = ("input/validation", "input/validation-complex")


def resolved(*, smoke: bool, nodes: int):
    with build_context(BuildContext(VersionCodex(VERSION))):
        step = build_run(smoke=smoke, nodes=nodes)
        step.fingerprint()
        context = StepContext.for_run(
            output_path=step.path(PREFIX),
            prefix=PREFIX,
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
        pod = step.build_config(context)
    return pod, apply_output_path(pod.train_config, pod.output_path)


def test_production_contract_survives_marin_lowering() -> None:
    pod, config = resolved(smoke=False, nodes=16)
    assert config.trainer.num_train_steps == EPOCH_TRAIN_STEPS
    assert config.trainer.train_batch_size * config.train_seq_len == 1_048_576
    assert config.train_seq_len == SEQ_LEN
    assert config.trainer.per_device_parallelism == 1
    assert config.trainer.seed == MODEL_SEED
    assert config.data_seed == DATA_SEED
    assert config.optimizer.learning_rate == 1e-3
    assert config.optimizer.weight_decay == 0.2
    assert config.optimizer.warmup == 0.1
    assert config.optimizer.decay == 0.2
    assert config.optimizer.min_lr_ratio == 0.1
    assert config.initialize_from_checkpoint_path is None
    assert config.initialize_model_from_checkpoint_path is None
    assert config.trainer.steps_per_eval == STEPS_PER_EVAL
    assert config.trainer.max_eval_batches is None
    assert (
        config.trainer.checkpointer.base_path == f"{PREFIX}/runs/{RUN_ID}/checkpoints"
    )
    assert config.trainer.checkpointer.keep == [{"every": EPOCH_TRAIN_STEPS // 10}]
    assert not {"WANDB_API_KEY", "FSSPEC_S3", "HF_TOKEN"}.intersection(pod.env_vars)


def test_the_epoch_concatenates_all_five_training_corpora() -> None:
    _, config = resolved(smoke=False, nodes=16)
    assert isinstance(config.data, OneEpochDataConfig)
    assert config.data.expected_packed_examples == EPOCH_PACKED_EXAMPLES
    assert config.data.augmentation_num_train_steps == EPOCH_TRAIN_STEPS
    assert config.data.block_cross_document_attention
    combined = config.data.components[FULL_CORPUS]
    assert isinstance(combined, ConcatDatasetComponent)
    assert set(combined.children) == {
        f"input/full-epoch/{corpus.name}" for corpus in CORPORA
    }
    assert f"input/full-epoch/{COMPLEX_TRAIN.name}" in combined.children
    assert all(component.pack for component in combined.children.values())


def test_both_validation_sets_are_packed_and_never_trained_on() -> None:
    _, config = resolved(smoke=False, nodes=16)
    assert all(config.data.components[name].pack for name in VALIDATIONS)
    assert config.data.train_weights == {
        FULL_CORPUS: 1.0,
        **{name: 0.0 for name in VALIDATIONS},
    }
    # The held-out complex shard must not also be reachable through the epoch.
    combined = config.data.components[FULL_CORPUS]
    assert not set(VALIDATIONS).intersection(combined.children)


def test_smoke_and_production_share_only_the_adopted_inputs() -> None:
    with build_context(BuildContext(VersionCodex(VERSION))):
        smoke = build_run(smoke=True, nodes=16)
        production = build_run(smoke=False, nodes=16)
    shared = {dep.name for dep in smoke.deps} & {dep.name for dep in production.deps}
    assert shared == set(VALIDATIONS)
    assert smoke.path(PREFIX) != production.path(PREFIX)


def test_a_single_node_smoke_keeps_the_global_batch_size() -> None:
    _, config = resolved(smoke=True, nodes=1)
    assert config.trainer.num_train_steps == 10
    assert config.trainer.train_batch_size == 128
    assert config.trainer.per_device_parallelism == 8
    assert config.trainer.max_eval_batches == 2
    assert config.trainer.checkpointer.keep == []
