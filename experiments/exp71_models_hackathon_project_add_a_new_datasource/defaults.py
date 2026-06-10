# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Vendored (and trimmed) from marin/experiments/defaults.py on the
# protein-training-1b branch. Only the protein-training surface area is
# kept: default_tokenize, default_train, and the helpers they call.
# Stripped: default_download, default_sft, default_dpo, the in-process
# training entry point, the default Paloma + lm-eval-harness validation
# wiring.
#
# Marin packages its `experiments/` tree as the `marin-root` distribution
# but only `marin` is on the wheel mirror we consume; hence the vendor.

import dataclasses
import logging
import os
from collections.abc import Sequence
from datetime import timedelta

import jmp
from fray import ResourceConfig
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    LmDatasetFormatBase,
    LMMixtureDatasetConfig,
    TextLmDatasetFormat,
)
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.optim.model_averaging import EmaModelAveragingConfig
from levanter.schedule import BatchSchedule
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    VersionedValue,
    ensure_versioned,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from marin.execution.remote import remote
from marin.processing.tokenize import (
    HfDatasetSpec,
    TokenizeConfig,
    lm_data_config,
    tokenize,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from marinfold_models.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


HF_BUCKET_URI_PREFIX = "hf://buckets/"
HF_BUCKET_PATH_PREFIX = "buckets/"


def _is_hf_bucket_path(path: str) -> bool:
    return path.startswith(HF_BUCKET_URI_PREFIX) or path.startswith(HF_BUCKET_PATH_PREFIX)


def _truncate_wandb_name(name: str) -> str:
    """Truncate a run name to fit W&B's 64-character limit, preserving the trailing suffix."""
    if len(name) <= 64:
        return name
    old_name = name
    if "-" not in name:
        name = name[:64]
    else:
        prefix, suffix = name.rsplit("-", 1)
        if len(suffix) >= 64:
            suffix = suffix[:64]
            name = suffix
        else:
            name = prefix[: 63 - len(suffix)] + "-" + suffix
    logger.warning("Truncated name from %s to %s to fit within W&B limits.", old_name, name)
    return name


def _resolve_hf_export_steps(steps_per_hf_export: int | None, steps_per_export: int | None) -> int | None:
    """Resolve the HF export step interval: None means same as checkpoint, -1 means disabled."""
    if steps_per_hf_export is None:
        return steps_per_export
    if steps_per_hf_export == -1:
        return None
    return steps_per_hf_export


def _checkpoint_keep(steps_per_export: int | None) -> list[dict]:
    """Build the `keep` list for `CheckpointerConfig`.

    None means keep no permanent intermediate checkpoints (only the final
    checkpoint is saved at end-of-training, plus a rolling temporary
    checkpoint for resumption).
    """
    if steps_per_export is None:
        return []
    return [dict(every=steps_per_export)]


def _validate_train_length(train_seq_len: int | None, model_config: LmConfig) -> int:
    """Resolve and validate the training sequence length against the model's max."""
    actual = unwrap_versioned_value(model_config)
    train_length = train_seq_len or actual.max_seq_len
    if train_length > actual.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {actual.max_seq_len}.")
    return train_length


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa: B008
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
    levanter_batch_size: int | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    """Tokenize a dataset using Levanter's tokenization infrastructure.

    See ``marin/experiments/defaults.py:default_tokenize`` for the
    upstream version this vendors (with HF / fsspec / TokenizeConfig
    branches preserved). Output paths land under ``tokenized/<name>``.
    """
    extra_kwargs: dict = {}
    if worker_resources is not None:
        extra_kwargs["worker_resources"] = worker_resources

    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    elif (
        isinstance(dataset, str)
        and not _is_hf_bucket_path(dataset)
        and dataset.count("/") == 1
        and not fsspec_utils.exists(dataset)
    ):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=remote(
            tokenize,
            resources=resources or ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
            pip_dependency_groups=["cpu"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
            },
        ),
        config=config,
    )


def _prepare_data_config(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    use_default_validation: bool,
) -> LMMixtureDatasetConfig:
    """Prepare the tokenized data for training.

    This vendor only supports ``use_default_validation=False`` — MarinFold
    runs explicitly enumerate their validation components (Paloma is not
    useful for a protein-structure vocabulary). Pass an
    ``LMMixtureDatasetConfig`` for fine-grained control, otherwise an
    ``ExecutorStep`` / ``InputName`` pointing at a single tokenized dataset.
    """
    if use_default_validation:
        raise NotImplementedError(
            "default validation sets (Paloma + uncheatable evals) are not "
            "vendored into MarinFold. Pass an explicit LMMixtureDatasetConfig "
            "with your validation components, and use_default_validation=False."
        )

    if isinstance(tokenized, (InputName, ExecutorStep)):
        return lm_data_config(training_set=tokenized, validation_sets={})
    return tokenized


def _build_train_lm_config(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    *,
    tags: Sequence[str] = (),
    use_default_validation: bool = False,
    eval_harness_tasks: Sequence[EvalTaskConfig] = (),
    wandb_name: str | None = None,
    wandb_group: str | None = None,
) -> tuple[str, TrainLmConfig]:
    """Build the ``TrainLmConfig`` used by ``default_train``."""
    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    if wandb_group is None:
        wandb_group = os.environ.get("WANDB_GROUP")

    name = _truncate_wandb_name(name)

    if eval_harness_tasks:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(eval_harness_tasks))
    else:
        harness_config = None

    steps_per_export = train_config.steps_per_export
    steps_per_export_hf = _resolve_hf_export_steps(train_config.steps_per_hf_export, steps_per_export)

    model_averaging = None
    if train_config.ema_beta is not None:
        model_averaging = EmaModelAveragingConfig(beta=train_config.ema_beta)

    if train_config.per_device_eval_parallelism is None:
        per_device_eval_parallelism = -1
    else:
        per_device_eval_parallelism = train_config.per_device_eval_parallelism

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf
    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    train_length = _validate_train_length(train_config.train_seq_len, model_config)

    inner_config = TrainLmConfig(
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="MarinFold",
                name=wandb_name,
                tags=[*tags],
                group=wandb_group,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_config.train_batch_size,
            per_device_parallelism=train_config.per_device_parallelism,
            num_train_steps=train_config.num_train_steps,
            steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=_checkpoint_keep(steps_per_export),
            ),
            model_averaging=model_averaging,
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": train_config.tensor_parallel_size},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            per_device_eval_parallelism=per_device_eval_parallelism,
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
            initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
            watch=train_config.watch,
            profiler=train_config.profiler,
            use_explicit_mesh_axes=train_config.explicit_mesh_axes,
        ),
        initialize_from_checkpoint_path=(
            checkpoint_path_to_load_from if train_config.reset_data_loader_on_init else None
        ),
        initialize_from_hf=hf_checkpoint_path_to_load_from or False,
        pad_tokenizer_to_match_model=train_config.pad_tokenizer_to_match_model,
        z_loss_weight=train_config.z_loss_weight,
        train_seq_len=train_length,
        model=model_config,
        optimizer=(
            train_config.optimizer_config
            if getattr(train_config, "optimizer_config", None) is not None
            else AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=(
                    train_config.weight_decay if train_config.weight_decay is not None else AdamConfig().weight_decay
                ),
                beta1=(train_config.beta1 if train_config.beta1 is not None else AdamConfig().beta1),
                beta2=(train_config.beta2 if train_config.beta2 is not None else AdamConfig().beta2),
                epsilon=(train_config.epsilon if train_config.epsilon is not None else AdamConfig().epsilon),
                max_grad_norm=(
                    train_config.max_grad_norm if train_config.max_grad_norm is not None else AdamConfig().max_grad_norm
                ),
                warmup=(train_config.warmup if train_config.warmup is not None else AdamConfig().warmup),
                rewarmup=(train_config.rewarmup if train_config.rewarmup is not None else AdamConfig().rewarmup),
                decay=(train_config.decay if train_config.decay is not None else AdamConfig().decay),
                lr_schedule=(
                    train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig().lr_schedule
                ),
                cycle_length=train_config.cycle_length,
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
                skip_bad_steps=train_config.skip_bad_steps,
            )
        ),
        hf_save_steps=steps_per_export_hf,
        hf_generation_eos_token_ids=train_config.hf_generation_eos_token_ids,
        data_seed=train_config.data_seed,
        eval_harness_steps=train_config.steps_per_task_eval or 10000,
        eval_harness=harness_config,
    )

    return name, inner_config


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = False,
    eval_harness_tasks: Sequence[EvalTaskConfig] = (),
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    override_output_path: str | None = None,
) -> ExecutorStep:
    """Train a language model using the MarinFold default recipe.

    Output path lands under ``checkpoints/<name>`` on the executor; W&B
    project is hardcoded to ``MarinFold``. See
    ``marin/experiments/defaults.py:default_train`` for the upstream
    reference. ``use_default_validation=True`` is not supported in this
    vendor — supply an ``LMMixtureDatasetConfig`` with explicit
    validation components instead.
    """
    name, inner_config = _build_train_lm_config(
        name,
        tokenized,
        model_config,
        train_config,
        tags=tags,
        use_default_validation=use_default_validation,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
    )

    pretraining_data = inner_config.data
    tokenizer_name = unwrap_versioned_value(pretraining_data.tokenizer)
    train_length = unwrap_versioned_value(inner_config.train_seq_len)
    schedule = BatchSchedule(unwrap_versioned_value(train_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(unwrap_versioned_value(train_config.num_train_steps))

    pod_config = train_config.resources

    config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=pod_config,
        output_path=this_output_path(),
        env_vars=train_config.env_vars,
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a model (tokenizer={tokenizer_name}) for "
            f"{unwrap_versioned_value(train_config.num_train_steps)} (steps) * "
            f"{unwrap_versioned_value(train_config.train_batch_size)} (batch_size) * "
            f"{train_length} (train_seq_len) "
            f"= {total_examples * train_length} tokens."
        ),
        fn=run_levanter_train_lm,
        resources=train_config.resources,
        config=config,
        override_output_path=override_output_path,
    )


__all__ = [
    "SimpleTrainConfig",
    "default_train",
    "default_tokenize",
]
