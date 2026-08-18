# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue selected exp232 CoreWeave models from their final trainer state.

The supported sources are the final ``step-145199`` checkpoints evaluated in
PR #244. Each continuation restores model, AdamW, RNG, data-position, and
absolute-step state, starts a new LR cycle at the source run's original peak,
holds it for 80% of the added training, and linearly decays to zero.

``SOURCE`` selects ``m2-p06-aug`` or ``m1-p02-aug``. ``CLUSTER`` and ``NODES``
select placement without entering production identity. Set ``SMOKE=1`` and
optionally ``SMOKE_STEPS`` for a short, separately named full-state validation
run. Omit ``--run`` to preview the lowered plan.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from datetime import timedelta

import click
from exp232_sweep import (
    AA_AUGMENTATION_SEED,
    AFDB_TOKENS,
    DATA_SEED,
    ESM_TOKENS,
    GLOBAL_BATCH_SIZE,
    MODEL_CONFIG,
    MODEL_PARAMS,
    NUM_TRAIN_STEPS,
    PERMANENT_CHECKPOINT_EVERY,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    TARGET_TRAIN_DOCUMENTS,
    TARGET_TRAIN_TOKENS,
    TOKENS_PER_STEP,
    WANDB_WATCH,
    ClusterSpec,
    GpuBatchConfig,
    _augment_lm_example,
    _parse_cluster,
    _parse_nodes,
    _run_exp232_train_job,
    _sweep_subversion,
    _truthy_env,
    _validate_contacts_v1_tokenizer,
    _verify_decontaminated_cache_counts,
    afdb_cache,
    esm_cache,
    gpu_batch_fit,
    validation_cache,
)
from exp232_sweep import EXPERIMENT_PREFIX as SWEEP_EXPERIMENT_PREFIX
from fray.types import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, marin_temp_bucket, prefix_join

RUN_PREFIX = "prot-exp232-cw-cv1-decontam-cont"
CONTINUATION_EXPERIMENT_PREFIX = (
    "s3://marin-us-east-02a/MarinFold/exp232_continue_cv1_decontam"
)

# Match exp199's continuation budget: three original-run token budgets.
ADDITIONAL_TRAIN_STEPS = 3 * NUM_TRAIN_STEPS
# The completed sources reached 10% of peak; the continuation ends at zero.
SOURCE_FINAL_LR_RATIO = 0.1
MIN_LR_RATIO = 0.0
WARMUP = 0.1
REWARMUP = 0.0
DECAY = 0.2
LR_SCHEDULE = "linear"
AUGMENTATION_KEY = "aug100"
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=30)


@dataclass(frozen=True)
class SourceModel:
    key: str
    identity: str
    run_id: str
    version: str
    checkpoint_step: int
    mixture_key: str
    point_key: str
    source_augmentation: str
    afdb_weight: float
    esm_weight: float
    learning_rate: float
    weight_decay: float

    @property
    def resume_step(self) -> int:
        """TrainerState.step restored by the zero-indexed checkpoint."""
        return self.checkpoint_step + 1


SOURCE_MODELS = {
    "m2-p06-aug": SourceModel(
        key="m2-p06-aug",
        identity="srcfinal",
        run_id="prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
        version="2026.08.14.2",
        checkpoint_step=145_199,
        mixture_key="m2",
        point_key="p06",
        source_augmentation="aug",
        afdb_weight=AFDB_TOKENS / TARGET_TRAIN_TOKENS,
        esm_weight=ESM_TOKENS / TARGET_TRAIN_TOKENS,
        learning_rate=1e-3,
        weight_decay=0.2,
    ),
    "m1-p02-aug": SourceModel(
        key="m1-p02-aug",
        identity="srcfinal",
        run_id="prot-exp232-cw-cv1-decontam-s02-m1-p02-aug",
        version="2026.08.14.2",
        checkpoint_step=145_199,
        mixture_key="m1",
        point_key="p02",
        source_augmentation="aug",
        afdb_weight=0.5,
        esm_weight=0.5,
        learning_rate=3.1623e-4,
        weight_decay=1.6,
    ),
}


class FullRateAminoAcidDataset(AsyncDataset[LmExample]):
    """Apply the validated contacts-v1 permutation to every example."""

    def __init__(self, dataset: AsyncDataset[LmExample], *, seed: int):
        self.dataset = dataset
        self.seed = seed

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        examples = await self.dataset.get_batch(indices)
        return [
            _augment_lm_example(
                example,
                seed=self.seed,
                index=index,
                probability=1.0,
            )
            for index, example in zip(indices, examples, strict=True)
        ]


@dataclass(frozen=True)
class FullRateAminoAcidDataConfig(LmDataConfig):
    augmentation_seed: int = AA_AUGMENTATION_SEED

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        return FullRateAminoAcidDataset(
            super().train_set(Pos, batch_schedule, key=key),
            seed=self.augmentation_seed,
        )


def augment_every_example(data: LmDataConfig) -> LmDataConfig:
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return FullRateAminoAcidDataConfig(**values)


def _parse_source() -> SourceModel:
    source = os.environ.get("SOURCE", "").strip().lower()
    try:
        return SOURCE_MODELS[source]
    except KeyError:
        choices = ", ".join(SOURCE_MODELS)
        raise SystemExit(f"SOURCE must be one of: {choices}") from None


def _parse_placement(*, smoke: bool) -> tuple[str, ClusterSpec, int]:
    cluster, spec = _parse_cluster()
    return cluster, spec, _parse_nodes(smoke=smoke)


def _training_env() -> dict[str, str]:
    required = ("WANDB_ENTITY", "WANDB_PROJECT")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(
            f"missing required environment variables: {', '.join(missing)}"
        )
    env = {
        "MARIN_PREFIX": CONTINUATION_EXPERIMENT_PREFIX,
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _validate_launch_prefix() -> None:
    configured = marin_prefix().rstrip("/")
    if configured != CONTINUATION_EXPERIMENT_PREFIX:
        raise ValueError(
            "MARIN_PREFIX must be exactly "
            f"{CONTINUATION_EXPERIMENT_PREFIX!r}, got {configured!r}"
        )


def source_checkpoint(source_model: SourceModel) -> ArtifactStep[LevanterCheckpoint]:
    source = prefix_join(
        SWEEP_EXPERIMENT_PREFIX,
        f"checkpoints/protein/{source_model.run_id}/{source_model.version}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        (
            "checkpoints/protein/exp232-cw-continuation-source/"
            f"{source_model.run_id}/step-{source_model.checkpoint_step}"
        ),
        source_model.version,
        source=source,
        kind=LevanterCheckpoint,
        config={
            "source_run": source_model.run_id,
            "source_version": source_model.version,
            "checkpoint_step": source_model.checkpoint_step,
        },
    )


@dataclass(frozen=True)
class RunShape:
    additional_steps: int
    end_step: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    wandb_group: str
    tags: list[str]


def _run_shape(
    source_model: SourceModel,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> RunShape:
    identity = (
        f"{subversion}-{source_model.mixture_key}-{source_model.point_key}-"
        f"{source_model.identity}-{AUGMENTATION_KEY}"
    )
    if smoke:
        additional_steps = int(os.environ.get("SMOKE_STEPS", "20"))
        if additional_steps < 10:
            raise ValueError("SMOKE_STEPS must be at least 10 to expose the LR shape")
        steps_per_eval = additional_steps
        permanent_checkpoint_every = None
        run_id = (
            f"{RUN_PREFIX}-smoke-{identity}-{cluster}-"
            f"{spec.gpu_variant.lower()}-n{nodes}"
        )
    else:
        additional_steps = ADDITIONAL_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{identity}"

    end_step = source_model.resume_step + additional_steps
    additional_tokens = additional_steps * TOKENS_PER_STEP
    cumulative_tokens = end_step * TOKENS_PER_STEP
    tags = [
        "protein",
        "exp232",
        "contacts-v1",
        "decontaminated",
        "continuation",
        f"sweep={subversion}",
        f"source={source_model.key}",
        f"source_run={source_model.run_id}",
        f"mixture={source_model.mixture_key}",
        f"point={source_model.point_key}",
        f"source_augmentation={source_model.source_augmentation}",
        f"augmentation={AUGMENTATION_KEY}",
        f"lr={source_model.learning_rate:g}",
        f"source_final_lr={source_model.learning_rate * SOURCE_FINAL_LR_RATIO:g}",
        f"continuation_final_lr={source_model.learning_rate * MIN_LR_RATIO:g}",
        f"wd={source_model.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={additional_steps}",
        f"tokens={additional_tokens}",
        f"source_documents={TARGET_TRAIN_DOCUMENTS}",
        f"source_tokens={TARGET_TRAIN_TOKENS}",
        f"source_checkpoint_step={source_model.checkpoint_step}",
        f"start_step={source_model.resume_step}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        f"cumulative_tokens={cumulative_tokens}",
        "schedule=constant80-linear20-zero",
        f"initialization=checkpoint-step-{source_model.checkpoint_step}",
    ]
    if smoke:
        tags.extend(
            [
                f"cluster={cluster}",
                f"gpu={spec.gpu_variant}",
                f"nodes={nodes}",
                "smoke",
            ]
        )
    return RunShape(
        additional_steps=additional_steps,
        end_step=end_step,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=f"checkpoints/protein/{run_id}",
        wandb_group=f"{RUN_PREFIX}-{subversion}",
        tags=tags,
    )


def _apply_continuation_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    source_model: SourceModel,
    shape: RunShape,
    batch: GpuBatchConfig,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        execution_prefix = ctx.prefix.rstrip("/")
        if (
            not ctx.is_fingerprint
            and execution_prefix != CONTINUATION_EXPERIMENT_PREFIX
            and not execution_prefix.startswith(f"{CONTINUATION_EXPERIMENT_PREFIX}/")
        ):
            raise ValueError(
                f"execution prefix {ctx.prefix!r} is outside "
                f"{CONTINUATION_EXPERIMENT_PREFIX!r}"
            )

        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError("continuation requires the source checkpoint dependency")
        exact_checkpoint = (
            prefix_join(
                source_checkpoint_dir,
                f"step-{source_model.checkpoint_step}",
            )
            if source_checkpoint_dir is not None
            else None
        )

        trainer = replace(
            pod.train_config.trainer,
            initialize_from=exact_checkpoint,
            max_eval_batches=1 if smoke else None,
            watch=WANDB_WATCH,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL,
                keep=(
                    [{"every": shape.permanent_checkpoint_every}]
                    if shape.permanent_checkpoint_every is not None
                    else []
                ),
            ),
        )
        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                key: replace(component, pack=True)
                for key, component in pod.train_config.data.components.items()
            },
            block_cross_document_attention=True,
        )
        data = augment_every_example(data)

        if not ctx.is_fingerprint:
            tracker = trainer.tracker
            if not smoke:
                tracker = replace(
                    tracker,
                    tags=[
                        *tracker.tags,
                        f"cluster={cluster}",
                        f"gpu={spec.gpu_variant}",
                        f"nodes={nodes}",
                    ],
                )
            trainer = replace(
                trainer,
                tracker=tracker,
                per_device_parallelism=batch.per_device_parallelism,
                per_device_eval_parallelism=batch.per_device_parallelism,
            )

        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            # Put the exact full-state source on TrainerConfig so the model,
            # optimizer, RNG, data position, and absolute step restore together.
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(
        step,
        build_config=build_config,
        run=_run_exp232_train_job,
    )


def build_run(
    source_model: SourceModel,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _verify_decontaminated_cache_counts()
    shape = _run_shape(
        source_model,
        subversion=subversion,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )
    batch = gpu_batch_fit(spec, nodes=nodes, smoke=smoke)
    env = _training_env()

    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=source_model.learning_rate,
            weight_decay=source_model.weight_decay,
            warmup=WARMUP,
            rewarmup=REWARMUP,
            decay=DECAY,
            cycle_length=[source_model.resume_step, shape.additional_steps],
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={
            afdb_cache(): source_model.afdb_weight,
            esm_cache(): source_model.esm_weight,
        },
        validation=[validation_cache()],
        init_from=source_checkpoint(source_model),
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.end_step,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu(
            spec.gpu_variant,
            count=spec.gpus_per_node,
            replicas=nodes,
            cpu=spec.cpu,
            ram=spec.ram,
            disk=spec.disk,
        ),
        tensor_parallel_size=batch.tensor_parallelism,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=shape.wandb_group,
        tags=shape.tags,
        env_vars=env,
    )
    if smoke:
        step = replace(
            step,
            override_path=marin_temp_bucket(1, f"checkpoints/{shape.run_id}"),
        )
    return _apply_continuation_overrides(
        step,
        source_model=source_model,
        shape=shape,
        batch=batch,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    smoke = _truthy_env("SMOKE")
    source_model = _parse_source()
    cluster, spec, nodes = _parse_placement(smoke=smoke)
    _validate_launch_prefix()
    return build_run(
        source_model,
        subversion=_sweep_subversion(),
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
