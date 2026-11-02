# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue either completed exp199 m1-p03 model from full trainer state.

Each run preserves the model, AdamW state, and absolute trainer step from the
selected augmented or unaugmented source model. It starts a new constant-then-
linear LR cycle and a new W&B run while continuing the same 50/50 AFDB+ESM
data stream with amino-acid statement permutation fixed at 100% probability.

The source checkpoint must first be restored from
``open-athena/marinfold-exp199`` into the region-local continuation-init path.

Preview::

    MARIN_PREFIX=gs://marin-us-east1/protein-structure/MarinFold/exp199_continue_contacts_v1 \
      REGION=us-east1 TPU=v6e-64 SOURCE=aug \
      uv run --extra tpu --frozen python exp199_continue_trc.py \
      --version 2026.08.10.3

Execute by adding ``--run``. Set ``SMOKE=1`` and optionally ``SMOKE_STEPS``
for a short, separately named validation run.
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import click
from fray.types import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
from marin.execution.build_context import current_build_context
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.rl.placement import marin_prefix_for_region, singleton_region_list
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

from exp199_sweep_trc import (
    AA_AUGMENTATION_SEED,
    DATA_SEED,
    GLOBAL_BATCH_SIZE,
    MODEL_CONFIG,
    MODEL_PARAMS,
    PERMANENT_CHECKPOINT_EVERY,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    TOKENS_PER_STEP,
    TpuBatchConfig,
    _augment_lm_example,
    _regional_prefix_guard,
    _training_env,
    _truthy_env,
    _validate_contacts_v1_tokenizer,
    _validate_launch_prefix,
    _validate_placement,
    afdb_cache,
    batch_fit,
    esm_cache,
    validation_cache,
)

RUN_PREFIX = "prot-exp199-cv1-cont"
SOURCE_SEED_NAMESPACE = "checkpoints/protein/exp199-continuation-init"
SOURCE_SEED_VERSION = "2026.08.09.1"

SOURCE_TRAIN_STEPS = 72_600
ADDITIONAL_TRAIN_STEPS = 72_600
LEARNING_RATE = 3.1623e-4
MIN_LR_RATIO = 0.0
WEIGHT_DECAY = 0.1
WARMUP = 0.0
REWARMUP = 0.0
DECAY = 0.2
LR_SCHEDULE = "linear"

MIXTURE_KEY = "m1"
POINT_KEY = "p03"
AUGMENTATION_KEY = "aug100"
AFDB_WEIGHT = 0.5
ESM_WEIGHT = 0.5


@dataclass(frozen=True)
class SourceModel:
    key: str
    identity: str
    run_id: str


SOURCE_MODELS = {
    "aug": SourceModel(
        key="aug",
        identity="srcaug",
        run_id="prot-exp199-cv1-s01-m1-p03-aug-us-east1",
    ),
    "base": SourceModel(
        key="base",
        identity="srcbase",
        run_id="prot-exp199-cv1-s01-m1-p03-base-us-east5",
    ),
}


class FullRateAminoAcidDataset(AsyncDataset[LmExample]):
    """Apply the already-validated contacts-v1 permutation to every example."""

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


def continuation_batch_fit(tpu: str, *, smoke: bool) -> TpuBatchConfig:
    config = batch_fit(tpu, GLOBAL_BATCH_SIZE)
    if smoke and tpu == "v6e-4" and config.per_device_parallelism > 8:
        # The #166 estimate puts this shape 10.56 MiB over v6e HBM at 16.
        per_device_parallelism = 8
        microbatch_size = per_device_parallelism * config.data_parallelism
        return replace(
            config,
            per_device_parallelism=per_device_parallelism,
            gradient_accumulation=GLOBAL_BATCH_SIZE // microbatch_size,
        )
    return config


def source_checkpoint(
    region: str, source_model: SourceModel
) -> ArtifactStep[LevanterCheckpoint]:
    source = prefix_join(
        marin_prefix_for_region(region),
        f"{SOURCE_SEED_NAMESPACE}/{source_model.run_id}/{SOURCE_SEED_VERSION}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        f"{SOURCE_SEED_NAMESPACE}/{source_model.run_id}",
        SOURCE_SEED_VERSION,
        source=source,
        kind=LevanterCheckpoint,
        config={"source_run": source_model.run_id, "region": region},
    )


@dataclass(frozen=True)
class RunShape:
    additional_steps: int
    end_step: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    tags: list[str]


def _subversion() -> str:
    context = current_build_context()
    if context is None:
        raise ValueError("exp199 continuation must be built under --version")
    if context.versions.overrides:
        raise ValueError("--override is not supported because sNN is run identity")
    match = re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.(\d+)", context.versions.default)
    if match is None:
        raise ValueError("--version must be a suffixed CalVer such as 2026.08.09.1")
    suffix = int(match.group(1))
    if not 1 <= suffix <= 99:
        raise ValueError(f"CalVer suffix must be in 1--99, got {suffix}")
    return f"s{suffix:02d}"


def _parse_region() -> str:
    region = os.environ.get("REGION", "").strip().lower()
    if not region:
        raise SystemExit("missing required env var REGION")
    marin_prefix_for_region(region)
    return region


def _parse_tpu() -> str:
    tpu = os.environ.get("TPU", "").strip().lower()
    if not tpu:
        raise SystemExit("missing required env var TPU")
    return tpu


def _parse_source() -> SourceModel:
    source = os.environ.get("SOURCE", "").strip().lower()
    try:
        return SOURCE_MODELS[source]
    except KeyError:
        choices = ", ".join(SOURCE_MODELS)
        raise SystemExit(f"SOURCE must be one of: {choices}") from None


def _run_shape(
    *,
    subversion: str,
    region: str,
    tpu: str,
    source_model: SourceModel,
    smoke: bool,
) -> RunShape:
    identity = (
        f"{subversion}-{MIXTURE_KEY}-{POINT_KEY}-"
        f"{source_model.identity}-{AUGMENTATION_KEY}"
    )
    if smoke:
        additional_steps = int(os.environ.get("SMOKE_STEPS", "20"))
        if additional_steps < 10:
            raise ValueError("SMOKE_STEPS must be at least 10 to expose the LR shape")
        steps_per_eval = additional_steps
        permanent_checkpoint_every = None
        run_id = f"{RUN_PREFIX}-smoke-{identity}-{region}-{tpu}"
        checkpoint_name = f"checkpoints/protein/{run_id}"
    else:
        additional_steps = ADDITIONAL_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{identity}-{region}"
        checkpoint_name = f"checkpoints/protein/{RUN_PREFIX}-{identity}"

    end_step = SOURCE_TRAIN_STEPS + additional_steps
    tags = [
        "protein",
        "exp199",
        "contacts-v1",
        f"sweep={subversion}",
        f"mixture={MIXTURE_KEY}",
        f"point={POINT_KEY}",
        f"augmentation={AUGMENTATION_KEY}",
        f"source={source_model.key}",
        f"source_run={source_model.run_id}",
        f"region={region}",
        f"tpu={tpu}",
        f"lr={LEARNING_RATE:g}",
        f"decay={DECAY:g}",
        "schedule=constant80-linear20",
        f"wd={WEIGHT_DECAY:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={additional_steps}",
        f"tokens={additional_steps * TOKENS_PER_STEP}",
        f"start_step={SOURCE_TRAIN_STEPS}",
        f"end_step={end_step}",
    ]
    if smoke:
        tags.append("smoke")
    return RunShape(
        additional_steps=additional_steps,
        end_step=end_step,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        tags=tags,
    )


def _apply_continuation_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    tpu: str,
    region: str,
    shape: RunShape,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        _regional_prefix_guard(ctx, region)
        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError(
                "continuation requires the region-local full-state checkpoint"
            )

        trainer = replace(
            pod.train_config.trainer,
            initialize_from=source_checkpoint_dir,
            max_eval_batches=1 if smoke else None,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                keep=(
                    [{"every": shape.permanent_checkpoint_every}]
                    if shape.permanent_checkpoint_every is not None
                    else []
                ),
            ),
        )
        data = pod.train_config.data
        data = replace(
            data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                key: replace(component, pack=True)
                for key, component in data.components.items()
            },
            block_cross_document_attention=True,
        )
        data = augment_every_example(data)

        if not ctx.is_fingerprint:
            batch_config = continuation_batch_fit(tpu, smoke=smoke)
            trainer = replace(
                trainer,
                per_device_parallelism=batch_config.per_device_parallelism,
                per_device_eval_parallelism=batch_config.per_device_parallelism,
            )

        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            # train_lm wires init_from here. Move it to TrainerConfig so the
            # checkpoint's model, optimizer, RNG, and absolute step all load.
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            # Trainer forces export hooks at completion; suppress intermediates.
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    *,
    subversion: str,
    region: str,
    tpu: str,
    source_model: SourceModel,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _validate_placement(tpu, region, smoke=smoke)
    shape = _run_shape(
        subversion=subversion,
        region=region,
        tpu=tpu,
        source_model=source_model,
        smoke=smoke,
    )
    batch_config = continuation_batch_fit(tpu, smoke=smoke)
    env = _training_env()

    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=LEARNING_RATE,
            min_lr_ratio=MIN_LR_RATIO,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            rewarmup=REWARMUP,
            decay=DECAY,
            cycle_length=[SOURCE_TRAIN_STEPS, shape.additional_steps],
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={
            afdb_cache(region): AFDB_WEIGHT,
            esm_cache(region): ESM_WEIGHT,
        },
        validation=[validation_cache(region)],
        init_from=source_checkpoint(region, source_model),
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.end_step,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(
            tpu,
            regions=singleton_region_list(region),
        ),
        tensor_parallel_size=batch_config.tensor_parallelism,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=f"{RUN_PREFIX}-{subversion}",
        tags=shape.tags,
        env_vars=env,
    )
    return _apply_continuation_overrides(
        step,
        tpu=tpu,
        region=region,
        shape=shape,
        smoke=smoke,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    region = _parse_region()
    tpu = _parse_tpu()
    source_model = _parse_source()
    _validate_launch_prefix(region)
    return build_run(
        subversion=_subversion(),
        region=region,
        tpu=tpu,
        source_model=source_model,
        smoke=_truthy_env("SMOKE"),
    )


if __name__ == "__main__":
    main()
