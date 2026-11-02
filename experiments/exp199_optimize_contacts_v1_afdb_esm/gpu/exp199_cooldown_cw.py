# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cool down an exp199 CoreWeave continuation from full trainer state.

The continuation ``prot-exp199-cw-cv1-cont-s02-m1-p06-srcaug-aug100`` was halted
before reaching its own scheduled cooldown. This script instead anneals it from
an exact permanent checkpoint: it restores model, AdamW, RNG, data-position and
absolute-step state, then decays the learning rate linearly from the restored
peak to zero across the whole added cycle, with no warmup and no stable phase.

Everything else -- model, tokenizer, mixture, augmentation, packing, shuffle,
global batch and sequence length -- is inherited unchanged from the continuation
so the anneal differs from its source in the learning-rate schedule alone.

``CLUSTER`` and ``NODES`` select placement without entering production identity.
Set ``SMOKE=1`` and optionally ``SMOKE_STEPS`` for a short, separately named
full-state validation run. Omit ``--run`` to preview the lowered plan.
"""

import os
from dataclasses import dataclass, replace

import click
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, marin_temp_bucket, prefix_join

from exp199_continue_cw import (
    CONTINUATION_EXPERIMENT_PREFIX,
    PRODUCTION_GANG_NODES,
    TEMPORARY_CHECKPOINT_INTERVAL,
    augment_every_example,
)
from exp199_sweep_cw import (
    DATA_SEED,
    GLOBAL_BATCH_SIZE,
    MODEL_CONFIG,
    MODEL_PARAMS,
    PERMANENT_CHECKPOINT_EVERY,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    TOKENS_PER_STEP,
    WANDB_WATCH,
    ClusterSpec,
    GpuBatchConfig,
    _parse_cluster,
    _parse_nodes,
    _sweep_subversion,
    _truthy_env,
    afdb_cache,
    esm_cache,
    gpu_batch_fit,
    validation_cache,
)

RUN_PREFIX = "prot-exp199-cw-cv1-p06-cool"

# The anneal is one added cycle that is entirely decay: no rewarmup, no stable
# phase, and a floor of exactly zero. WARMUP applies only to the notional first
# cycle, which covers steps this run restores past and never trains.
COOLDOWN_TRAIN_STEPS = 29_040
MIN_LR_RATIO = 0.0
WARMUP = 0.0
REWARMUP = 0.0
DECAY = 1.0
LR_SCHEDULE = "linear"


@dataclass(frozen=True)
class CooldownSource:
    """The continuation checkpoint this run anneals from."""

    run_id: str
    version: str
    checkpoint_step: int
    mixture_key: str
    point_key: str
    augmentation: str
    afdb_weight: float
    esm_weight: float
    learning_rate: float
    weight_decay: float

    @property
    def resume_step(self) -> int:
        """TrainerState.step restored by the zero-indexed checkpoint."""
        return self.checkpoint_step + 1


# step-261360 is the latest permanent checkpoint the continuation wrote and was
# verified complete on S3 (72 objects, 16.44 GiB) at 2026-08-14T15:37:40Z.
COOLDOWN_SOURCE = CooldownSource(
    run_id="prot-exp199-cw-cv1-cont-s02-m1-p06-srcaug-aug100",
    version="2026.08.10.2",
    checkpoint_step=261_360,
    mixture_key="m1",
    point_key="p06",
    augmentation="aug100",
    afdb_weight=0.5,
    esm_weight=0.5,
    learning_rate=1e-3,
    weight_decay=0.2,
)


def _parse_placement(*, smoke: bool) -> tuple[str, ClusterSpec, int]:
    cluster, spec = _parse_cluster()
    nodes = _parse_nodes(smoke=smoke)
    if not smoke:
        required_nodes = PRODUCTION_GANG_NODES[spec.gpu_variant]
        if nodes != required_nodes:
            raise SystemExit(
                "production cooldown supports exactly two gang profiles: "
                "8 H100 nodes or 8 GB200 nodes; "
                f"got {spec.gpu_variant} with NODES={nodes}"
            )
    return cluster, spec, nodes


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


def source_checkpoint(source: CooldownSource) -> ArtifactStep[LevanterCheckpoint]:
    """Adopt the continuation's permanent checkpoint as this run's initializer."""
    origin = prefix_join(
        CONTINUATION_EXPERIMENT_PREFIX,
        f"checkpoints/protein/{source.run_id}/{source.version}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        (
            "checkpoints/protein/exp199-cw-cooldown-source/"
            f"{source.run_id}/step-{source.checkpoint_step}"
        ),
        source.version,
        source=origin,
        kind=LevanterCheckpoint,
        config={
            "source_run": source.run_id,
            "source_version": source.version,
            "checkpoint_step": source.checkpoint_step,
        },
    )


@dataclass(frozen=True)
class RunShape:
    cooldown_steps: int
    end_step: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    wandb_group: str
    tags: list[str]


def _run_shape(
    source: CooldownSource,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> RunShape:
    if smoke:
        cooldown_steps = int(os.environ.get("SMOKE_STEPS", "20"))
        if cooldown_steps < 10:
            raise ValueError("SMOKE_STEPS must be at least 10 to expose the LR shape")
        steps_per_eval = cooldown_steps
        permanent_checkpoint_every = None
        run_id = (
            f"{RUN_PREFIX}-smoke-{subversion}-{cluster}-"
            f"{spec.gpu_variant.lower()}-n{nodes}"
        )
    else:
        cooldown_steps = COOLDOWN_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{subversion}"

    end_step = source.resume_step + cooldown_steps
    cooldown_tokens = cooldown_steps * TOKENS_PER_STEP
    cumulative_tokens = end_step * TOKENS_PER_STEP
    tags = [
        "protein",
        "exp199",
        "contacts-v1",
        "cooldown",
        f"sweep={subversion}",
        f"source_run={source.run_id}",
        f"source_version={source.version}",
        f"mixture={source.mixture_key}",
        f"point={source.point_key}",
        f"augmentation={source.augmentation}",
        f"lr={source.learning_rate:g}",
        f"wd={source.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={cooldown_steps}",
        f"tokens={cooldown_tokens}",
        f"source_checkpoint_step={source.checkpoint_step}",
        f"start_step={source.resume_step}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        f"cumulative_tokens={cumulative_tokens}",
        "schedule=linear-to-zero",
        f"initialization=checkpoint-step-{source.checkpoint_step}",
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
        cooldown_steps=cooldown_steps,
        end_step=end_step,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=f"checkpoints/protein/{run_id}",
        wandb_group=f"{RUN_PREFIX}-{subversion}",
        tags=tags,
    )


def cooldown_optimizer(source: CooldownSource, shape: RunShape) -> AdamConfig:
    """Hold nothing: decay the restored peak linearly to zero across the cycle."""
    return AdamConfig(
        learning_rate=source.learning_rate,
        weight_decay=source.weight_decay,
        warmup=WARMUP,
        rewarmup=REWARMUP,
        decay=DECAY,
        cycle_length=[source.resume_step, shape.cooldown_steps],
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule=LR_SCHEDULE,
    )


def _apply_cooldown_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    source: CooldownSource,
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
            raise ValueError("cooldown requires the source checkpoint dependency")
        exact_checkpoint = (
            prefix_join(source_checkpoint_dir, f"step-{source.checkpoint_step}")
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
            # Move the exact source into TrainerConfig so model, optimizer,
            # RNG, data position, and absolute step all load together.
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    source: CooldownSource,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    shape = _run_shape(
        source,
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
        optimizer=cooldown_optimizer(source, shape),
        datasets={
            afdb_cache(): source.afdb_weight,
            esm_cache(): source.esm_weight,
        },
        validation=[validation_cache()],
        init_from=source_checkpoint(source),
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
    return _apply_cooldown_overrides(
        step,
        source=source,
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
    cluster, spec, nodes = _parse_placement(smoke=smoke)
    _validate_launch_prefix()
    return build_run(
        COOLDOWN_SOURCE,
        subversion=_sweep_subversion(),
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
