# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Cool down three selected exp232 checkpoints on CoreWeave from full state.

Each source restores model, AdamW or skip-step optimizer state, RNG, data
position, and absolute trainer step from the requested permanent checkpoint.
The logged learning rate at that step is then decayed linearly to exactly zero
over the original continued-training cooldown budget. Model, data mixture,
augmentation, packing, shuffle, global batch, and sequence length remain
continuous with the source run.

``SOURCE`` selects one checkpoint and ``CLUSTER``/``NODES`` select placement.
Production identity is independent of placement. Set ``SMOKE=1`` and optionally
``SMOKE_STEPS`` for a short, separately named full-state validation run.
"""

import os
import sys
from dataclasses import dataclass, replace
from datetime import timedelta

import click
import optax
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig, LrSchedule, LrScheduleContext
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, marin_temp_bucket, prefix_join

from experiments.exp232_sweep_cv1_decontam.gpu.exp232_sweep_cw import (
    EXPERIMENT_PREFIX,
    ClusterSpec,
    GpuBatchConfig,
    _parse_cluster,
    _run_exp232_train_job,
    _sweep_subversion,
    _truthy_env,
    _verify_decontaminated_cache_counts,
    afdb_cache,
    esm_cache,
    gpu_batch_fit,
    validation_cache,
)
from experiments.exp232_sweep_cv1_decontam.training_contract import (
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
    augment_amino_acids,
    augmentation_probability,
)

CANONICAL_MODULE = "experiments.exp232_sweep_cv1_decontam.gpu.exp232_cooldown_cw"
if __name__ == "__main__":
    sys.modules.setdefault(CANONICAL_MODULE, sys.modules[__name__])

RUN_PREFIX = "prot-exp232-cw-cv1-decontam-cooldown"
COOLDOWN_STEPS = round(0.20 * (3 * NUM_TRAIN_STEPS))
COOLDOWN_TOKENS = COOLDOWN_STEPS * TOKENS_PER_STEP
MIN_LR_RATIO = 0.0
AUGMENTATION_RAMP_STEPS = NUM_TRAIN_STEPS
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=30)
RECOVERY_DATA_SEED = 232


@dataclass(frozen=True)
class CooldownSource:
    key: str
    run_id: str
    version: str
    checkpoint_step: int
    backend: str
    mixture_key: str
    point_key: str
    afdb_weight: float
    esm_weight: float
    learning_rate: float
    weight_decay: float
    data_seed: int
    skip_bad_steps: bool

    @property
    def resume_step(self) -> int:
        """TrainerState.step restored by the zero-indexed checkpoint."""
        return self.checkpoint_step + 1


SOURCES = {
    source.key: source
    for source in (
        CooldownSource(
            key="m2-p06-a03",
            run_id=(
                "prot-exp232-cw-cv1-decontam-recover-a03-skipstep-"
                "m2-p06-srcpeak-augcont"
            ),
            version="2026.08.20.1",
            checkpoint_step=377_520,
            backend="cw",
            mixture_key="m2",
            point_key="p06",
            afdb_weight=AFDB_TOKENS / TARGET_TRAIN_TOKENS,
            esm_weight=ESM_TOKENS / TARGET_TRAIN_TOKENS,
            learning_rate=1e-3,
            weight_decay=0.2,
            data_seed=DATA_SEED,
            skip_bad_steps=True,
        ),
        CooldownSource(
            key="m1-p02-s01",
            run_id="prot-exp232-cw-cv1-decontam-train-s01-m1-p02-srcpeak-augcont",
            version="2026.08.18.1",
            checkpoint_step=348_480,
            backend="cw",
            mixture_key="m1",
            point_key="p02",
            afdb_weight=0.5,
            esm_weight=0.5,
            learning_rate=3.1623e-4,
            weight_decay=1.6,
            data_seed=DATA_SEED,
            skip_bad_steps=False,
        ),
        CooldownSource(
            key="m2-p06-lr005-trc",
            run_id=(
                "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-"
                "srcpeak-augcont-lr005-us-east1"
            ),
            version="2026.08.21.1",
            checkpoint_step=363_000,
            backend="trc",
            mixture_key="m2",
            point_key="p06",
            afdb_weight=AFDB_TOKENS / TARGET_TRAIN_TOKENS,
            esm_weight=ESM_TOKENS / TARGET_TRAIN_TOKENS,
            learning_rate=5e-5,
            weight_decay=0.2,
            data_seed=RECOVERY_DATA_SEED,
            skip_bad_steps=True,
        ),
    )
}


@LrSchedule.register_subclass("exp232_cooldown_linear")
@dataclass(frozen=True)
class CooldownLrSchedule(LrSchedule):
    """Absolute-step linear cooldown with inclusive start and zero endpoints."""

    __module__ = CANONICAL_MODULE
    resume_step: int
    cooldown_steps: int

    def build(self, ctx: LrScheduleContext):
        if ctx.min_lr != 0.0:
            raise ValueError("exp232 cooldown requires a zero final learning rate")
        if self.resume_step < 1 or self.cooldown_steps < 2:
            raise ValueError(
                "invalid cooldown schedule: "
                f"{self.resume_step=}, {self.cooldown_steps=}"
            )
        cooldown = optax.linear_schedule(
            ctx.learning_rate,
            0.0,
            transition_steps=self.cooldown_steps - 1,
        )
        return optax.join_schedules(
            schedules=(optax.constant_schedule(ctx.learning_rate), cooldown),
            boundaries=(self.resume_step,),
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


def _parse_source() -> CooldownSource:
    key = os.environ.get("SOURCE", "").strip().lower()
    try:
        return SOURCES[key]
    except KeyError:
        raise SystemExit(f"SOURCE must be one of: {', '.join(SOURCES)}") from None


def _parse_placement() -> tuple[str, ClusterSpec, int]:
    cluster, spec = _parse_cluster()
    raw = os.environ.get("NODES")
    if raw is None:
        raise SystemExit("missing required env var NODES")
    nodes = int(raw)
    allowed = {1, 2, 4, 8, 16}
    if nodes not in allowed:
        choices = ", ".join(str(value) for value in sorted(allowed))
        raise SystemExit(f"NODES must be one of {choices}, got {nodes}")
    return cluster, spec, nodes


def _training_env() -> dict[str, str]:
    expected = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
    missing = [key for key in expected if not os.environ.get(key)]
    if missing:
        raise ValueError(f"missing required variables: {', '.join(missing)}")
    unexpected = {
        key: os.environ[key]
        for key, value in expected.items()
        if os.environ[key] != value
    }
    if unexpected:
        raise ValueError(
            "cooldown W&B routing must be open-athena/MarinFold, got "
            + ", ".join(f"{key}={value!r}" for key, value in unexpected.items())
        )
    env = {**expected, "MARIN_PREFIX": EXPERIMENT_PREFIX}
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _validate_launch_prefix() -> None:
    configured = marin_prefix().rstrip("/")
    if configured != EXPERIMENT_PREFIX:
        raise ValueError(
            f"MARIN_PREFIX must be exactly {EXPERIMENT_PREFIX!r}, got {configured!r}"
        )


def source_checkpoint(source: CooldownSource) -> ArtifactStep[LevanterCheckpoint]:
    origin = prefix_join(
        EXPERIMENT_PREFIX,
        f"checkpoints/protein/{source.run_id}/{source.version}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        (
            "checkpoints/protein/exp232-cw-cooldown-source/"
            f"{source.run_id}/step-{source.checkpoint_step}"
        ),
        source.version,
        source=origin,
        kind=LevanterCheckpoint,
        config={
            "source_run": source.run_id,
            "source_version": source.version,
            "checkpoint_step": source.checkpoint_step,
            "source_backend": source.backend,
            "full_state": True,
        },
    )


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
            f"{RUN_PREFIX}-smoke-{subversion}-{source.key}-from{source.checkpoint_step}-"
            f"{cluster}-{spec.gpu_variant.lower()}-n{nodes}"
        )
    else:
        cooldown_steps = COOLDOWN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{subversion}-{source.key}-from{source.checkpoint_step}"

    end_step = source.resume_step + cooldown_steps
    tags = [
        "protein",
        "exp232",
        "contacts-v1",
        "decontaminated",
        "cooldown",
        f"sweep={subversion}",
        f"source={source.key}",
        f"source_version={source.version}",
        f"source_backend={source.backend}",
        f"source_checkpoint_step={source.checkpoint_step}",
        f"mixture={source.mixture_key}",
        f"point={source.point_key}",
        "augmentation=augcont",
        "augmentation_schedule=exp232-linear-global-clamp100",
        f"augmentation_resume_probability={augmentation_probability(source.resume_step, AUGMENTATION_RAMP_STEPS):.12f}",
        f"start_lr={source.learning_rate:g}",
        "final_lr=0",
        f"wd={source.weight_decay:g}",
        f"data_seed={source.data_seed}",
        f"skip_bad_steps={str(source.skip_bad_steps).lower()}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={cooldown_steps}",
        f"tokens={cooldown_steps * TOKENS_PER_STEP}",
        f"source_documents={TARGET_TRAIN_DOCUMENTS}",
        f"source_tokens={TARGET_TRAIN_TOKENS}",
        f"start_step={source.resume_step}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        "schedule=linear-inclusive-to-zero",
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
    return AdamConfig(
        learning_rate=source.learning_rate,
        weight_decay=source.weight_decay,
        warmup=0.0,
        rewarmup=0.0,
        decay=1.0,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule=CooldownLrSchedule(source.resume_step, shape.cooldown_steps),
        skip_bad_steps=source.skip_bad_steps,
    )


def _apply_training_overrides(
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
            and execution_prefix != EXPERIMENT_PREFIX
            and not execution_prefix.startswith(f"{EXPERIMENT_PREFIX}/")
        ):
            raise ValueError(
                f"execution prefix {ctx.prefix!r} is outside {EXPERIMENT_PREFIX!r}"
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
            allow_partial_checkpoint=False,
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
        data = augment_amino_acids(data, AUGMENTATION_RAMP_STEPS)

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
            data_seed=source.data_seed,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config, run=_run_exp232_train_job)


def build_run(
    source: CooldownSource,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _verify_decontaminated_cache_counts()
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
    return _apply_training_overrides(
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
    source = _parse_source()
    cluster, spec, nodes = _parse_placement()
    smoke = _truthy_env("SMOKE")
    _validate_launch_prefix()
    return build_run(
        source,
        subversion=_sweep_subversion(),
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
