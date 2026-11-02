# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover stable training for a diverged exp232 selected-training run.

Both selected trials diverged mid-training at unrelated global steps:

``m2-p06`` (lr 1e-3)
    Diverged at step 228783: loss 2.887 -> 3.327 -> 6.282 -> 8.274 -> 9.485 over
    four steps, and the next evaluation moved 3.0597 -> 3.8248. Over the 283
    clean steps before it, loss was mean 2.8988 with standard deviation 0.0309,
    putting the first bad step at +13.9 sigma and the rest at +100 to +214.

``m1-p02`` (lr 3.1623e-4)
    Diverged at step 357489: loss 3.102 -> 7.442 in one step, peaking at 8.928.

``SOURCE`` selects which one to recover. The entry point restarts from that
run's last permanent checkpoint strictly before its divergence and adds a
mitigation, leaving the original run's W&B history and checkpoints untouched and
read-only. It writes to its own experiment prefix, run id, and checkpoint root
because it is exploratory.

Mitigations, selected with ``MITIGATION``:

``skipstep`` (default)
    Enable Levanter's skip-step optimizer, which keeps a rolling 128-step window
    of losses and gradient norms and zeroes the update whenever the current step
    exceeds mean + 6 sigma. Note the observed limit: on the m2-p06 recovery it
    caught the first 18 bad steps but its threshold, computed over a window that
    now contained the anomalies, inflated from 0.8 to 232 and let later bad
    steps through. It contains isolated bad batches, not sustained excursions.

``dataseed``
    Keep the optimizer as the original run had it and change the data seed
    instead, so the post-restore batch stream no longer replays the same
    sequence.

``both``
    Skip-step protection plus a fresh data seed.

The LR schedule, augmentation ramp, batch size, model, and end step stay exactly
as the original run defined them, so a given global step keeps its original LR
and augmentation probability. ``CLUSTER`` and ``NODES`` select placement without
entering run identity. Omit ``--run`` to preview the lowered plan.
"""

import os
import sys
from dataclasses import dataclass, replace
from datetime import timedelta

import click
import optax
from exp232_sweep_cw import (
    ClusterSpec,
    GpuBatchConfig,
    _parse_cluster,
    _parse_nodes,
    _run_exp232_train_job,
    _verify_decontaminated_cache_counts,
    afdb_cache,
    esm_cache,
    gpu_batch_fit,
    validation_cache,
)
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig, LrSchedule, LrScheduleContext
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, prefix_join

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
    TARGET_TRAIN_TOKENS,
    TOKENS_PER_STEP,
    WANDB_WATCH,
    augment_amino_acids,
    augmentation_probability,
)

RUN_PREFIX = "prot-exp232-cw-cv1-decontam-recover"
RECOVER_EXPERIMENT_PREFIX = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam"

# The diverged production runs, read only. Their outputs are never written to.
SOURCE_EXPERIMENT_PREFIX = RECOVER_EXPERIMENT_PREFIX


@dataclass(frozen=True)
class RecoverySource:
    """One diverged production run and the state to restart it from.

    ``resume_checkpoint_step`` is the last permanent checkpoint strictly before
    ``divergence_step``. Permanent checkpoints land every 14520 steps and the
    temporary ones in between rotate away, so this is the closest recoverable
    full state. The learning rate and weight decay restate the original trial so
    a given global step keeps the learning rate it always had; source of truth
    is ``exp232_train_cw.SOURCE_MODELS``.
    """

    key: str
    run_id: str
    version: str
    resume_checkpoint_step: int
    divergence_step: int
    learning_rate: float
    weight_decay: float
    mixture_key: str
    point_key: str


RECOVERY_SOURCES = {
    "m2-p06": RecoverySource(
        key="m2-p06",
        run_id="prot-exp232-cw-cv1-decontam-train-s01-m2-p06-srcpeak-augcont",
        version="2026.08.18.1",
        resume_checkpoint_step=217_800,
        divergence_step=228_783,
        learning_rate=1e-3,
        weight_decay=0.2,
        mixture_key="m2",
        point_key="p06",
    ),
    "m1-p02": RecoverySource(
        key="m1-p02",
        run_id="prot-exp232-cw-cv1-decontam-train-s01-m1-p02-srcpeak-augcont",
        version="2026.08.18.1",
        resume_checkpoint_step=348_480,
        divergence_step=357_489,
        learning_rate=3.1623e-4,
        weight_decay=1.6,
        mixture_key="m1",
        point_key="p02",
    ),
}

ORIGINAL_RESUME_STEP = 116_161
ADDITIONAL_TRAIN_STEPS = 3 * NUM_TRAIN_STEPS
MIN_LR_RATIO = 0.0
WARMUP = 0.1
REWARMUP = 0.0
DECAY = 0.2
AUGMENTATION_KEY = "augcont"
AUGMENTATION_RAMP_STEPS = NUM_TRAIN_STEPS
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=30)

# Levanter's default skip-step window and threshold. See the module docstring
# for why 6 sigma separates this run's normal variation from its divergence.
SKIP_STEP_ROLLING_INTERVAL = 128
SKIP_STEP_SIGMA_FACTOR = 6.0

MITIGATIONS = ("skipstep", "dataseed", "both")
RECOVERY_DATA_SEED = 232


# Iris serializes the experiment graph before the training worker imports it.
# Give this one-file entrypoint a canonical module identity even when launched
# as ``python exp232_train_cw_recover.py`` so Draccus can resolve the choice.
if __name__ == "__main__":
    sys.modules.setdefault("exp232_train_cw_recover", sys.modules[__name__])


@LrSchedule.register_subclass("linear_inclusive_recover")
@dataclass(frozen=True)
class InclusiveLinearLrSchedule(LrSchedule):
    """Linearly decay to the minimum on the last executed decay update.

    Behaviourally identical to the production run's ``linear_inclusive``. It is
    redeclared under its own registry name and module so this script never has
    to import the production entry point on the worker.
    """

    __module__ = "exp232_train_cw_recover"

    def build(self, ctx: LrScheduleContext):
        if ctx.decay_steps < 2:
            raise ValueError("inclusive linear decay requires at least two updates")
        return optax.linear_schedule(
            ctx.learning_rate,
            ctx.min_lr,
            transition_steps=ctx.decay_steps - 1,
        )


LR_SCHEDULE = InclusiveLinearLrSchedule()


def _parse_source() -> RecoverySource:
    # Defaults to m2-p06 so an in-flight dispatch submitted before this script
    # took a second source keeps working if its root driver re-executes.
    source = os.environ.get("SOURCE", "m2-p06").strip().lower()
    try:
        return RECOVERY_SOURCES[source]
    except KeyError:
        raise SystemExit(
            f"SOURCE must be one of: {', '.join(RECOVERY_SOURCES)}"
        ) from None


def _parse_mitigation() -> str:
    mitigation = os.environ.get("MITIGATION", "skipstep").strip().lower()
    if mitigation not in MITIGATIONS:
        raise SystemExit(f"MITIGATION must be one of: {', '.join(MITIGATIONS)}")
    return mitigation


def _parse_attempt() -> str:
    attempt = os.environ.get("ATTEMPT", "a01").strip().lower()
    if not attempt.isalnum():
        raise SystemExit(f"ATTEMPT must be alphanumeric, got {attempt!r}")
    return attempt


def _training_env() -> dict[str, str]:
    expected_wandb = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
    missing = [key for key in expected_wandb if not os.environ.get(key)]
    if missing:
        raise ValueError(
            f"missing required environment variables: {', '.join(missing)}"
        )
    unexpected = {
        key: os.environ[key]
        for key, expected in expected_wandb.items()
        if os.environ[key] != expected
    }
    if unexpected:
        raise ValueError(
            "training W&B routing must be open-athena/MarinFold, got "
            + ", ".join(f"{key}={value!r}" for key, value in unexpected.items())
        )
    env = {
        "MARIN_PREFIX": RECOVER_EXPERIMENT_PREFIX,
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _validate_launch_prefix() -> None:
    configured = marin_prefix().rstrip("/")
    if configured != RECOVER_EXPERIMENT_PREFIX:
        raise ValueError(
            "MARIN_PREFIX must be exactly "
            f"{RECOVER_EXPERIMENT_PREFIX!r}, got {configured!r}"
        )


def source_checkpoint(source: RecoverySource) -> ArtifactStep[LevanterCheckpoint]:
    """Adopt the diverged run's pre-divergence permanent checkpoint, read only."""
    origin = prefix_join(
        SOURCE_EXPERIMENT_PREFIX,
        f"checkpoints/protein/{source.run_id}/{source.version}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        (
            "checkpoints/protein/exp232-cw-recover-source/"
            f"{source.run_id}/step-{source.resume_checkpoint_step}"
        ),
        source.version,
        source=origin,
        kind=LevanterCheckpoint,
        config={
            "source_run": source.run_id,
            "source_version": source.version,
            "checkpoint_step": source.resume_checkpoint_step,
            "divergence_step": source.divergence_step,
        },
    )


@dataclass(frozen=True)
class RunShape:
    end_step: int
    run_id: str
    checkpoint_name: str
    wandb_group: str
    data_seed: int
    skip_bad_steps: bool
    tags: list[str]


def _run_shape(
    *,
    source: RecoverySource,
    mitigation: str,
    attempt: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
) -> RunShape:
    end_step = ORIGINAL_RESUME_STEP + ADDITIONAL_TRAIN_STEPS
    skip_bad_steps = mitigation in ("skipstep", "both")
    data_seed = RECOVERY_DATA_SEED if mitigation in ("dataseed", "both") else DATA_SEED
    run_id = (
        f"{RUN_PREFIX}-{attempt}-{mitigation}-{source.key}-srcpeak-{AUGMENTATION_KEY}"
    )
    resume_step = source.resume_checkpoint_step + 1
    tags = [
        "protein",
        "exp232",
        "contacts-v1",
        "decontaminated",
        "selected-training",
        "divergence-recovery",
        f"attempt={attempt}",
        f"mitigation={mitigation}",
        f"skip_bad_steps={str(skip_bad_steps).lower()}",
        f"skip_step_rolling_interval={SKIP_STEP_ROLLING_INTERVAL}",
        f"skip_step_sigma_factor={SKIP_STEP_SIGMA_FACTOR:g}",
        f"data_seed={data_seed}",
        # W&B caps tags at 64 characters, so carry the distinguishing suffix of
        # the source run rather than its full id. The full id is recorded in the
        # adopted checkpoint's config and in the sweep database.
        f"recovered_from_run={source.run_id.removeprefix('prot-exp232-cw-cv1-decontam-')}",
        f"recovered_from_step={source.resume_checkpoint_step}",
        f"divergence_step={source.divergence_step}",
        f"mixture={source.mixture_key}",
        f"point={source.point_key}",
        f"augmentation={AUGMENTATION_KEY}",
        "augmentation_schedule=exp232-linear-global-clamp100",
        (
            "augmentation_resume_probability="
            f"{augmentation_probability(resume_step, AUGMENTATION_RAMP_STEPS):.12f}"
        ),
        f"augmentation_full_step={AUGMENTATION_RAMP_STEPS - 1}",
        f"lr={source.learning_rate:g}",
        f"final_lr={source.learning_rate * MIN_LR_RATIO:g}",
        f"wd={source.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"start_step={resume_step}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        f"cumulative_tokens={end_step * TOKENS_PER_STEP}",
        "schedule=constant80-linear20-zero",
        f"initialization=checkpoint-step-{source.resume_checkpoint_step}",
        f"cluster={cluster}",
        f"gpu={spec.gpu_variant}",
        f"nodes={nodes}",
    ]
    # W&B rejects the whole run if any tag exceeds 64 characters. Fail here, in
    # the lowered plan, instead of after the gang has already been scheduled.
    if oversized := [tag for tag in tags if not 1 <= len(tag) <= 64]:
        raise ValueError(
            "W&B tags must be 1 to 64 characters, got "
            + ", ".join(f"{tag!r} ({len(tag)})" for tag in oversized)
        )
    return RunShape(
        end_step=end_step,
        run_id=run_id,
        checkpoint_name=f"checkpoints/protein/{run_id}",
        wandb_group=f"{RUN_PREFIX}-{attempt}",
        data_seed=data_seed,
        skip_bad_steps=skip_bad_steps,
        tags=tags,
    )


def _apply_training_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    source: RecoverySource,
    shape: RunShape,
    batch: GpuBatchConfig,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        execution_prefix = ctx.prefix.rstrip("/")
        if (
            not ctx.is_fingerprint
            and execution_prefix != RECOVER_EXPERIMENT_PREFIX
            and not execution_prefix.startswith(f"{RECOVER_EXPERIMENT_PREFIX}/")
        ):
            raise ValueError(
                f"execution prefix {ctx.prefix!r} is outside "
                f"{RECOVER_EXPERIMENT_PREFIX!r}"
            )

        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError("recovery requires the source checkpoint dependency")
        exact_checkpoint = (
            prefix_join(source_checkpoint_dir, f"step-{source.resume_checkpoint_step}")
            if source_checkpoint_dir is not None
            else None
        )

        trainer = replace(
            pod.train_config.trainer,
            initialize_from=exact_checkpoint,
            # Skip-step wraps the optimizer, so its state carries five extra
            # bookkeeping arrays (_skipstep_losses, _skipstep_grad_norms,
            # _skipstep_valid_mask, _skipstep_current_idx, _skipstep_count) that
            # a checkpoint written without it cannot contain. Those five are the
            # only additions -- model weights and the Adam mu/nu moments restore
            # exactly -- and their fresh init is the empty rolling window the
            # optimizer expects, so a partial restore is the intended path here.
            allow_partial_checkpoint=shape.skip_bad_steps,
            max_eval_batches=None,
            watch=WANDB_WATCH,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL,
                keep=[{"every": PERMANENT_CHECKPOINT_EVERY}],
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
            trainer = replace(
                trainer,
                per_device_parallelism=batch.per_device_parallelism,
                per_device_eval_parallelism=batch.per_device_parallelism,
            )

        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=shape.data_seed,
            # Put the exact full state on TrainerConfig so model, optimizer,
            # RNG, data position, and absolute step restore together.
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config, run=_run_exp232_train_job)


def build_run(
    *,
    source: RecoverySource,
    mitigation: str,
    attempt: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
) -> ArtifactStep[LevanterCheckpoint]:
    _verify_decontaminated_cache_counts()
    shape = _run_shape(
        source=source,
        mitigation=mitigation,
        attempt=attempt,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
    )
    batch = gpu_batch_fit(spec, nodes=nodes, smoke=False)
    env = _training_env()

    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=source.learning_rate,
            weight_decay=source.weight_decay,
            warmup=WARMUP,
            rewarmup=REWARMUP,
            decay=DECAY,
            cycle_length=[ORIGINAL_RESUME_STEP, ADDITIONAL_TRAIN_STEPS],
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
            # True selects Levanter's default SkipStepConfig, which is exactly
            # rolling_interval_length=128 and sigma_factor=6.0. Passing the bool
            # keeps the union field trivially serializable.
            skip_bad_steps=shape.skip_bad_steps,
        ),
        datasets={
            afdb_cache(): AFDB_TOKENS / TARGET_TRAIN_TOKENS,
            esm_cache(): ESM_TOKENS / TARGET_TRAIN_TOKENS,
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
        steps_per_eval=STEPS_PER_EVAL,
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=shape.wandb_group,
        tags=shape.tags,
        env_vars=env,
    )
    return _apply_training_overrides(step, source=source, shape=shape, batch=batch)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    source = _parse_source()
    mitigation = _parse_mitigation()
    attempt = _parse_attempt()
    cluster, spec = _parse_cluster()
    nodes = _parse_nodes(smoke=False)
    _validate_launch_prefix()
    return build_run(
        source=source,
        mitigation=mitigation,
        attempt=attempt,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
    )


if __name__ == "__main__":
    main()
