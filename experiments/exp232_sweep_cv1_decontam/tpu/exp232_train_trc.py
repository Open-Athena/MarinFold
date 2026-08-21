# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train three exp232 LR-recovery variants on region-local TRC resources.

Each run strictly restores the full Levanter ``step-333960`` state from the
same region: model, AdamW and skip-step state, RNG, data position, and absolute
trainer step. It changes the data seed, linearly lowers the source peak LR over
the first 5% of remaining training, holds the selected LR, then follows the
original final 20% cooldown to zero. Augmentation remains on its original global
schedule and is therefore continuously clamped at 100%.

``VARIANT`` selects ``lr050``, ``lr010``, or ``lr005``. ``REGION`` and ``TPU``
select a regional replica without changing the logical recipe. Set ``SMOKE=1``
for a short full-state validation run after all regional inputs are verified.
"""

import math
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import timedelta

import click
import optax
from experiments.exp232_sweep_cv1_decontam.exp232_sweep import (
    AFDB_DOCUMENTS,
    AFDB_TOKENS,
    CACHE_VERSION,
    ESM_DOCUMENTS,
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
    VALIDATION_CACHE_VERSION,
    WANDB_WATCH,
    _existing_cache,
    _truthy_env,
    augment_amino_acids,
    augmentation_probability,
)
from fray.types import (
    ResourceConfig,
    get_tpu_topology,
    tpu_family,
    tpu_hbm_capacity_bytes,
)
from levanter.optim.config import AdamConfig, LrSchedule, LrScheduleContext
from marin.execution.build_context import current_build_context
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.rl.placement import marin_prefix_for_region, singleton_region_list
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, prefix_join

CANONICAL_MODULE = "experiments.exp232_sweep_cv1_decontam.tpu.exp232_train_trc"
if __name__ == "__main__":
    sys.modules.setdefault(CANONICAL_MODULE, sys.modules[__name__])

RUN_PREFIX = "prot-exp232-trc-cv1-decontam-train"
REGIONAL_EXPERIMENT_RELATIVE = "protein-structure/MarinFold/exp232_train_trc"
SOURCE_SEED_NAMESPACE = "checkpoints/protein/exp232-trc-init"
SOURCE_SEED_VERSION = "2026.08.21.1"
SOURCE_RUN_ID = (
    "prot-exp232-cw-cv1-decontam-recover-a03-skipstep-m2-p06-srcpeak-augcont"
)
SOURCE_CHECKPOINT_STEP = 333_960
RESUME_STEP = SOURCE_CHECKPOINT_STEP + 1

ORIGINAL_RESUME_STEP = 116_161
ADDITIONAL_TRAIN_STEPS = 3 * NUM_TRAIN_STEPS
END_STEP = ORIGINAL_RESUME_STEP + ADDITIONAL_TRAIN_STEPS
FINAL_CHECKPOINT_STEP = END_STEP - 1
REMAINING_STEPS = END_STEP - RESUME_STEP

PEAK_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.2
INITIAL_LR_TRANSITION_STEPS = round(0.05 * REMAINING_STEPS)
FINAL_COOLDOWN_STEPS = round(0.20 * ADDITIONAL_TRAIN_STEPS)
INITIAL_LR_HOLD_START = RESUME_STEP + INITIAL_LR_TRANSITION_STEPS
FINAL_COOLDOWN_START = END_STEP - FINAL_COOLDOWN_STEPS
HOLD_STEPS = FINAL_COOLDOWN_START - INITIAL_LR_HOLD_START

RECOVERY_DATA_SEED = 232
AUGMENTATION_RAMP_STEPS = NUM_TRAIN_STEPS
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=30)

SUPPORTED_REGIONS = ("europe-west4", "us-east1", "us-east5", "us-west4")
ALLOWED_FAMILIES_BY_REGION = {
    "europe-west4": {"v5e", "v6e"},
    "us-east1": {"v6e"},
    "us-east5": {"v5p", "v6e"},
    "us-west4": {"v5e"},
}
CORRECTION_FACTORS = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45}


@dataclass(frozen=True)
class Variant:
    key: str
    target_ratio: float

    @property
    def target_learning_rate(self) -> float:
        return PEAK_LEARNING_RATE * self.target_ratio


VARIANTS = {
    variant.key: variant
    for variant in (
        Variant("lr050", 0.50),
        Variant("lr010", 0.10),
        Variant("lr005", 0.05),
    )
}


@LrSchedule.register_subclass("exp232_trc_recovery")
@dataclass(frozen=True)
class Exp232RecoveryLrSchedule(LrSchedule):
    """Absolute-step LR schedule for the post-step-333960 recovery."""

    __module__ = CANONICAL_MODULE
    target_ratio: float

    def build(self, ctx: LrScheduleContext):
        if ctx.learning_rate != PEAK_LEARNING_RATE or ctx.min_lr != 0.0:
            raise ValueError(
                "exp232 recovery schedule requires the source peak LR and zero final LR"
            )
        target_lr = ctx.learning_rate * self.target_ratio
        initial_transition = optax.linear_schedule(
            ctx.learning_rate,
            target_lr,
            transition_steps=INITIAL_LR_TRANSITION_STEPS - 1,
        )
        final_cooldown = optax.linear_schedule(
            target_lr,
            0.0,
            transition_steps=FINAL_COOLDOWN_STEPS - 1,
        )
        return optax.join_schedules(
            schedules=(
                optax.constant_schedule(ctx.learning_rate),
                initial_transition,
                optax.constant_schedule(target_lr),
                final_cooldown,
            ),
            boundaries=(
                RESUME_STEP,
                INITIAL_LR_HOLD_START,
                FINAL_COOLDOWN_START,
            ),
        )


@dataclass(frozen=True)
class TpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def _batch_memory_bytes(batch_size: int, correction_factor: float) -> int:
    parameter_bytes = MODEL_PARAMS * 4
    optimizer_bytes = MODEL_PARAMS * 8
    hidden = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 2
    attention = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 4 * 2
    mlp = batch_size * SEQ_LEN * MODEL_CONFIG.intermediate_dim * 2
    saved_layers = max(math.floor(MODEL_CONFIG.num_layers * 0.75), 4)
    activation_bytes = (hidden + attention + mlp) * saved_layers
    return math.ceil(
        (parameter_bytes + optimizer_bytes + activation_bytes) * correction_factor
    )


def batch_fit(tpu: str) -> TpuBatchConfig:
    family = tpu_family(tpu)
    try:
        correction_factor = CORRECTION_FACTORS[family]
    except KeyError:
        raise ValueError(f"unsupported TPU family {family!r}") from None
    chips = get_tpu_topology(tpu).chip_count
    data_parallelism = math.gcd(GLOBAL_BATCH_SIZE, chips)
    tensor_parallelism = chips // data_parallelism
    batch_bytes = _batch_memory_bytes(GLOBAL_BATCH_SIZE, correction_factor)
    capacity_bytes = tpu_hbm_capacity_bytes(tpu)
    full_per_device_batch = GLOBAL_BATCH_SIZE // data_parallelism

    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism:
            continue
        microbatch_size = per_device_parallelism * data_parallelism
        microbatch_bytes = math.ceil(batch_bytes * microbatch_size / GLOBAL_BATCH_SIZE)
        if microbatch_bytes <= capacity_bytes:
            return TpuBatchConfig(
                data_parallelism=data_parallelism,
                tensor_parallelism=tensor_parallelism,
                per_device_parallelism=per_device_parallelism,
                gradient_accumulation=GLOBAL_BATCH_SIZE // microbatch_size,
            )
    raise ValueError(f"global batch {GLOBAL_BATCH_SIZE} does not fit on {tpu}")


def regional_experiment_prefix(region: str) -> str:
    return prefix_join(marin_prefix_for_region(region), REGIONAL_EXPERIMENT_RELATIVE)


def _parse_region() -> str:
    region = os.environ.get("REGION", "").strip().lower()
    if region not in SUPPORTED_REGIONS:
        raise SystemExit(f"REGION must be one of: {', '.join(SUPPORTED_REGIONS)}")
    return region


def _parse_tpu() -> str:
    tpu = os.environ.get("TPU", "").strip().lower()
    if not tpu:
        raise SystemExit("missing required env var TPU")
    get_tpu_topology(tpu)
    return tpu


def _parse_variant() -> Variant:
    key = os.environ.get("VARIANT", "").strip().lower()
    try:
        return VARIANTS[key]
    except KeyError:
        raise SystemExit(f"VARIANT must be one of: {', '.join(VARIANTS)}") from None


def _validate_placement(tpu: str, region: str, *, smoke: bool) -> None:
    family = tpu_family(tpu)
    if family not in ALLOWED_FAMILIES_BY_REGION[region]:
        raise ValueError(f"{tpu} is not available in exp232 region {region}")
    if smoke:
        return
    chips = get_tpu_topology(tpu).chip_count
    low, high = (16, 256) if family == "v5p" else (32, 512)
    if not low <= chips <= high:
        raise ValueError(
            f"production {tpu} has {chips} chips; expected {low}--{high} for {family}"
        )


def _subversion() -> str:
    context = current_build_context()
    if context is None:
        raise ValueError("exp232 TRC training must be built under --version")
    if context.versions.overrides:
        raise ValueError("--override is not supported because sNN is run identity")
    match = re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.(\d+)", context.versions.default)
    if match is None:
        raise ValueError("--version must be a suffixed CalVer such as 2026.08.21.1")
    suffix = int(match.group(1))
    if not 1 <= suffix <= 99:
        raise ValueError(f"CalVer suffix must be in 1--99, got {suffix}")
    return f"s{suffix:02d}"


def _training_env(region: str) -> dict[str, str]:
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
            "training W&B routing must be open-athena/MarinFold, got "
            + ", ".join(f"{key}={value!r}" for key, value in unexpected.items())
        )
    env = {
        **expected,
        "MARIN_PREFIX": regional_experiment_prefix(region),
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _validate_launch_prefix(region: str) -> None:
    expected = regional_experiment_prefix(region).rstrip("/")
    configured = marin_prefix().rstrip("/")
    if configured != expected:
        raise ValueError(
            f"MARIN_PREFIX must be exactly {expected!r}, got {configured!r}"
        )


def _regional_prefix_guard(ctx, region: str) -> None:
    expected = regional_experiment_prefix(region).rstrip("/")
    if not ctx.is_fingerprint and ctx.prefix.rstrip("/") != expected:
        raise ValueError(
            f"execution prefix {ctx.prefix!r} must be region-local {expected!r}"
        )


def afdb_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts_v1/afdb",
        version=CACHE_VERSION,
        source=prefix_join(
            regional_experiment_prefix(region),
            f"tokenized/contacts_v1/afdb/{CACHE_VERSION}",
        ),
        tags=["protein", "contacts-v1", "decontaminated", "afdb"],
        expected_documents=AFDB_DOCUMENTS,
        expected_tokens=AFDB_TOKENS,
    )


def esm_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts_v1/esm",
        version=CACHE_VERSION,
        source=prefix_join(
            regional_experiment_prefix(region),
            f"tokenized/contacts_v1/esm/{CACHE_VERSION}",
        ),
        tags=["protein", "contacts-v1", "decontaminated", "esm"],
        expected_documents=ESM_DOCUMENTS,
        expected_tokens=ESM_TOKENS,
    )


def validation_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts_v1/validation",
        version=VALIDATION_CACHE_VERSION,
        source=prefix_join(
            regional_experiment_prefix(region),
            f"tokenized/contacts-v1-val/{VALIDATION_CACHE_VERSION}",
        ),
        tags=["protein", "contacts-v1", "validation", "exp199"],
    )


def source_checkpoint(region: str) -> ArtifactStep[LevanterCheckpoint]:
    source = prefix_join(
        regional_experiment_prefix(region),
        f"{SOURCE_SEED_NAMESPACE}/{SOURCE_RUN_ID}/{SOURCE_SEED_VERSION}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        f"{SOURCE_SEED_NAMESPACE}/{SOURCE_RUN_ID}",
        SOURCE_SEED_VERSION,
        source=source,
        kind=LevanterCheckpoint,
        config={
            "source_run": SOURCE_RUN_ID,
            "checkpoint_step": SOURCE_CHECKPOINT_STEP,
            "region": region,
            "full_state": True,
        },
    )


@dataclass(frozen=True)
class RunShape:
    end_step: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    wandb_group: str
    tags: list[str]


def _run_shape(
    *,
    variant: Variant,
    subversion: str,
    region: str,
    tpu: str,
    smoke: bool,
) -> RunShape:
    identity = f"{subversion}-m2-p06-srcpeak-augcont-{variant.key}"
    if smoke:
        smoke_steps = int(os.environ.get("SMOKE_STEPS", "20"))
        if smoke_steps < 2:
            raise ValueError("SMOKE_STEPS must be at least 2")
        end_step = RESUME_STEP + smoke_steps
        steps_per_eval = smoke_steps
        permanent_checkpoint_every = None
        run_id = f"{RUN_PREFIX}-smoke-{identity}-{region}-{tpu}"
    else:
        end_step = END_STEP
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{identity}-{region}"
    checkpoint_name = f"checkpoints/protein/{run_id}"
    tags = [
        "protein",
        "exp232",
        "contacts-v1",
        "decontaminated",
        "selected-training",
        "trc-lr-recovery",
        f"sweep={subversion}",
        f"variant={variant.key}",
        "source=m2-p06",
        f"source_checkpoint_step={SOURCE_CHECKPOINT_STEP}",
        "skip_bad_steps=true",
        f"data_seed={RECOVERY_DATA_SEED}",
        "augmentation=augcont",
        "augmentation_schedule=exp232-linear-global-clamp100",
        f"augmentation_resume_probability={augmentation_probability(RESUME_STEP, AUGMENTATION_RAMP_STEPS):.12f}",
        f"peak_lr={PEAK_LEARNING_RATE:g}",
        f"target_lr={variant.target_learning_rate:g}",
        f"target_ratio={variant.target_ratio:g}",
        f"initial_lr_transition_steps={INITIAL_LR_TRANSITION_STEPS}",
        f"final_cooldown_start={FINAL_COOLDOWN_START}",
        "final_lr=0",
        f"wd={WEIGHT_DECAY:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"start_step={RESUME_STEP}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        f"region={region}",
        f"tpu={tpu}",
    ]
    if smoke:
        tags.append("smoke")
    if oversized := [tag for tag in tags if not 1 <= len(tag) <= 64]:
        raise ValueError(f"W&B tags exceed 64 characters: {oversized}")
    return RunShape(
        end_step=end_step,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        wandb_group=f"{RUN_PREFIX}-{subversion}",
        tags=tags,
    )


def _apply_training_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    region: str,
    batch: TpuBatchConfig,
    shape: RunShape,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        _regional_prefix_guard(ctx, region)
        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError("TRC recovery requires the regional full-state checkpoint")
        exact_checkpoint = (
            prefix_join(source_checkpoint_dir, f"step-{SOURCE_CHECKPOINT_STEP}")
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
            per_device_parallelism=batch.per_device_parallelism,
            per_device_eval_parallelism=batch.per_device_parallelism,
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
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=RECOVERY_DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    *,
    variant: Variant,
    subversion: str,
    region: str,
    tpu: str,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _validate_placement(tpu, region, smoke=smoke)
    shape = _run_shape(
        variant=variant,
        subversion=subversion,
        region=region,
        tpu=tpu,
        smoke=smoke,
    )
    batch = batch_fit(tpu)
    env = _training_env(region)
    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=PEAK_LEARNING_RATE,
            min_lr_ratio=0.0,
            weight_decay=WEIGHT_DECAY,
            warmup=0.0,
            rewarmup=0.0,
            decay=1.0,
            lr_schedule=Exp232RecoveryLrSchedule(variant.target_ratio),
            skip_bad_steps=True,
        ),
        datasets={
            afdb_cache(region): AFDB_TOKENS / TARGET_TRAIN_TOKENS,
            esm_cache(region): ESM_TOKENS / TARGET_TRAIN_TOKENS,
        },
        validation=[validation_cache(region)],
        init_from=source_checkpoint(region),
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.end_step,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(
            tpu,
            regions=singleton_region_list(region),
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
            override_path=prefix_join(
                regional_experiment_prefix(region),
                f"tmp/checkpoints/{shape.run_id}",
            ),
        )
    return _apply_training_overrides(
        step,
        region=region,
        batch=batch,
        shape=shape,
        smoke=smoke,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    region = _parse_region()
    tpu = _parse_tpu()
    variant = _parse_variant()
    smoke = _truthy_env("SMOKE")
    _validate_launch_prefix(region)
    return build_run(
        variant=variant,
        subversion=_subversion(),
        region=region,
        tpu=tpu,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
