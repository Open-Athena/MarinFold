# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave-only 1.5B sweep on the exp232 decontaminated contacts-v1 data.

``TRIAL`` selects one of ten always-augmented trials. ``CLUSTER`` and ``NODES``
control placement without changing the production W&B or checkpoint identity,
so one writer can resume on another production CoreWeave cluster.
"""

import math
import os
import re
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import cache

import click
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from levanter.optim.config import AdamConfig
from marin.execution.build_context import current_build_context
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.run_environment import (
    dependency_groups_for_resources,
    env_vars_for_dependency_groups,
)
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AFDB_DOCUMENTS,
    AFDB_TOKENS,
    CACHE_VERSION,
    DATA_SEED,
    DECAY,
    ESM_DOCUMENTS,
    ESM_TOKENS,
    GLOBAL_BATCH_SIZE,
    LR_SCHEDULE,
    MIN_LR_RATIO,
    MODEL_CONFIG,
    MODEL_PARAMS,
    MODEL_SEED,
    NUM_TRAIN_STEPS,
    PERMANENT_CHECKPOINT_EVERY,
    SEQ_LEN,
    SHUFFLE,
    STEPS_PER_EVAL,
    TARGET_TRAIN_DOCUMENTS,
    TARGET_TRAIN_TOKENS,
    TOKENS_PER_STEP,
    VALIDATION_CACHE_VERSION,
    WANDB_WATCH,
    WARMUP,
    augment_amino_acids,
    existing_cache,
)

# --- Identity and storage ---------------------------------------------------

RUN_PREFIX = "prot-exp232-cw-cv1-decontam"
IRIS_PRIORITY_BAND_BATCH = 3

CUDNN_X86_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-"
    "py3-none-manylinux_2_27_x86_64.whl"
)
CUDNN_ARM_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-"
    "py3-none-manylinux_2_27_aarch64.whl"
)

EXPERIMENT_PREFIX = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam"
TOKENIZED_PREFIX = f"{EXPERIMENT_PREFIX}/tokenized/contacts_v1"
AFDB_CACHE = f"{TOKENIZED_PREFIX}/afdb/{CACHE_VERSION}"
ESM_CACHE = f"{TOKENIZED_PREFIX}/esm/{CACHE_VERSION}"

# Keep exp199's validation set for direct comparability.
VALIDATION_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    "tokenized/contacts-v1-val/2026.07.25"
)
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=15)


def _pinned_cuda_toolchain_setup_script() -> str:
    """Keep Iris's CUDA precedence repair deterministic for the locked cuDNN."""
    script = cuda_toolchain_setup_script()
    install_marker = '    uv pip install --python "$IRIS_VENV/bin/python" \\\n'
    package_marker = '      "$_cuda13_package==$_cuda13_version"'
    if script.count(install_marker) != 1 or script.count(package_marker) != 1:
        raise ValueError("Iris CUDA setup script changed; update the exp232 pin")
    choose_spec = f"""    if [ "$_cuda13_package" = "nvidia-cudnn-cu13" ]; then
      case "$(uname -m)" in
        x86_64) _cuda13_spec={CUDNN_X86_WHEEL!r} ;;
        aarch64) _cuda13_spec={CUDNN_ARM_WHEEL!r} ;;
        *) _cuda13_spec="$_cuda13_package==$_cuda13_version" ;;
      esac
    else
      _cuda13_spec="$_cuda13_package==$_cuda13_version"
    fi
"""
    return script.replace(install_marker, choose_spec + install_marker).replace(
        package_marker,
        '      "$_cuda13_spec"',
    )


def _run_exp232_train_job(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch training with the normal GPU setup plus the pinned cuDNN repair."""
    env_vars = resolve_training_env(pod_config.env_vars, pod_config.resources)
    dependency_groups = dependency_groups_for_resources(pod_config.resources, None)
    child_env = env_vars_for_dependency_groups(
        pod_config.resources,
        dependency_groups,
        env_vars,
    )
    setup_scripts = [
        default_setup_script(
            extras=dependency_groups,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        ),
        _pinned_cuda_toolchain_setup_script(),
    ]
    handle = current_client().submit(
        JobRequest(
            name=f"run_levanter_train_lm-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(
                lambda: run_levanter_train_lm(pod_config)
            ),
            resources=pod_config.resources,
            environment=create_environment(
                extras=dependency_groups,
                env_vars=child_env,
                setup_scripts=setup_scripts,
            ),
            priority=IRIS_PRIORITY_BAND_BATCH,
        )
    )
    handle.wait(raise_on_failure=True)


@dataclass(frozen=True)
class Point:
    key: str
    learning_rate: float
    weight_decay: float


POINTS = (
    Point("p01", learning_rate=3.1623e-3, weight_decay=0.2),
    Point("p02", learning_rate=3.1623e-4, weight_decay=1.6),
    Point("p03", learning_rate=3.1623e-3, weight_decay=0.1),
    Point("p04", learning_rate=1e-3, weight_decay=0.8),
    Point("p06", learning_rate=1e-3, weight_decay=0.2),
)


@dataclass(frozen=True)
class Mixture:
    key: str
    afdb_weight: float
    esm_weight: float


MIXTURES = (
    Mixture("m1", afdb_weight=0.5, esm_weight=0.5),
    Mixture(
        "m2",
        afdb_weight=AFDB_TOKENS / TARGET_TRAIN_TOKENS,
        esm_weight=ESM_TOKENS / TARGET_TRAIN_TOKENS,
    ),
)


@dataclass(frozen=True)
class Trial:
    mixture: Mixture
    point: Point

    @property
    def key(self) -> str:
        return f"{self.mixture.key}-{self.point.key}-aug"


TRIALS = {
    f"{mixture.key}-{point.key}-aug": Trial(mixture=mixture, point=point)
    for mixture in MIXTURES
    for point in POINTS
}


@cache
def _verify_decontaminated_cache_counts() -> None:
    expected = (
        ("afdb", AFDB_CACHE, AFDB_DOCUMENTS, AFDB_TOKENS),
        ("esm", ESM_CACHE, ESM_DOCUMENTS, ESM_TOKENS),
    )
    for name, cache_path, expected_documents, expected_tokens in expected:
        stats = read_tokenized_cache_stats(cache_path, "train")
        observed = (stats.total_elements, stats.total_tokens)
        pinned = (expected_documents, expected_tokens)
        if observed != pinned:
            raise ValueError(
                f"{name} cache stats do not match the pinned exp232 data: "
                f"{observed=}, {pinned=}, {cache_path=}"
            )


def afdb_cache() -> ArtifactStep[TokenizedCache]:
    return existing_cache(
        name="tokenized/contacts_v1/afdb",
        version=CACHE_VERSION,
        source=AFDB_CACHE,
        tags=["protein", "contacts-v1", "decontaminated", "afdb"],
        expected_documents=AFDB_DOCUMENTS,
        expected_tokens=AFDB_TOKENS,
    )


def esm_cache() -> ArtifactStep[TokenizedCache]:
    return existing_cache(
        name="tokenized/contacts_v1/esm",
        version=CACHE_VERSION,
        source=ESM_CACHE,
        tags=["protein", "contacts-v1", "decontaminated", "esm"],
        expected_documents=ESM_DOCUMENTS,
        expected_tokens=ESM_TOKENS,
    )


def validation_cache() -> ArtifactStep[TokenizedCache]:
    return existing_cache(
        name="tokenized/contacts_v1/validation",
        version=VALIDATION_CACHE_VERSION,
        source=VALIDATION_CACHE,
        tags=["protein", "contacts-v1", "validation", "exp199"],
    )


# --- CoreWeave placement ----------------------------------------------------


@dataclass(frozen=True)
class ClusterSpec:
    gpu_variant: str
    gpus_per_node: int
    cpu: int
    ram: str
    disk: str


CLUSTERS = {
    "cw-us-east-08a": ClusterSpec("GB200", 4, 32, "256g", "256g"),
    "cw-us-east-02a": ClusterSpec("H100", 8, 32, "256g", "256g"),
    "cw-rno2a": ClusterSpec("H100", 8, 32, "256g", "256g"),
}

MAX_SEQS_PER_DEVICE = {
    "GB200": 32,
    "H100": 8,
}


@dataclass(frozen=True)
class GpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _per_device_capacity(spec: ClusterSpec, *, smoke: bool) -> int:
    raw = os.environ.get("PER_DEVICE")
    if raw is None:
        return MAX_SEQS_PER_DEVICE[spec.gpu_variant]
    if not smoke:
        raise ValueError("PER_DEVICE is only allowed for smoke calibration")
    capacity = int(raw)
    if capacity < 1:
        raise ValueError(f"PER_DEVICE must be positive, got {capacity}")
    return capacity


def gpu_batch_fit(
    spec: ClusterSpec,
    *,
    nodes: int,
    smoke: bool,
) -> GpuBatchConfig:
    """Select exp199's measured GPU microbatch and accumulation settings."""
    devices = spec.gpus_per_node * nodes
    data_parallelism = math.gcd(GLOBAL_BATCH_SIZE, devices)
    tensor_parallelism = devices // data_parallelism
    sequences_per_device = GLOBAL_BATCH_SIZE // data_parallelism
    per_device_parallelism = min(
        sequences_per_device,
        _per_device_capacity(spec, smoke=smoke),
    )
    while sequences_per_device % per_device_parallelism:
        per_device_parallelism -= 1
    return GpuBatchConfig(
        data_parallelism=data_parallelism,
        tensor_parallelism=tensor_parallelism,
        per_device_parallelism=per_device_parallelism,
        gradient_accumulation=sequences_per_device // per_device_parallelism,
    )


def _parse_cluster() -> tuple[str, ClusterSpec]:
    cluster = os.environ.get("CLUSTER", "").strip().lower()
    try:
        return cluster, CLUSTERS[cluster]
    except KeyError as exc:
        raise SystemExit(f"CLUSTER must be one of: {', '.join(CLUSTERS)}") from exc


def _parse_nodes(*, smoke: bool) -> int:
    raw = os.environ.get("NODES")
    if raw is None:
        raise SystemExit("missing required env var NODES")
    nodes = int(raw)
    allowed = {1, 2, 4} if smoke else {2, 4, 8, 16}
    if nodes not in allowed:
        choices = ", ".join(str(value) for value in sorted(allowed))
        raise SystemExit(f"NODES must be one of {choices} for this run, got {nodes}")
    return nodes


# --- Run construction -------------------------------------------------------


@dataclass(frozen=True)
class RunShape:
    num_train_steps: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    wandb_group: str
    tags: list[str]


def _sweep_subversion() -> str:
    context = current_build_context()
    if context is None:
        raise ValueError("exp232 must be built under Marin's --version context")
    if context.versions.overrides:
        raise ValueError("--override is not supported because sNN is run-wide identity")
    match = re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.(\d+)", context.versions.default)
    if match is None:
        raise ValueError(
            "--version must be a CalVer with numeric suffix, for example 2026.08.14.1"
        )
    suffix = int(match.group(1))
    if suffix < 1 or suffix > 99:
        raise ValueError(f"CalVer suffix must be in 1--99, got {suffix}")
    return f"s{suffix:02d}"


def _parse_trial() -> Trial:
    key = os.environ.get("TRIAL", "").strip().lower()
    try:
        return TRIALS[key]
    except KeyError as exc:
        raise SystemExit(f"TRIAL must be one of: {', '.join(TRIALS)}") from exc


def _training_env() -> dict[str, str]:
    required = ("WANDB_ENTITY", "WANDB_PROJECT")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(
            f"missing required environment variables: {', '.join(missing)}"
        )
    env = {
        "MARIN_PREFIX": EXPERIMENT_PREFIX,
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _validate_launch_prefix() -> None:
    configured = marin_prefix().rstrip("/")
    if configured != EXPERIMENT_PREFIX:
        raise ValueError(
            f"MARIN_PREFIX must be exactly {EXPERIMENT_PREFIX!r}, got {configured!r}"
        )


def _run_shape(
    trial: Trial,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> RunShape:
    trial_identity = f"{subversion}-{trial.key}"
    if smoke:
        num_train_steps = int(os.environ.get("SMOKE_STEPS", "10"))
        if num_train_steps < 2:
            raise ValueError("SMOKE_STEPS must be at least 2")
        steps_per_eval = num_train_steps
        permanent_checkpoint_every = None
        probe = f"-pd{os.environ['PER_DEVICE']}" if "PER_DEVICE" in os.environ else ""
        run_id = (
            f"{RUN_PREFIX}-smoke-{trial_identity}-{cluster}-"
            f"{spec.gpu_variant.lower()}-n{nodes}{probe}"
        )
        wandb_group = f"{RUN_PREFIX}-smoke-{subversion}"
    else:
        num_train_steps = NUM_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{trial_identity}"
        wandb_group = f"{RUN_PREFIX}-{subversion}"

    tags = [
        "protein",
        "exp232",
        "contacts-v1",
        "decontaminated",
        f"sweep={subversion}",
        f"mixture={trial.mixture.key}",
        f"point={trial.point.key}",
        "augmentation=aug",
        f"lr={trial.point.learning_rate:g}",
        f"wd={trial.point.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={num_train_steps}",
        f"tokens={num_train_steps * TOKENS_PER_STEP}",
        f"source_documents={TARGET_TRAIN_DOCUMENTS}",
        f"source_tokens={TARGET_TRAIN_TOKENS}",
        "schedule=wsd",
        "initialization=scratch",
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
        num_train_steps=num_train_steps,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=f"checkpoints/protein/{run_id}",
        wandb_group=wandb_group,
        tags=tags,
    )


def _apply_training_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
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
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            max_eval_batches=None,
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
        data = augment_amino_acids(data, shape.num_train_steps)
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.num_train_steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(
        step,
        build_config=build_config,
        run=_run_exp232_train_job,
    )


def build_run(
    trial: Trial,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _verify_decontaminated_cache_counts()
    shape = _run_shape(
        trial,
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
            learning_rate=trial.point.learning_rate,
            weight_decay=trial.point.weight_decay,
            warmup=WARMUP,
            decay=DECAY,
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={
            afdb_cache(): trial.mixture.afdb_weight,
            esm_cache(): trial.mixture.esm_weight,
        },
        validation=[validation_cache()],
        init_from=None,
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
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
    trial = _parse_trial()
    cluster, spec = _parse_cluster()
    nodes = _parse_nodes(smoke=smoke)
    _validate_launch_prefix()
    return build_run(
        trial,
        subversion=_sweep_subversion(),
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
