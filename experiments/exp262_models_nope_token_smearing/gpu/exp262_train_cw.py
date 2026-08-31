# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 of issue #262: the architecture ablation on CoreWeave GPUs.

Everything except the architecture is pinned to the exp232 contract — the same
decontaminated tokenized caches, tokenizer, mixture, sequence length, global
batch, optimizer shape, shuffle, amino-acid augmentation and seeds — so a
difference between arms is a difference between architectures. The two things
that move are the two under test:

* ``smear_width`` — 0, or 2 for the width-3 causal smear.
* ``rope`` — exp232's Llama3 rope, or ``NoRotaryEmbeddingsConfig`` (NoPE).

``ARM`` selects one cell of that 2x2. ``CLUSTER`` and ``NODES`` select
placement. ``LEARNING_RATE`` overrides the arm's default peak rate, which
matters because removing rope changes the scale of the attention logits and a
single shared rate would confound the architecture with the optimizer; the
per-arm defaults below come from the local pilot's stage-1 sweep.

**HF export is off for every arm, including the control.** A NoPE model has no
HF Qwen3 representation and the smear weights have no home in one, so
``to_hf_config`` refuses rather than exporting something that would load as a
different model. Levanter-native checkpoints are still written, and Phase 1 is
decided on validation loss; promoting a winner to a rollout evaluation needs a
real exporter first, which is Phase 2's problem. Disabling it for the control
too keeps the arms symmetric.

Budget is a fraction of exp232's, set by ``TOKEN_FRACTION``. This is an
architecture screen, not a production run.
"""

import math
import os
import sys
import uuid
from dataclasses import dataclass, replace

import click
from architecture import NoRotaryEmbeddingsConfig, SmearQwen3Config
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.setup_scripts import default_setup_script
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
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
    MODEL_SEED,
    NUM_TRAIN_STEPS,
    SEQ_LEN,
    SHUFFLE,
    TOKENS_PER_STEP,
    VALIDATION_CACHE_VERSION,
    WANDB_WATCH,
    WARMUP,
    augment_amino_acids,
    existing_cache,
)

RUN_PREFIX = "prot-exp262-cw-cv1-arch"
IRIS_PRIORITY_BAND_BATCH = 3

# exp262 writes under its own prefix but READS exp232's caches, which is the
# point: identical data, so only the architecture differs.
EXPERIMENT_PREFIX = "s3://marin-us-east-02a/MarinFold/exp262_models_nope_token_smearing"
EXP232_PREFIX = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam"
EXP232_TOKENIZED = f"{EXP232_PREFIX}/tokenized/contacts_v1"
AFDB_CACHE = f"{EXP232_TOKENIZED}/afdb/{CACHE_VERSION}"
ESM_CACHE = f"{EXP232_TOKENIZED}/esm/{CACHE_VERSION}"
VALIDATION_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    f"tokenized/contacts-v1-val/{VALIDATION_CACHE_VERSION}"
)

# exp232's published winner: the 5.95% AFDB / 94.05% ESM mixture at lr 1e-3,
# wd 0.2. Held fixed so the arms differ only in architecture.
AFDB_WEIGHT = AFDB_TOKENS / (AFDB_TOKENS + ESM_TOKENS)
ESM_WEIGHT = 1.0 - AFDB_WEIGHT
WEIGHT_DECAY = 0.2

# A screen, not a production run: a tenth of exp232's 152B-token budget.
TOKEN_FRACTION = 0.1

CUDNN_X86_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-py3-none-manylinux_2_27_x86_64.whl"
)
CUDNN_ARM_WHEEL = (
    "https://pypi.nvidia.com/nvidia-cudnn-cu13/"
    "nvidia_cudnn_cu13-9.26.0.17.dev59162438-py3-none-manylinux_2_27_aarch64.whl"
)


@dataclass(frozen=True)
class Arm:
    """One cell of the 2x2, and the peak learning rate the pilot chose for it."""

    key: str
    use_rope: bool
    smear_width: int
    learning_rate: float
    label: str


ARMS = {
    arm.key: arm
    for arm in (
        Arm("a-rope", True, 0, 1e-3, "RoPE, no smear (control = exp232 m2-p06-aug)"),
        Arm("b-rope-smear", True, 2, 1e-3, "RoPE + smear(2)"),
        Arm("c-nope-smear", False, 2, 1e-3, "NoPE + smear(2)"),
        Arm("d-nope", False, 0, 1e-3, "NoPE, no smear"),
    )
}


def model_config(arm: Arm) -> SmearQwen3Config:
    """exp232's model with only the arm's two fields changed."""
    fields = {
        field.name: getattr(MODEL_CONFIG, field.name)
        for field in MODEL_CONFIG.__dataclass_fields__.values()
    }
    fields["smear_width"] = arm.smear_width
    if not arm.use_rope:
        fields["rope"] = NoRotaryEmbeddingsConfig()
    return SmearQwen3Config(**fields)


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
MAX_SEQS_PER_DEVICE = {"GB200": 32, "H100": 8}


@dataclass(frozen=True)
class GpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def gpu_batch_fit(spec: ClusterSpec, *, nodes: int) -> GpuBatchConfig:
    """exp199's measured GPU microbatch and accumulation settings."""
    devices = spec.gpus_per_node * nodes
    data_parallelism = math.gcd(GLOBAL_BATCH_SIZE, devices)
    tensor_parallelism = devices // data_parallelism
    sequences_per_device = GLOBAL_BATCH_SIZE // data_parallelism
    per_device_parallelism = min(sequences_per_device, MAX_SEQS_PER_DEVICE[spec.gpu_variant])
    while sequences_per_device % per_device_parallelism:
        per_device_parallelism -= 1
    return GpuBatchConfig(
        data_parallelism=data_parallelism,
        tensor_parallelism=tensor_parallelism,
        per_device_parallelism=per_device_parallelism,
        gradient_accumulation=sequences_per_device // per_device_parallelism,
    )


def afdb_cache() -> ArtifactStep:
    return existing_cache(
        name="tokenized/contacts_v1/afdb",
        version=CACHE_VERSION,
        source=AFDB_CACHE,
        tags=["protein", "contacts-v1", "decontaminated", "afdb"],
        expected_documents=AFDB_DOCUMENTS,
        expected_tokens=AFDB_TOKENS,
    )


def esm_cache() -> ArtifactStep:
    return existing_cache(
        name="tokenized/contacts_v1/esm",
        version=CACHE_VERSION,
        source=ESM_CACHE,
        tags=["protein", "contacts-v1", "decontaminated", "esm"],
        expected_documents=ESM_DOCUMENTS,
        expected_tokens=ESM_TOKENS,
    )


def validation_cache() -> ArtifactStep:
    return existing_cache(
        name="tokenized/contacts_v1/validation",
        version=VALIDATION_CACHE_VERSION,
        source=VALIDATION_CACHE,
        tags=["protein", "contacts-v1", "validation", "exp199"],
    )


def _pinned_cuda_toolchain_setup_script() -> str:
    """exp232's pinned cuDNN wheel, kept because the NVIDIA placeholder
    downloader returned mismatched payload hashes on multi-node cold starts."""
    return "\n".join(
        [
            "set -euo pipefail",
            'if [ "$(uname -m)" = "aarch64" ]; then',
            f'  WHEEL="{CUDNN_ARM_WHEEL}"',
            "else",
            f'  WHEEL="{CUDNN_X86_WHEEL}"',
            "fi",
            'uv pip install --reinstall "$WHEEL" "nvidia-nccl-cu13==2.30.7"',
        ]
    )


def _run_train_job(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch training at batch priority with exp232's GPU setup."""
    env_vars = resolve_training_env(pod_config.env_vars, pod_config.resources)
    dependency_groups = dependency_groups_for_resources(pod_config.resources, None)
    child_env = env_vars_for_dependency_groups(pod_config.resources, dependency_groups, env_vars)
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
            entrypoint=Entrypoint.from_callable(lambda: run_levanter_train_lm(pod_config)),
            resources=pod_config.resources,
            environment=create_environment(
                extras=dependency_groups, env_vars=child_env, setup_scripts=setup_scripts
            ),
            priority=IRIS_PRIORITY_BAND_BATCH,
        )
    )
    handle.wait(raise_on_failure=True)


def _training_env() -> dict[str, str]:
    required = ("WANDB_ENTITY", "WANDB_PROJECT")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")
    env = {
        "MARIN_PREFIX": EXPERIMENT_PREFIX,
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _parse_arm() -> Arm:
    key = os.environ.get("ARM", "").strip().lower()
    try:
        arm = ARMS[key]
    except KeyError as exc:
        raise SystemExit(f"ARM must be one of: {', '.join(ARMS)}") from exc
    override = os.environ.get("LEARNING_RATE")
    return replace(arm, learning_rate=float(override)) if override else arm


def _parse_cluster() -> tuple[str, ClusterSpec]:
    cluster = os.environ.get("CLUSTER", "").strip().lower()
    try:
        return cluster, CLUSTERS[cluster]
    except KeyError as exc:
        raise SystemExit(f"CLUSTER must be one of: {', '.join(CLUSTERS)}") from exc


def _parse_nodes() -> int:
    raw = os.environ.get("NODES")
    if raw is None:
        raise SystemExit("missing required env var NODES")
    nodes = int(raw)
    if nodes not in {1, 2, 4, 8, 16}:
        raise SystemExit(f"NODES must be one of 1, 2, 4, 8, 16; got {nodes}")
    return nodes


def build_run(arm: Arm, *, cluster: str, spec: ClusterSpec, nodes: int, steps: int) -> ArtifactStep[LevanterCheckpoint]:
    batch = gpu_batch_fit(spec, nodes=nodes)
    env = _training_env()
    run_id = f"{RUN_PREFIX}-{arm.key}-lr{arm.learning_rate:g}"

    step = train_lm(
        name=run_id,
        run_id=run_id,
        model=model_config(arm),
        optimizer=AdamConfig(
            learning_rate=arm.learning_rate,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            decay=DECAY,
            min_lr_ratio=MIN_LR_RATIO,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={afdb_cache(): AFDB_WEIGHT, esm_cache(): ESM_WEIGHT},
        validation=[validation_cache()],
        init_from=None,
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=steps,
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
        steps_per_eval=max(1, steps // 20),
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=f"exp262-{arm.key}",
        tags=["exp262", f"arm={arm.key}", f"rope={arm.use_rope}", f"smear={arm.smear_width}"],
        env_vars=env,
    )

    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            seed=MODEL_SEED,
            max_eval_batches=None,
            watch=WANDB_WATCH,
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
        data = augment_amino_acids(data, steps)
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            # No HF export for ANY arm — see the module docstring.
            hf_save_steps=None,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config, run=_run_train_job)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    arm = _parse_arm()
    cluster, spec = _parse_cluster()
    nodes = _parse_nodes()
    steps = int(NUM_TRAIN_STEPS * TOKEN_FRACTION)
    print(
        f"[exp262] {arm.key} ({arm.label}) lr={arm.learning_rate:g} "
        f"{steps} steps x {TOKENS_PER_STEP} tokens = "
        f"{steps * TOKENS_PER_STEP / 1e9:.1f}B on {nodes} x {spec.gpu_variant} ({cluster})"
    )
    return build_run(arm, cluster=cluster, spec=spec, nodes=nodes, steps=steps)


if __name__ == "__main__":
    main()
