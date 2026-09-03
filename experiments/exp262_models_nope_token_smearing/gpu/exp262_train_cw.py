# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #262 at production scale: does NoPE + token smearing train better?

One question, asked cleanly: take the exp232 1.5B recipe exactly as it is, change
only the positional handling, and see whether validation loss improves. Every
other architectural detail is imported from the exp232 contract — the same
decontaminated tokenized caches, tokenizer, mixture, sequence length, global
batch, optimizer shape, shuffle, amino-acid augmentation, and seeds — and a test
fails if any of them moves. What changes is only:

* ``smear_width`` — 0, or 2 for the width-3 causal smear.
* ``rope`` — exp232's Llama3 rope, or ``NoRotaryEmbeddingsConfig`` (NoPE).

Two arms. ``control`` is the usual setup, exp232's ``m2-p06-aug``. ``nope-smear``
is the proposal. The only thing tuned is the optimizer point: removing rope
changes the scale of the attention logits, so holding one learning rate across
both arms would confound the architecture with the optimizer. The control needs
no tuning — exp232 already swept five points at this exact scale, data and budget
and ``p06`` won — so the whole tuning budget goes to the new arm.

Two phases, selected by ``PHASE``:

``screen``
    ``SCREEN_FRACTION`` of exp232's schedule. Picks the new arm's optimizer point
    and gives an early kill signal against a matched-budget control. A reduced
    run is NOT comparable to a prefix of a full one — the WSD schedule decays
    relative to the total — which is why the control is re-run here rather than
    read off exp232's curve at the same step.
``full``
    exp232's schedule exactly: 145,200 steps, 152B tokens. This is the headline
    comparison. For reference, exp232's ``s02-m2-p06-aug`` reached eval loss
    2.9918 in 75.8 hours.

Set ``SMOKE=1`` (optionally ``SMOKE_STEPS``) for a short, separately named run on
a temporary path. Run it before anything else on a new arm: the custom
``SmearQwen3Config`` has to survive being pickled into the worker and rebuilt
there, which nothing local can prove.

**HF export is off for both arms.** A NoPE model has no HF Qwen3 representation
and the smear weights have no home in one, so ``to_hf_config`` refuses rather
than exporting something that would load as a different model. Levanter-native
checkpoints are still written and the comparison is decided on validation loss;
promoting a winner to a rollout evaluation needs a real exporter first. Disabled
for the control too, so the arms stay symmetric.

The marin pin is newer than exp232's. That is why the control is re-run rather
than compared against exp232's published number: a stack difference could move
the loss scale on its own (see the #7209 note in experiments/exp180).
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
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.training.run_environment import dependency_groups_for_resources
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)
from rigging.filesystem.cluster_config import marin_temp_bucket

from exp232_contract import (
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
    """One architecture under comparison."""

    key: str
    use_rope: bool
    smear_width: int
    label: str


ARMS = {
    arm.key: arm
    for arm in (
        Arm("control", True, 0, "the usual setup — exp232 m2-p06-aug"),
        Arm("nope-smear", False, 2, "NoPE + width-3 causal smear"),
    )
}


@dataclass(frozen=True)
class Point:
    """An optimizer point. exp232's naming, extended where this arm needs it."""

    key: str
    learning_rate: float
    weight_decay: float


# p01-p06 are exp232's own sweep points, kept at their exp232 values so a shared
# name means a shared setting. p07/p08 extended the grid upward on the local
# pilot's advice; the screen then showed that advice does not transfer — at 1.5B
# the NoPE arm gets monotonically worse with rate (p06 3.1598, p01 3.2395, and
# p07 diverged, eval rising from 3.597 to 3.802 before it was cancelled). So the
# grid extends DOWNWARD instead: p09 brackets the optimum from below, and p10
# varies weight decay at the best rate found.
POINTS = {
    point.key: point
    for point in (
        Point("p06", 1e-3, 0.2),      # exp232's winner; the control's setting
        Point("p01", 3.1623e-3, 0.2),
        Point("p03", 3.1623e-3, 0.1),
        Point("p07", 1e-2, 0.2),
        Point("p08", 1e-2, 0.1),
        Point("p09", 3.1623e-4, 0.2),
        Point("p10", 1e-3, 0.1),
    )
}

# The control is not tuned here: exp232 swept five points at this exact scale,
# data and budget, and p06 won. Re-tuning it would spend the budget re-deriving
# a known answer.
CONTROL_POINT = "p06"

# What `PHASE=screen` costs, as a fraction of exp232's 145,200-step schedule.
SCREEN_FRACTION = 0.1


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


def verify_decontaminated_cache_counts() -> None:
    """Assert the caches hold exactly exp225's post-decontamination corpora.

    Carried over from exp232's ``_verify_decontaminated_cache_counts``. Declaring
    the counts in the artifact config only records an intention; this reads the
    live cache and compares. exp262's whole result is a comparison against the
    usual setup, and "the usual setup" includes training on the decontaminated
    data — so a silent cache swap would invalidate the experiment rather than
    just annoy us.
    """
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
                f"{name} cache stats do not match the pinned decontaminated data: "
                f"{observed=}, {pinned=}, {cache_path=}"
            )
        print(f"[exp262] {name} cache verified: {observed[0]:,} documents, {observed[1]:,} tokens")


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


def verify_and_train(pod_config: TrainLmOnPodConfig, *, expect_rope: bool, expect_smear: int) -> None:
    """Check the architecture in the WORKER, then train.

    This is the last line of defence against training the wrong model for three
    days. ``NoRotaryEmbeddingsConfig`` inherits ``theta`` from its base and
    serialises as ``{"theta": 10000.0}`` — byte-identical to a default rope
    config. The object reaches the worker by cloudpickle, which should preserve
    its class, but "should" is how the transformers-5 ``rope_parameters`` bug
    cost us 0.76 nats/token silently. So the worker asserts what it actually
    holds before it spends a single step on it.
    """
    model = pod_config.train_config.model
    is_nope = isinstance(model.rope, NoRotaryEmbeddingsConfig)
    smear = getattr(model, "smear_width", 0)
    print(f"[exp262] worker sees rope={type(model.rope).__name__} smear_width={smear}")
    if is_nope is expect_rope:
        raise ValueError(
            f"architecture did not survive dispatch: expected "
            f"{'NoPE' if not expect_rope else 'rope'}, worker holds "
            f"{type(model.rope).__name__}"
        )
    if smear != expect_smear:
        raise ValueError(
            f"architecture did not survive dispatch: expected smear_width="
            f"{expect_smear}, worker holds {smear}"
        )
    run_levanter_train_lm(pod_config)


def _make_run_train_job(arm: Arm):
    """Bind the arm's expectations into the dispatcher."""

    def _run_train_job(pod_config: TrainLmOnPodConfig) -> None:
        return _dispatch(pod_config, arm)

    return _run_train_job


def _dispatch(pod_config: TrainLmOnPodConfig, arm: Arm) -> None:
    """Dispatch training at batch priority with exp232's GPU setup."""
    # marin's own ``_train_job`` dispatches through ``remote()``, which offers no
    # priority and would land at the interactive band. #108 requires batch, so
    # the JobRequest is assembled here instead; everything else mirrors it.
    env_vars = resolve_training_env(pod_config.env_vars, pod_config.resources)
    dependency_groups = dependency_groups_for_resources(pod_config.resources, None)
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
                lambda: verify_and_train(
                    pod_config, expect_rope=arm.use_rope, expect_smear=arm.smear_width
                )
            ),
            resources=pod_config.resources,
            environment=create_environment(
                extras=dependency_groups, env_vars=env_vars, setup_scripts=setup_scripts
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
        return ARMS[key]
    except KeyError as exc:
        raise SystemExit(f"ARM must be one of: {', '.join(ARMS)}") from exc


def _parse_point(arm: Arm) -> Point:
    """The optimizer point. The control is pinned; the new arm selects one."""
    key = os.environ.get("POINT", "").strip().lower()
    if arm.key == "control" and not _truthy("SMOKE"):
        if key and key != CONTROL_POINT:
            raise SystemExit(
                f"the control is fixed at {CONTROL_POINT} (exp232's swept winner); "
                f"re-tuning it would spend budget re-deriving a known answer"
            )
        return POINTS[CONTROL_POINT]
    if arm.key == "control":
        return POINTS[key or CONTROL_POINT]
    try:
        return POINTS[key]
    except KeyError as exc:
        raise SystemExit(f"POINT must be one of: {', '.join(POINTS)}") from exc


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _parse_phase() -> tuple[str, int]:
    """``screen`` for the optimizer-point search, ``full`` for the headline."""
    if _truthy("SMOKE"):
        return "smoke", int(os.environ.get("SMOKE_STEPS", "20"))
    phase = os.environ.get("PHASE", "").strip().lower()
    if phase == "screen":
        return phase, int(NUM_TRAIN_STEPS * SCREEN_FRACTION)
    if phase == "full":
        return phase, NUM_TRAIN_STEPS
    raise SystemExit("PHASE must be 'screen' or 'full' (or set SMOKE=1)")


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


def build_run(
    arm: Arm, point: Point, *, phase: str, cluster: str, spec: ClusterSpec, nodes: int, steps: int
) -> ArtifactStep[LevanterCheckpoint]:
    verify_decontaminated_cache_counts()
    batch = gpu_batch_fit(spec, nodes=nodes)
    env = _training_env()
    run_id = f"{RUN_PREFIX}-{phase}-{arm.key}-{point.key}"

    step = train_lm(
        name=run_id,
        run_id=run_id,
        model=model_config(arm),
        optimizer=AdamConfig(
            learning_rate=point.learning_rate,
            weight_decay=point.weight_decay,
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
        steps_per_eval=max(1, steps // 20 if phase != "smoke" else steps),
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=f"exp262-{phase}",
        tags=[
            "exp262", f"phase={phase}", f"arm={arm.key}", f"point={point.key}",
            f"rope={arm.use_rope}", f"smear={arm.smear_width}",
        ],
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

    step = replace(step, build_config=build_config, run=_make_run_train_job(arm))
    if phase == "smoke":
        # Never let a smoke run write where a real one would.
        step = replace(step, override_path=marin_temp_bucket(1, f"checkpoints/{run_id}"))
    return step


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    arm = _parse_arm()
    point = _parse_point(arm)
    phase, steps = _parse_phase()
    cluster, spec = _parse_cluster()
    nodes = _parse_nodes()
    print(
        f"[exp262] {phase}: {arm.key} ({arm.label}) at {point.key} "
        f"lr={point.learning_rate:g} wd={point.weight_decay:g} — "
        f"{steps} steps x {TOKENS_PER_STEP} tokens = {steps * TOKENS_PER_STEP / 1e9:.1f}B "
        f"on {nodes} x {spec.gpu_variant} ({cluster})"
    )
    return build_run(
        arm, point, phase=phase, cluster=cluster, spec=spec, nodes=nodes, steps=steps
    )


if __name__ == "__main__":
    main()
