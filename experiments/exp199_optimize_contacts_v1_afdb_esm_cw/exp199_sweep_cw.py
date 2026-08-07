# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp199 CoreWeave sweep over the existing AFDB and ESM contacts-v1 caches.

This is the CoreWeave counterpart of ``exp199_sweep_trc.py`` in an isolated
GPU workspace. It keeps the
model, data mixture, token budget, evaluation cadence, checkpoint cadence, and
scheduled amino-acid augmentation, but initializes every run from scratch and
uses a warmup-stable-decay learning-rate schedule.

``TRIAL`` selects one of the 20 logical trials (for example ``m1-p01-base``).
``CLUSTER`` and ``NODES`` are placement only: they do not enter production run
or checkpoint identity, so a retry can move between CoreWeave clusters while
resuming the same output. Four nodes is the normal production gang; two nodes
is the crowded-cluster fallback. One node is reserved for smoke/calibration.

The CalVer suffix passed to ``--version`` becomes the sweep subversion. A
version ending in ``.1`` uses ``s01`` in W&B and checkpoint identities.

This experiment only adopts pre-existing token caches. It never copies or
tokenizes data.
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Self

import click
import jax
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.callbacks.watch import WatchConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
from marin.execution.build_context import current_build_context
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, marin_temp_bucket

# --- Identity and storage ---------------------------------------------------

RUN_PREFIX = "prot-exp199-cw-cv1"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
VOCAB_SIZE = 2845
TEXT_KEY = "document"

EXPERIMENT_PREFIX = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp199_optimize_contacts_v1_afdb_esm"
)

# AFDB and validation are byte-identical to the corresponding exp199 GCS
# caches. Only their embedded absolute ledger paths differ, so read them in
# place instead of making another copy.
AFDB_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    "tokenized/contacts-v1/2026.07.25"
)
VALIDATION_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    "tokenized/contacts-v1-val/2026.07.25"
)
ESM_CACHE = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp137_contacts_and_crops_v1_1_5b/tokenized/"
    "contacts-v1-esm-atlas-train-568225"
)

AFDB_CACHE_VERSION = "2026.07.25"
ESM_CACHE_VERSION = "2026.07.21"
VALIDATION_CACHE_VERSION = "2026.07.25"

# --- Fixed recipe and token accounting -------------------------------------

SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
NUM_TRAIN_STEPS = 145_200
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LEN
EFFECTIVE_TRAIN_TOKENS = NUM_TRAIN_STEPS * TOKENS_PER_STEP

AFDB_TOKENS = 4_676_753_425
ESM_TOKENS = 71_450_105_324
TARGET_TRAIN_TOKENS = AFDB_TOKENS + ESM_TOKENS

# One eval about every half AFDB epoch. Permanent checkpoints land at each 10%
# boundary through 90%, plus the forced final save: 10 per trial. In particular,
# step 116,160 captures the state at the 80% stable/decay boundary.
EVAL_TARGET_TOKENS = 2_338_376_712
STEPS_PER_EVAL = round(EVAL_TARGET_TOKENS / TOKENS_PER_STEP)
PERMANENT_CHECKPOINT_EVERY = NUM_TRAIN_STEPS // 10

MODEL_SEED = 0
DATA_SEED = 0
AA_AUGMENTATION_SEED = 166
WARMUP = 0.1
DECAY = 0.2
MIN_LR_RATIO = 0.1
LR_SCHEDULE = "linear"

SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WANDB_WATCH = WatchConfig(watch_targets=[], interval=0)

MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_qk_norm=True,
    # GPU defaults may fall back to a quadratic reference kernel when
    # Transformer Engine is absent. The blocked JAX kernel is required at 8192.
    attn_backend=AttentionBackend.JAX_FLASH,
)
QK_NORM_PARAMS = 2 * MODEL_CONFIG.num_layers * MODEL_CONFIG.actual_head_size
MODEL_PARAMS = int(MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)) + QK_NORM_PARAMS

CONTACTS_V1_TOKEN_IDS = {
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
}


@dataclass(frozen=True)
class Point:
    key: str
    learning_rate: float
    weight_decay: float


# p05 was identical to p02 once exp117 initialization was removed: its only
# former distinction was the source checkpoint's batch size. Retaining both
# would duplicate four full scratch runs.
POINTS = (
    Point(
        key="p01",
        learning_rate=3.1623e-3,
        weight_decay=0.2,
    ),
    Point(
        key="p02",
        learning_rate=3.1623e-4,
        weight_decay=1.6,
    ),
    Point(
        key="p03",
        learning_rate=3.1623e-3,
        weight_decay=0.1,
    ),
    Point(
        key="p04",
        learning_rate=1e-3,
        weight_decay=0.8,
    ),
    Point(
        key="p06",
        learning_rate=1e-3,
        weight_decay=0.2,
    ),
)


@dataclass(frozen=True)
class Mixture:
    key: str
    afdb_weight: float
    esm_weight: float


MIXTURES = (
    Mixture(
        key="m1",
        afdb_weight=0.5,
        esm_weight=0.5,
    ),
    Mixture(
        key="m2",
        afdb_weight=AFDB_TOKENS / TARGET_TRAIN_TOKENS,
        esm_weight=ESM_TOKENS / TARGET_TRAIN_TOKENS,
    ),
)


class Augmentation(StrEnum):
    BASE = "base"
    AUG = "aug"


@dataclass(frozen=True)
class Trial:
    mixture: Mixture
    point: Point
    augmentation: Augmentation

    @property
    def key(self) -> str:
        return f"{self.mixture.key}-{self.point.key}-{self.augmentation.value}"


TRIALS = {
    trial.key: trial
    for mixture in MIXTURES
    for point in POINTS
    for trial in (
        Trial(
            mixture=mixture,
            point=point,
            augmentation=Augmentation.BASE,
        ),
        Trial(
            mixture=mixture,
            point=point,
            augmentation=Augmentation.AUG,
        ),
    )
}

# --- Existing CoreWeave caches ---------------------------------------------


class ExistingContactsV1TokenizerCache(TokenizedCache):
    """Path-only view of an existing cache under the stable tokenizer contract."""

    @classmethod
    def raw_load(cls, source: str) -> Self:
        return cls(path=source)

    @property
    def cache_dir(self) -> str:
        return self.path

    @property
    def tokenizer(self) -> str:
        return TOKENIZER

    @property
    def format(self) -> TextLmDatasetFormat:
        return TextLmDatasetFormat(text_key=TEXT_KEY)

    @property
    def tags(self) -> list[str]:
        return ["protein", "contacts-v1", "pretokenized"]


def _existing_cache(
    *,
    name: str,
    version: str,
    source: str,
) -> ArtifactStep[TokenizedCache]:
    return ArtifactStep[TokenizedCache].adopt(
        name,
        version,
        source=source,
        kind=ExistingContactsV1TokenizerCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": TEXT_KEY},
            "tags": ["protein", "contacts-v1", name],
        },
    )


def afdb_cache() -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1",
        version=AFDB_CACHE_VERSION,
        source=AFDB_CACHE,
    )


def esm_cache() -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-esm-atlas",
        version=ESM_CACHE_VERSION,
        source=ESM_CACHE,
    )


def validation_cache() -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-val",
        version=VALIDATION_CACHE_VERSION,
        source=VALIDATION_CACHE,
    )


# --- Training-only scheduled augmentation ----------------------------------


@dataclass(frozen=True)
class AugmentationStats:
    documents: int = 0
    moved_statements: int = 0
    changed_token_positions: int = 0


def shuffle_amino_acid_statements(
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, AugmentationStats]:
    """Re-permute two-token sequence statements, leaving structure unchanged."""
    if token_ids.ndim != 1:
        raise ValueError(f"expected one token sequence, got shape {token_ids.shape}")

    augmented = token_ids.copy()
    begin_sequence_id = CONTACTS_V1_TOKEN_IDS["<begin_sequence>"]
    begin_statements_id = CONTACTS_V1_TOKEN_IDS["<begin_statements>"]
    documents = 0
    moved_statements = 0
    cursor = 0

    while cursor < augmented.size:
        begin_offsets = np.flatnonzero(augmented[cursor:] == begin_sequence_id)
        if begin_offsets.size == 0:
            break
        begin = cursor + int(begin_offsets[0])
        structure_offsets = np.flatnonzero(
            augmented[begin + 1 :] == begin_statements_id
        )
        if structure_offsets.size == 0:
            raise ValueError(
                "contacts-v1 sequence marker has no following structure marker"
            )

        structure = begin + 1 + int(structure_offsets[0])
        sequence_token_count = structure - begin - 1
        if sequence_token_count % 2:
            raise ValueError(
                f"contacts-v1 sequence section has odd token count {sequence_token_count}"
            )
        statement_count = sequence_token_count // 2
        if statement_count < 2:
            raise ValueError(
                f"contacts-v1 sequence section has only {statement_count} statement(s)"
            )

        statements = augmented[begin + 1 : structure].reshape(statement_count, 2).copy()
        permutation = rng.permutation(statement_count)
        augmented[begin + 1 : structure] = statements[permutation].reshape(-1)
        moved_statements += int(
            np.count_nonzero(permutation != np.arange(statement_count))
        )
        documents += 1
        cursor = structure + 1

    return augmented, AugmentationStats(
        documents=documents,
        moved_statements=moved_statements,
        changed_token_positions=int(np.count_nonzero(augmented != token_ids)),
    )


def augmentation_probability(step: int, num_train_steps: int) -> float:
    """Linearly ramp from zero at step 0 to one at the final step."""
    if step < 0 or num_train_steps < 2:
        raise ValueError(f"invalid augmentation schedule: {step=}, {num_train_steps=}")
    return min(step, num_train_steps - 1) / (num_train_steps - 1)


def _augmentation_rng(seed: int, index: int) -> np.random.Generator:
    if index < 0:
        raise ValueError(f"dataset index must be nonnegative, got {index}")
    return np.random.default_rng(
        np.random.SeedSequence([seed, index & 0xFFFFFFFF, index >> 32])
    )


def _augment_lm_example(
    example: LmExample,
    *,
    seed: int,
    index: int,
    probability: float,
) -> LmExample:
    rng = _augmentation_rng(seed, index)
    selected = probability >= 1.0 or (probability > 0.0 and rng.random() < probability)
    if not selected:
        return example

    original = np.asarray(jax.device_get(example.tokens.array))
    augmented, stats = shuffle_amino_acid_statements(original, rng)
    if stats.documents == 0:
        raise ValueError(
            "packed contacts-v1 training example contains no complete document"
        )
    if stats.changed_token_positions == 0:
        raise ValueError("selected contacts-v1 augmentation was a silent no-op")

    token_array = jax.device_put(augmented, example.tokens.array.sharding)
    return replace(example, tokens=replace(example.tokens, array=token_array))


class AminoAcidAugmentedDataset(AsyncDataset[LmExample]):
    """Apply deterministic scheduled augmentation to the global training stream."""

    def __init__(
        self,
        dataset: AsyncDataset[LmExample],
        *,
        seed: int,
        batch_schedule: BatchSchedule,
        num_train_steps: int,
    ):
        self.dataset = dataset
        self.seed = seed
        self.batch_schedule = batch_schedule
        self.num_train_steps = num_train_steps

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
                probability=augmentation_probability(
                    self.batch_schedule.find_step_containing_offset(index),
                    self.num_train_steps,
                ),
            )
            for index, example in zip(indices, examples, strict=True)
        ]


def _validate_contacts_v1_tokenizer(data: LmDataConfig) -> None:
    tokenizer = data.the_tokenizer
    observed = tokenizer.convert_tokens_to_ids(list(CONTACTS_V1_TOKEN_IDS))
    expected = list(CONTACTS_V1_TOKEN_IDS.values())
    if observed != expected or len(tokenizer) != VOCAB_SIZE:
        raise ValueError(
            "contacts-v1 tokenizer contract changed: "
            f"{observed=}, {expected=}, vocab_size={len(tokenizer)}"
        )


@dataclass(frozen=True)
class AminoAcidAugmentedDataConfig(LmDataConfig):
    augmentation_seed: int = AA_AUGMENTATION_SEED
    augmentation_num_train_steps: int = NUM_TRAIN_STEPS

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        dataset = super().train_set(Pos, batch_schedule, key=key)
        return AminoAcidAugmentedDataset(
            dataset,
            seed=self.augmentation_seed,
            batch_schedule=batch_schedule,
            num_train_steps=self.augmentation_num_train_steps,
        )


def augment_amino_acids(data: LmDataConfig, num_train_steps: int) -> LmDataConfig:
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return AminoAcidAugmentedDataConfig(
        **values,
        augmentation_num_train_steps=num_train_steps,
    )


# --- CoreWeave placement and batch fitting ---------------------------------


@dataclass(frozen=True)
class ClusterSpec:
    gpu_variant: str
    gpus_per_node: int
    cpu: int
    ram: str
    disk: str


CLUSTERS = {
    "cw-us-east-08a": ClusterSpec(
        gpu_variant="GB200",
        gpus_per_node=4,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
    "cw-us-east-02a": ClusterSpec(
        gpu_variant="H100",
        gpus_per_node=8,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
    "cw-rno2a": ClusterSpec(
        gpu_variant="H100",
        gpus_per_node=8,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
}

# Both capacities completed exp199 training, full validation, and final export
# in the one-node smoke. Production gangs need no gradient accumulation.
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
    if raw is not None:
        if not smoke:
            raise ValueError("PER_DEVICE is only allowed for smoke calibration")
        capacity = int(raw)
        if capacity < 1:
            raise ValueError(f"PER_DEVICE must be positive, got {capacity}")
        return capacity
    return MAX_SEQS_PER_DEVICE[spec.gpu_variant]


def gpu_batch_fit(
    spec: ClusterSpec,
    *,
    nodes: int,
    smoke: bool,
) -> GpuBatchConfig:
    """Use measured GPU sequence capacity to select microbatch/accumulation."""
    devices = spec.gpus_per_node * nodes
    if GLOBAL_BATCH_SIZE % devices:
        raise ValueError(
            f"global batch {GLOBAL_BATCH_SIZE} is not divisible by {devices} GPUs"
        )

    sequences_per_device = GLOBAL_BATCH_SIZE // devices
    per_device_parallelism = min(
        sequences_per_device,
        _per_device_capacity(spec, smoke=smoke),
    )
    while sequences_per_device % per_device_parallelism:
        per_device_parallelism -= 1

    return GpuBatchConfig(
        data_parallelism=devices,
        tensor_parallelism=1,
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
    allowed = {1, 2, 4} if smoke else {2, 4}
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
        raise ValueError("exp199 must be built under Marin's --version context")
    if context.versions.overrides:
        raise ValueError("--override is not supported because sNN is run-wide identity")
    match = re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.(\d+)", context.versions.default)
    if match is None:
        raise ValueError(
            "--version must be a CalVer with numeric suffix, for example 2026.08.07.1"
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
            raise ValueError("SMOKE_STEPS must be at least 2 for the augmentation ramp")
        steps_per_eval = num_train_steps
        permanent_checkpoint_every = None
        probe = f"-pd{os.environ['PER_DEVICE']}" if "PER_DEVICE" in os.environ else ""
        run_id = (
            f"{RUN_PREFIX}-smoke-{trial_identity}-{cluster}-"
            f"{spec.gpu_variant.lower()}-n{nodes}{probe}"
        )
        checkpoint_name = f"checkpoints/protein/{run_id}"
        wandb_group = f"{RUN_PREFIX}-smoke-{subversion}"
    else:
        num_train_steps = NUM_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{trial_identity}"
        checkpoint_name = f"checkpoints/protein/{run_id}"
        wandb_group = f"{RUN_PREFIX}-{subversion}"

    tokens = num_train_steps * TOKENS_PER_STEP
    tags = [
        "protein",
        "exp199",
        "contacts-v1",
        f"sweep={subversion}",
        f"mixture={trial.mixture.key}",
        f"point={trial.point.key}",
        f"augmentation={trial.augmentation.value}",
        f"lr={trial.point.learning_rate:g}",
        f"wd={trial.point.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={num_train_steps}",
        f"tokens={tokens}",
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
        checkpoint_name=checkpoint_name,
        wandb_group=wandb_group,
        tags=tags,
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    trial: Trial,
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
                keep=(
                    [{"every": shape.permanent_checkpoint_every}]
                    if shape.permanent_checkpoint_every is not None
                    else []
                ),
            ),
        )
        # Placement is operational, not artifact identity. Apply both the GPU
        # microbatch and placement tags only after fingerprinting so a retry
        # can move between clusters or gang sizes while retaining one run.
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
        if trial.augmentation is Augmentation.AUG:
            data = augment_amino_acids(data, shape.num_train_steps)

        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            # Suppress intermediate HF exports; the completion hook still
            # writes the final export.
            hf_save_steps=shape.num_train_steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    trial: Trial,
    *,
    subversion: str,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
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
            override_path=marin_temp_bucket(
                1,
                f"checkpoints/{shape.run_id}",
            ),
        )
    return _apply_recipe_overrides(
        step,
        trial=trial,
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
