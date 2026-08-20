# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp199 TRC sweep over the existing AFDB and ESM contacts-v1 caches.

This is the #166 Qwen3/optimizer recipe with the AFDB+ESM mixture from #137,
the stable contacts-v1 tokenizer, and strict model-only initialization from the
corresponding region-local #117 checkpoint. It never copies or tokenizes data.

``TRIAL`` selects one of the 24 logical trials (for example
``m1-p01-base``); ``REGION`` and ``TPU`` select only its execution placement.
The CalVer suffix passed to ``--version`` becomes the sweep subversion: a
version ending in ``.1`` uses ``s01`` in W&B and checkpoint identities.

Preview without executing::

    MARIN_PREFIX=gs://marin-us-east5/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm \
      TRIAL=m1-p01-base REGION=us-east5 TPU=v6e-64 \
      uv run --extra tpu --frozen python exp199_sweep_trc.py \
      --version 2026.08.07.1

Run only after the regional caches and all six seed checkpoints have passed the
one-time preflight review::

    source ~/marin.env
    MARIN_PREFIX=gs://marin-us-east5/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm \
      TRIAL=m1-p01-base REGION=us-east5 TPU=v6e-64 \
      uv run --extra tpu --frozen python exp199_sweep_trc.py \
      --version 2026.08.07.1 --run
"""

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Self

import click
import jax
import numpy as np
from fray.types import (
    ResourceConfig,
    get_tpu_topology,
    tpu_family,
    tpu_hbm_capacity_bytes,
)
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
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
from marin.rl.placement import marin_prefix_for_region, singleton_region_list
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, prefix_join

# --- Identity and regional artifacts ----------------------------------------

RUN_PREFIX = "prot-exp199-cv1"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
VOCAB_SIZE = 2845
TEXT_KEY = "document"

AFDB_CACHE_RELATIVE = "tokenized/contacts-v1/2026.07.13.1"
ESM_CACHE_RELATIVE = (
    "protein-structure/MarinFold/"
    "exp137_contacts_and_crops_v1_1_5b/tokenized/contacts-v1-esm-atlas-train-568225"
)
VALIDATION_CACHE_RELATIVE = "tokenized/contacts-v1-val/2026.07.13.1"

AFDB_CACHE_VERSION = "2026.07.13.1"
ESM_CACHE_VERSION = "2026.07.21"
VALIDATION_CACHE_VERSION = "2026.07.13.1"
EXP117_SEED_VERSION = "2026.07.13.02"
EXP117_SEED_NAMESPACE = "checkpoints/protein/exp166-init"

# --- Fixed recipe and token accounting -------------------------------------

SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
NUM_TRAIN_STEPS = 72_600
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LEN
EFFECTIVE_TRAIN_TOKENS = NUM_TRAIN_STEPS * TOKENS_PER_STEP

AFDB_EXAMPLES = 4_129_682
AFDB_TOKENS = 4_676_753_425
ESM_EXAMPLES = 66_759_922
ESM_TOKENS = 71_450_105_324
COMBINED_EXAMPLES = AFDB_EXAMPLES + ESM_EXAMPLES
TARGET_TRAIN_TOKENS = AFDB_TOKENS + ESM_TOKENS
STEP_ROUNDING_DIFFERENCE = EFFECTIVE_TRAIN_TOKENS - TARGET_TRAIN_TOKENS

# One eval about every half AFDB epoch. Rounding to whole training steps gives
# 2,338,324,480 tokens, 52,232 below the requested 2,338,376,712-token cadence.
EVAL_TARGET_TOKENS = 2_338_376_712
STEPS_PER_EVAL = round(EVAL_TARGET_TOKENS / TOKENS_PER_STEP)
EVAL_TOKENS = STEPS_PER_EVAL * TOKENS_PER_STEP
PERMANENT_CHECKPOINT_EVERY = 4 * STEPS_PER_EVAL

DATA_SEED = 0
AA_AUGMENTATION_SEED = 166
WARMUP = 0.1
LR_SCHEDULE = "cosine"
CORRECTION_FACTORS = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45}

SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")

# Qwen3 already installs RMS Q/K normalization through attention_config(); set
# the inherited flag explicitly as well so the serialized config says so.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_qk_norm=True,
)
# Levanter's inherited Llama parameter-count helper omits Qwen3's Q/K norm
# weights. Include their two [layers, head_dim] arrays in the W&B/HBM count.
QK_NORM_PARAMS = 2 * MODEL_CONFIG.num_layers * MODEL_CONFIG.actual_head_size
MODEL_PARAMS = int(MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)) + QK_NORM_PARAMS

CONTACTS_V1_TOKEN_IDS = {
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
}


@dataclass(frozen=True)
class Point:
    """One #166 optimization point and its corresponding #117 source run."""

    key: str
    learning_rate: float
    weight_decay: float
    source_batch_size: int
    exp117_run: str


POINTS = (
    Point(
        key="p01",
        learning_rate=3.1623e-3,
        weight_decay=0.2,
        source_batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-europe-west4",
    ),
    Point(
        key="p02",
        learning_rate=3.1623e-4,
        weight_decay=1.6,
        source_batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-4-wd1p6-bs64-us-east5",
    ),
    Point(
        key="p03",
        learning_rate=3.1623e-3,
        weight_decay=0.1,
        source_batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p1-bs128-europe-west4",
    ),
    Point(
        key="p04",
        learning_rate=1e-3,
        weight_decay=0.8,
        source_batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr1e-3-wd0p8-bs128-us-east5",
    ),
    Point(
        key="p05",
        learning_rate=3.1623e-4,
        weight_decay=1.6,
        source_batch_size=128,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr3p162e-4-wd1p6-bs128-us-east1",
    ),
    Point(
        key="p06",
        learning_rate=1e-3,
        weight_decay=0.2,
        source_batch_size=64,
        exp117_run="prot-exp117-cv1-s02-1_5b-e8-lr1e-3-wd0p2-bs64-us-east5",
    ),
)


@dataclass(frozen=True)
class Mixture:
    key: str
    afdb_weight: float
    esm_weight: float


MIXTURES = (
    Mixture("m1", 0.5, 0.5),
    Mixture(
        "m2",
        AFDB_TOKENS / TARGET_TRAIN_TOKENS,
        ESM_TOKENS / TARGET_TRAIN_TOKENS,
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
        Trial(mixture, point, Augmentation.BASE),
        Trial(mixture, point, Augmentation.AUG),
    )
}


# --- Existing regional caches and checkpoints ------------------------------


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


def _regional_prefix_guard(ctx, region: str) -> None:
    region_root = marin_prefix_for_region(region)
    if not ctx.is_fingerprint and not ctx.prefix.startswith(f"{region_root}/"):
        raise ValueError(
            f"execution prefix {ctx.prefix!r} must be below the {region!r} root "
            f"{region_root!r}; refusing a bucket-root or cross-region execution"
        )


def _validate_launch_prefix(region: str) -> None:
    """Reject a bucket-root/cross-region prefix before adopted deps can run."""
    configured = marin_prefix()
    region_root = marin_prefix_for_region(region)
    if not configured.startswith(f"{region_root}/"):
        raise ValueError(
            f"MARIN_PREFIX {configured!r} must be below the {region!r} root "
            f"{region_root!r}"
        )


def _existing_cache(
    *,
    name: str,
    version: str,
    relative_path: str,
    region: str,
) -> ArtifactStep[TokenizedCache]:
    return ArtifactStep[TokenizedCache].adopt(
        name,
        version,
        source=prefix_join(marin_prefix_for_region(region), relative_path),
        kind=ExistingContactsV1TokenizerCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": TEXT_KEY},
            "tags": ["protein", "contacts-v1", name],
        },
    )


def afdb_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1",
        version=AFDB_CACHE_VERSION,
        relative_path=AFDB_CACHE_RELATIVE,
        region=region,
    )


def esm_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-esm-atlas",
        version=ESM_CACHE_VERSION,
        relative_path=ESM_CACHE_RELATIVE,
        region=region,
    )


def validation_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-val",
        version=VALIDATION_CACHE_VERSION,
        relative_path=VALIDATION_CACHE_RELATIVE,
        region=region,
    )


def exp117_checkpoint(point: Point, region: str) -> ArtifactStep[LevanterCheckpoint]:
    source = prefix_join(
        marin_prefix_for_region(region),
        f"{EXP117_SEED_NAMESPACE}/{point.exp117_run}/{EXP117_SEED_VERSION}",
    )
    return ArtifactStep[LevanterCheckpoint].adopt(
        f"{EXP117_SEED_NAMESPACE}/{point.exp117_run}",
        EXP117_SEED_VERSION,
        source=source,
        kind=LevanterCheckpoint,
        config={"source_run": point.exp117_run, "region": region},
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


# --- #166 TPU batch calibration --------------------------------------------


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


def batch_fit(tpu: str, batch_size: int) -> TpuBatchConfig:
    """Select #166's DP/TP/microbatch/accumulation settings for one TPU slice."""
    family = tpu_family(tpu)
    try:
        correction_factor = CORRECTION_FACTORS[family]
    except KeyError as exc:
        raise ValueError(f"unsupported TPU family {family!r}") from exc

    topology = get_tpu_topology(tpu)
    chips = topology.chip_count
    data_parallelism = math.gcd(batch_size, chips)
    tensor_parallelism = chips // data_parallelism
    batch_bytes = _batch_memory_bytes(batch_size, correction_factor)
    capacity_bytes = tpu_hbm_capacity_bytes(tpu)

    if batch_bytes <= capacity_bytes:
        return TpuBatchConfig(
            data_parallelism=data_parallelism,
            tensor_parallelism=tensor_parallelism,
            per_device_parallelism=batch_size // data_parallelism,
            gradient_accumulation=1,
        )

    full_per_device_batch = batch_size // data_parallelism
    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism:
            continue
        microbatch_size = per_device_parallelism * data_parallelism
        microbatch_bytes = math.ceil(batch_bytes * microbatch_size / batch_size)
        if microbatch_bytes <= capacity_bytes:
            return TpuBatchConfig(
                data_parallelism=data_parallelism,
                tensor_parallelism=tensor_parallelism,
                per_device_parallelism=per_device_parallelism,
                gradient_accumulation=batch_size // microbatch_size,
            )
    raise ValueError(f"global batch {batch_size} does not fit on {tpu}")


def _validate_placement(tpu: str, region: str, *, smoke: bool) -> None:
    family = tpu_family(tpu)
    allowed_regions = {
        "v6e": {"europe-west4", "us-east1", "us-east5"},
        "v5e": {"europe-west4", "us-west4"},
        "v5p": {"us-east5"},
    }
    if region not in allowed_regions.get(family, set()):
        raise ValueError(f"#166 placement policy does not allow {tpu} in {region}")

    if smoke:
        return

    chips = get_tpu_topology(tpu).chip_count
    low, high = (16, 256) if family == "v5p" else (32, 512)
    if not low <= chips <= high:
        raise ValueError(
            f"production {tpu} has {chips} chips; "
            f"exp199 requires {low}--{high} for {family}"
        )


# --- Run construction -------------------------------------------------------


@dataclass(frozen=True)
class RunShape:
    num_train_steps: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
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


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _parse_trial() -> Trial:
    key = os.environ.get("TRIAL", "").strip().lower()
    try:
        return TRIALS[key]
    except KeyError as exc:
        raise SystemExit(f"TRIAL must be one of: {', '.join(TRIALS)}") from exc


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
    get_tpu_topology(tpu)
    return tpu


def _training_env() -> dict[str, str]:
    required = ("WANDB_ENTITY", "WANDB_PROJECT")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(
            f"missing required environment variables: {', '.join(missing)}"
        )
    env = {key: os.environ[key] for key in required}
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _run_shape(
    trial: Trial,
    *,
    subversion: str,
    region: str,
    tpu: str,
    smoke: bool,
) -> RunShape:
    trial_identity = f"{subversion}-{trial.key}"
    if smoke:
        num_train_steps = int(os.environ.get("SMOKE_STEPS", "10"))
        if num_train_steps < 2:
            raise ValueError("SMOKE_STEPS must be at least 2 for the augmentation ramp")
        steps_per_eval = num_train_steps
        permanent_checkpoint_every = None
        run_id = f"{RUN_PREFIX}-smoke-{trial_identity}-{region}-{tpu}"
        checkpoint_name = (
            f"checkpoints/protein/{RUN_PREFIX}-smoke-{trial_identity}-{region}-{tpu}"
        )
    else:
        num_train_steps = NUM_TRAIN_STEPS
        steps_per_eval = STEPS_PER_EVAL
        permanent_checkpoint_every = PERMANENT_CHECKPOINT_EVERY
        run_id = f"{RUN_PREFIX}-{trial_identity}-{region}"
        checkpoint_name = f"checkpoints/protein/{RUN_PREFIX}-{trial_identity}"

    tokens = num_train_steps * TOKENS_PER_STEP
    tags = [
        "protein",
        "exp199",
        "contacts-v1",
        f"sweep={subversion}",
        f"mixture={trial.mixture.key}",
        f"point={trial.point.key}",
        f"augmentation={trial.augmentation.value}",
        f"region={region}",
        f"tpu={tpu}",
        f"lr={trial.point.learning_rate:g}",
        f"wd={trial.point.weight_decay:g}",
        f"batch={GLOBAL_BATCH_SIZE}",
        f"params={MODEL_PARAMS}",
        f"steps={num_train_steps}",
        f"tokens={tokens}",
    ]
    if smoke:
        tags.append("smoke")
    return RunShape(
        num_train_steps,
        steps_per_eval,
        permanent_checkpoint_every,
        run_id,
        checkpoint_name,
        tags,
    )


def _apply_recipe_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    trial: Trial,
    tpu: str,
    region: str,
    shape: RunShape,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        _regional_prefix_guard(ctx, region)
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=None,
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
        components = {
            key: replace(component, pack=True)
            for key, component in data.components.items()
        }
        data = replace(
            data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components=components,
            block_cross_document_attention=True,
        )
        if trial.augmentation is Augmentation.AUG:
            data = augment_amino_acids(data, shape.num_train_steps)

        if not ctx.is_fingerprint:
            batch_config = batch_fit(tpu, GLOBAL_BATCH_SIZE)
            trainer = replace(
                trainer,
                per_device_parallelism=batch_config.per_device_parallelism,
                per_device_eval_parallelism=batch_config.per_device_parallelism,
            )

        initialize_model = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and initialize_model is None:
            raise ValueError("every exp199 trial requires a #117 model checkpoint")
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=initialize_model,
            # Trainer forces hooks at completion, so this suppresses intermediate
            # HF exports while retaining the final export.
            hf_save_steps=shape.num_train_steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    trial: Trial,
    *,
    subversion: str,
    region: str,
    tpu: str,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    _validate_placement(tpu, region, smoke=smoke)
    shape = _run_shape(
        trial,
        subversion=subversion,
        region=region,
        tpu=tpu,
        smoke=smoke,
    )
    batch_config = batch_fit(tpu, GLOBAL_BATCH_SIZE)
    afdb = afdb_cache(region)
    esm = esm_cache(region)
    validation = validation_cache(region)
    env = _training_env()

    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=trial.point.learning_rate,
            weight_decay=trial.point.weight_decay,
            warmup=WARMUP,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={
            afdb: trial.mixture.afdb_weight,
            esm: trial.mixture.esm_weight,
        },
        validation=[validation],
        init_from=exp117_checkpoint(trial.point, region),
        batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=shape.num_train_steps,
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
    return _apply_recipe_overrides(
        step,
        trial=trial,
        tpu=tpu,
        region=region,
        shape=shape,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    trial = _parse_trial()
    region = _parse_region()
    tpu = _parse_tpu()
    _validate_launch_prefix(region)
    return build_run(
        trial,
        subversion=_sweep_subversion(),
        region=region,
        tpu=tpu,
        smoke=_truthy_env("SMOKE"),
    )


if __name__ == "__main__":
    main()
