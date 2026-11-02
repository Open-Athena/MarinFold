# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared model, data, and schedule contract for exp232 training backends."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Self

import jax
import numpy as np
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
from levanter.schedule import BatchSchedule
from marin.execution.lazy import ArtifactStep
from marin.processing.tokenize.tokenize import TokenizedCache

TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
VOCAB_SIZE = 2845
TEXT_KEY = "document"
CACHE_VERSION = "2026.08.14"
VALIDATION_CACHE_VERSION = "2026.07.25"

SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 128
NUM_TRAIN_STEPS = 145_200
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LEN
EFFECTIVE_TRAIN_TOKENS = NUM_TRAIN_STEPS * TOKENS_PER_STEP

AFDB_DOCUMENTS = 3_963_003
AFDB_TOKENS = 4_432_940_838
ESM_DOCUMENTS = 65_553_178
ESM_TOKENS = 70_042_923_165
TARGET_TRAIN_DOCUMENTS = AFDB_DOCUMENTS + ESM_DOCUMENTS
TARGET_TRAIN_TOKENS = AFDB_TOKENS + ESM_TOKENS

EVAL_TARGET_TOKENS = AFDB_TOKENS // 2
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
    attn_backend=AttentionBackend.JAX_FLASH,
)
QK_NORM_PARAMS = 2 * MODEL_CONFIG.num_layers * MODEL_CONFIG.actual_head_size
MODEL_PARAMS = int(MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)) + QK_NORM_PARAMS

CONTACTS_V1_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
    "<end>": 10,
    "<UNK>": 2844,
}


class ExistingContactsV1TokenizerCache(TokenizedCache):
    """Path-only view of a completed cache under the fixed tokenizer contract."""

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


def existing_cache(
    *,
    name: str,
    version: str,
    source: str,
    tags: list[str],
    expected_documents: int | None = None,
    expected_tokens: int | None = None,
) -> ArtifactStep[TokenizedCache]:
    """Adopt an already-built contacts-v1 cache with optional pinned counts."""
    if (expected_documents is None) != (expected_tokens is None):
        raise ValueError("expected document and token counts must be provided together")
    expected_counts: dict[str, int] = {}
    if expected_documents is not None and expected_tokens is not None:
        expected_counts = {
            "expected_documents": expected_documents,
            "expected_tokens": expected_tokens,
        }
    return ArtifactStep[TokenizedCache].adopt(
        name,
        version,
        source=source,
        kind=ExistingContactsV1TokenizerCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": TEXT_KEY},
            "tags": tags,
            **expected_counts,
        },
    )


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
    """Linearly ramp augmentation from zero to one over training."""
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
    """Apply deterministic scheduled augmentation to the training stream."""

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
