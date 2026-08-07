# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-step exp199 smoke test over existing AFDB and ESM token caches.

This temporary test isolates the modern Marin cache-consumption path. It keeps
exp166's model, optimizer, packing, block shuffle, and training-only amino-acid
augmentation, but trains on a 50/50 AFDB/ESM mixture and evaluates the complete
existing contacts-v1 validation cache.

All three data dependencies are path-only references to existing regional
caches. The graph contains no raw-document paths or tokenization function, and
Levanter cache auto-building is disabled. A missing cache therefore fails
instead of being copied or rebuilt.

Print the plan without executing it::

    REGION=us-east5 uv run --extra tpu --frozen \
        python train_exp166_cache_mixture_smoke.py --version dev

Do not pass ``--run`` until the script and lowered plan have been reviewed.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Self

import click
import jax
import numpy as np
from fray.types import ResourceConfig
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
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.rl.placement import marin_prefix_for_region, singleton_region_list
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

# Cache locations are relative to Marin's configured bucket for REGION. The
# historical ESM key is awkward, but keeping it relative prevents an accidental
# cross-region read.
AFDB_CACHE_RELATIVE = "tokenized/contacts-v1/2026.07.13.1"
VALIDATION_CACHE_RELATIVE = "tokenized/contacts-v1-val/2026.07.13.1"

# The completed exp137 ESM cache. It was produced with the larger crops
# tokenizer, whose contacts-v1 token ids occupy the same 0--2844 prefix used by
# this experiment's tokenizer.
ESM_CACHE_RELATIVE = (
    "protein-structure/MarinFold/"
    "exp137_contacts_and_crops_v1_1_5b/tokenized/contacts-v1-esm-atlas-train-568225"
)

# This repository republishes exp166's pinned contacts-v1 tokenizer as its
# latest revision, so Marin can load it without an unsupported ``repo@revision``
# suffix. Validate its vocabulary contract at training startup below.
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
VOCAB_SIZE = 2845
TEXT_KEY = "document"

TPU = "v6e-4"
RUN_PREFIX = "prot-exp199-smoke-cache-validation"

SEQ_LEN = 8192
BATCH_SIZE = 64
NUM_TRAIN_STEPS = 2
STEPS_PER_EVAL = 1
PER_DEVICE_PARALLELISM = 16

LEARNING_RATE = 3.1623e-3
WEIGHT_DECAY = 0.2
WARMUP = 0.1
LR_SCHEDULE = "cosine"
DATA_SEED = 0
AA_AUGMENTATION_SEED = 166
AA_AUGMENTATION_LOG_LIMIT = 4
_augmentation_log_count = 0

CONTACTS_V1_TOKEN_IDS: dict[str, int] = {
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
}

SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")

MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
MODEL_PARAMS = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
TRAIN_TOKENS = BATCH_SIZE * SEQ_LEN * NUM_TRAIN_STEPS


class ExistingContactsV1TokenizerCache(TokenizedCache):
    """Describe an existing compatible cache with the experiment tokenizer.

    AFDB and validation use the contacts-v1 tokenizer. ESM was produced with
    the larger crops tokenizer but contains ids from its identical contacts-v1
    prefix, and has a legacy ``.artifact.json`` containing JSON ``null``. This
    explicit path-only view gives every mixture component one tokenizer
    contract. It has no tokenization, transformation, or copy implementation.
    """

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
    expected_prefix = marin_prefix_for_region(region)
    if not ctx.is_fingerprint and not (
        ctx.prefix == expected_prefix or ctx.prefix.startswith(f"{expected_prefix}/")
    ):
        raise ValueError(
            f"execution prefix {ctx.prefix!r} is outside {region!r} ({expected_prefix}); "
            "refusing cross-region data access"
        )


def _existing_cache(
    *,
    name: str,
    version: str,
    cache_relative: str,
    region: str,
) -> ArtifactStep[TokenizedCache]:
    """Return a non-computable reference to an existing regional cache."""
    source = prefix_join(marin_prefix_for_region(region), cache_relative)
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


def parse_region() -> str:
    region = os.environ.get("REGION", "").strip().lower()
    if not region:
        raise SystemExit("missing required env var REGION")
    marin_prefix_for_region(region)  # Validate that Marin configures a bucket there.
    return region


def regional_run_id(region: str) -> str:
    return f"{RUN_PREFIX}-{region}-{TPU}"


def train_resources(region: str) -> ResourceConfig:
    return ResourceConfig.with_tpu(
        TPU,
        slice_count=1,
        cpu=32,
        ram="128g",
        disk="50g",
        regions=singleton_region_list(region),
    )


def afdb_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1",
        version="2026.07.13.1",
        cache_relative=AFDB_CACHE_RELATIVE,
        region=region,
    )


def esm_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-esm-atlas",
        version="2026.07.21",
        cache_relative=ESM_CACHE_RELATIVE,
        region=region,
    )


def validation_cache(region: str) -> ArtifactStep[TokenizedCache]:
    return _existing_cache(
        name="tokenized/contacts-v1-val",
        version="2026.07.13.1",
        cache_relative=VALIDATION_CACHE_RELATIVE,
        region=region,
    )


@dataclass(frozen=True)
class AugmentationStats:
    """Observable effect of re-randomizing sequence statements."""

    documents: int = 0
    residue_statements: int = 0
    moved_statements: int = 0
    changed_token_positions: int = 0


def shuffle_amino_acid_statements(
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, AugmentationStats]:
    """Re-permute each contacts-v1 sequence section without changing meaning."""
    if token_ids.ndim != 1:
        raise ValueError(f"expected one token sequence, got shape {token_ids.shape}")

    augmented = token_ids.copy()
    begin_sequence_id = CONTACTS_V1_TOKEN_IDS["<begin_sequence>"]
    begin_statements_id = CONTACTS_V1_TOKEN_IDS["<begin_statements>"]
    documents = 0
    residue_statements = 0
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
        sequence_length = structure - begin - 1
        if sequence_length % 2:
            raise ValueError(
                f"contacts-v1 sequence section has odd token count {sequence_length}"
            )

        statement_count = sequence_length // 2
        if statement_count < 2:
            raise ValueError(
                f"contacts-v1 sequence section has only {statement_count} statement(s)"
            )
        statements = augmented[begin + 1 : structure].reshape(statement_count, 2).copy()
        permutation = rng.permutation(statement_count)
        augmented[begin + 1 : structure] = statements[permutation].reshape(-1)
        documents += 1
        residue_statements += statement_count - 2
        moved_statements += int(
            np.count_nonzero(permutation != np.arange(statement_count))
        )
        cursor = structure + 1

    return augmented, AugmentationStats(
        documents=documents,
        residue_statements=residue_statements,
        moved_statements=moved_statements,
        changed_token_positions=int(np.count_nonzero(augmented != token_ids)),
    )


def _augmentation_rng(seed: int, index: int) -> np.random.Generator:
    if index < 0:
        raise ValueError(f"dataset index must be nonnegative, got {index}")
    entropy = [seed, index & 0xFFFFFFFF, index >> 32]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _augment_lm_example(example: LmExample, *, seed: int, index: int) -> LmExample:
    global _augmentation_log_count

    original = np.asarray(jax.device_get(example.tokens.array))
    augmented, stats = shuffle_amino_acid_statements(
        original, _augmentation_rng(seed, index)
    )
    if stats.documents == 0:
        raise ValueError(
            "packed contacts-v1 training example contains no complete document"
        )

    token_array = jax.device_put(augmented, example.tokens.array.sharding)
    result = replace(example, tokens=replace(example.tokens, array=token_array))

    if jax.process_index() == 0 and _augmentation_log_count < AA_AUGMENTATION_LOG_LIMIT:
        logging.getLogger(__name__).info(
            "exp166 AA augmentation runtime effect: documents=%d residue_statements=%d "
            "moved_statements=%d changed_token_positions=%d",
            stats.documents,
            stats.residue_statements,
            stats.moved_statements,
            stats.changed_token_positions,
        )
        _augmentation_log_count += 1
    return result


class AminoAcidAugmentedDataset(AsyncDataset[LmExample]):
    """Apply deterministic, occurrence-indexed training augmentation."""

    def __init__(self, dataset: AsyncDataset[LmExample], seed: int):
        self.dataset = dataset
        self.seed = seed

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        examples = await self.dataset.get_batch(indices)
        return [
            _augment_lm_example(example, seed=self.seed, index=index)
            for index, example in zip(indices, examples, strict=True)
        ]


def _validate_contacts_v1_tokenizer(data: LmDataConfig) -> None:
    tokenizer = data.the_tokenizer
    observed = tokenizer.convert_tokens_to_ids(list(CONTACTS_V1_TOKEN_IDS))
    expected = list(CONTACTS_V1_TOKEN_IDS.values())
    if observed != expected or len(tokenizer) != VOCAB_SIZE:
        message = f"contacts-v1 tokenizer contract changed: {observed=}, {expected=}, vocab_size={len(tokenizer)}"
        raise ValueError(message)


@dataclass(frozen=True)
class AminoAcidAugmentedDataConfig(LmDataConfig):
    """LmDataConfig variant that augments only the indexed training stream."""

    augmentation_seed: int = AA_AUGMENTATION_SEED

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        dataset = super().train_set(Pos, batch_schedule, key=key)
        return AminoAcidAugmentedDataset(dataset, self.augmentation_seed)


def augment_amino_acids(data: LmDataConfig) -> LmDataConfig:
    """Enable exp166's training-only sequence-statement augmentation."""
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return AminoAcidAugmentedDataConfig(**values)


def training_env() -> dict[str, str]:
    """Return non-secret W&B routing from the launch environment."""
    required = ("WANDB_ENTITY", "WANDB_PROJECT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    env = {
        "WANDB_ENTITY": os.environ["WANDB_ENTITY"],
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _apply_exp166_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    region: str,
) -> ArtifactStep[LevanterCheckpoint]:
    """Apply exp166's data semantics and fixed v6e-4 microbatch shape."""
    base_build_config = step.build_config

    def build_config(ctx):
        _regional_prefix_guard(ctx, region)
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=None,
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
        )
        data = augment_amino_acids(data)
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=PER_DEVICE_PARALLELISM,
                per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
            )
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=DATA_SEED,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build(region: str) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the reviewed-only two-step cache compatibility smoke test."""
    afdb = afdb_cache(region)
    esm = esm_cache(region)
    validation = validation_cache(region)
    env = training_env()
    run_id = regional_run_id(region)
    step = train_lm(
        name=f"checkpoints/{run_id}",
        run_id=run_id,
        model=MODEL_CONFIG,
        optimizer=AdamConfig(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={afdb: 0.5, esm: 0.5},
        validation=[validation],
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=None,
        evals=None,
        resources=train_resources(region),
        tensor_parallel_size=1,
        steps_per_eval=STEPS_PER_EVAL,
        wandb_project=env["WANDB_PROJECT"],
        wandb_group=RUN_PREFIX,
        tags=[
            "protein",
            "exp199",
            "contacts-v1",
            "smoke",
            f"params={MODEL_PARAMS}",
            f"tokens={TRAIN_TOKENS}",
            f"tokenizer={TOKENIZER}",
            f"region={region}",
            f"tpu={TPU}",
        ],
        env_vars=env,
    )
    return _apply_exp166_overrides(step, region=region)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    return build(parse_region())


if __name__ == "__main__":
    main()
