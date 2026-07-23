# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for exp137: train a 1.5B protein LM on the
contacts-and-crops-v1 corpus, from scratch, reproducing Eric's exp117 recipe
(issue #137).

exp137 is the 8k-crops analog of the contacts-v1 tuning lineage (#75 -> exp117):
a fresh Qwen3 1.47B trained on the **contacts-and-crops-v1** corpus (exp130/#130
format, exp132/#132 corpus -- 8192-token coordinate documents: contacts + Pass-1
coarse boxes + a few Pass-2 fine crops). Because crops is an 8192-token format,
which is exp117's *native* sequence length, Eric's recipe transfers 1:1 with no
RoPE re-tuning.

Key differences from exp120 (the warm-start template this is adapted from):

* **Train from scratch** (random init) -- no ``initialize_from_checkpoint_path``,
  no ``pad_tokenizer_to_match_model``. This reproduces exp117's from-scratch
  sweep, not a continue-train.
* **Crops tokenizer** (``timodonnell/contacts-and-crops-v1-tokenizer``, 3848
  vocab), a strict superset of the contacts-v1 tokenizer -- ids 2..2845 are
  byte-identical, then the ``<xyz-000..999>`` coordinate tokens (2846..3846) and
  a single ``<crop>`` token (3847). Because contacts-v1 documents tokenize
  identically under it, ``eval/contacts-v1-val/loss`` is directly comparable to
  Eric's exp117 leader (2.7112).
* **Single train corpus** (crops train, weight 1.0) + **two validation sets**:
  ``contacts-and-crops-v1-val`` (primary) and ``contacts-v1-val`` (secondary,
  read from exp53's published contacts-v1 val split).

The training stream is **shuffled** (full Feistel permutation, fixed ``data_seed``)
because the corpus shards are round/pLDDT-ordered; validation is read
sequentially and we eval the FULL split each time (``max_eval_batches=None``).
"""

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

from fray import ResourceConfig
import numpy as np
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution import ExecutorStep, output_path_of, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

# Region bucket for ALL data + executor output (caches, checkpoints, HF exports).
# Default us-east5 (where the corpus was first staged); override with EXP137_BUCKET
# to run co-located in another region (e.g. gs://marin-us-east1 when us-east5 TPU
# capacity is scarce -- marin HARD-BLOCKS training when the TPU zone's region !=
# the data region, so the corpus must be mirrored to the same-region bucket first;
# see mirror_crops_corpus.py / the region note in README). Path layout is identical
# across regions -- only the bucket changes.
_BUCKET = os.environ.get("EXP137_BUCKET", "gs://marin-us-east5").rstrip("/")
_ROOT = f"{_BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{_ROOT}/exp137_contacts_and_crops_v1_1_5b"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

# contacts-and-crops-v1 tokenizer (3848 vocab, superset of contacts-v1). Use the
# BARE repo id -- the ``repo@rev`` form is rejected by huggingface_hub's
# validate_repo_id on the training tokenizer-load path (verbatim rationale from
# exp85/exp120). The repo is single-revision; the pin below is documentation only.
CROPS_TOKENIZER_REPO = "timodonnell/contacts-and-crops-v1-tokenizer"
CROPS_TOKENIZER_REVISION = "80fe4ee788708cb96e1de6ef74309a71d42c8323"  # documented pin
CROPS_TOKENIZER = CROPS_TOKENIZER_REPO

# --- Corpora (region-local GCS working copies, one text column ``document``) ---
# The crops corpus is published to the open-athena HF *bucket*, which is NOT
# levanter/fsspec-addressable on the worker (HfFileSystem 404s -- the recurring
# exp53/85/108/120 gotcha), so exp132's shards are mirrored byte-for-byte to GCS
# under the exp132 data prefix (see mirror_crops_corpus.py). Explicit ``*.parquet``
# globs so neither marin's expand_tokenize_paths nor levanter's URL globber falls
# back to the default ``**/*.json.gz`` pattern.
CROPS_DATA_PREFIX = f"{_ROOT}/exp132_contacts_and_crops_v1/documents"
CROPS_TRAIN_GLOB = f"{CROPS_DATA_PREFIX}/train/*.parquet"
CROPS_VAL_GLOB = f"{CROPS_DATA_PREFIX}/val/*.parquet"
# contacts-v1 val split -- exp53's published corpus. Tokenized with the CROPS
# tokenizer (superset -> contacts-v1 docs tokenize identically), so this loss is
# directly comparable to Eric's exp117 contacts-v1-val leader (2.7112).
CONTACTS_V1_VAL_GLOB = f"{_ROOT}/exp53_contacts_v1_5x/documents/val/*.parquet"
# contacts-v1 TRAIN split -- exp53's published corpus (same proteins/rounds as the
# crops corpus, one contacts-v1 doc per). Used ONLY for the optional mix-in variant
# (a token-minority alongside the crops bulk, a la #121, to keep pure-contacts
# capability sharp). Tokenized with the crops tokenizer (subset ids).
CONTACTS_V1_TRAIN_GLOB = f"{_ROOT}/exp53_contacts_v1_5x/documents/train/*.parquet"
# contacts-v1 ESMFold2-Atlas distillation corpus (exp139, issue #139): contacts-v1
# format documents from ~67M ESM-Atlas proteins (~71.4B tokens), same tokenizer as
# contacts-v1 (subset of the crops tokenizer). Train-only. Used ONLY for the 3-way
# mix-in variant. Mirrored to GCS via mirror_on_pod.py (MIRROR_SRC_PREFIX=.../train).
ESM_ATLAS_TRAIN_GLOB = f"{_ROOT}/exp139_contacts_v1_esm_atlas/documents/train/*.parquet"

# Qwen3 1.47B -- exp117 / #75 / exp44 dims + Llama3 rope, verbatim. Vocab is set
# from the tokenizer (3848) at build time. Do NOT change the architecture.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

CROPS_DATA_SEED = 0  # exp117 uses data_seed=0

# Shuffle policy. Default "block" == Eric's exp117 EXACTLY: a hierarchical Feistel
# BLOCK shuffle (BlockShuffleConfig(io_block_size=256, window_blocks=512, feistel),
# also the levanter LmDataConfig default -> ds.block_shuffle). EXP137_SHUFFLE=full
# selects `shuffle=True`, which levanter routes to a FULL Feistel permutation
# (ds.shuffle) -- a real divergence on the round/pLDDT-ordered corpus (full perm
# mixes all rounds uniformly; block preserves macro round-order, mixing within
# ~131k-sequence windows). The two original runs launched on "full" (before the fix
# was found); use EXP137_SHUFFLE=full to RESUME them consistently.
_SHUFFLE_MODE = os.environ.get("EXP137_SHUFFLE", "block")
CROPS_SHUFFLE: object = (
    True if _SHUFFLE_MODE == "full"
    else BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
)

# Default TPU slice. exp137 is a long (~75B-token) from-scratch run, so we target
# a large dedicated v5p slice in us-east5 (in-region with the marin-us-east5 data
# mirror + checkpoint bucket). Overridable by env EXP137_TPU / EXP137_ZONE for
# allocation fallback (v5p-128 -> v5p-64 -> v5p-32 -> v6e-*), see the launcher.
_TPU_TYPE = os.environ.get("EXP137_TPU", "v5p-128")
_TPU_ZONE = os.environ.get("EXP137_ZONE", "us-east5-a")
# Placement can be a single ZONE (default) or a REGION list (Eric-style,
# `regions=[region]`). v5p gangs only register autoscaler demand under the region
# form -- a zone= v5p request sits in "coscheduling" with Demand=0 forever, whereas
# regions=["us-east5"] binds (matches exp117_sweep.py's with_tpu(regions=...)).
# Set EXP137_REGION (e.g. "us-east5") to use the region form.
_TPU_REGION = os.environ.get("EXP137_REGION")
_TPU_SLICES = int(os.environ.get("EXP137_SLICES", "1"))
_PLACEMENT = {"regions": [_TPU_REGION.lower()]} if _TPU_REGION else {"zone": _TPU_ZONE}
PROTEIN_RESOURCES = ResourceConfig.with_tpu(
    _TPU_TYPE, slice_count=_TPU_SLICES, cpu=32, ram="128g", disk="50g", **_PLACEMENT,
)


class ArrayExemplarBatchTokenizer(BatchTokenizer):
    """BatchTokenizer variant whose exemplar treats token sequences as array leaves."""

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        examples = super().__call__(batch)
        return [{key: np.asarray(value) for key, value in example.items()} for example in examples]

    @property
    def output_exemplar(self) -> dict[str, np.ndarray]:
        exemplar = super().output_exemplar
        return {key: np.asarray(value) for key, value in exemplar.items()}


@dataclasses.dataclass(frozen=True)
class ArrayExemplarTextLmDatasetFormat(TextLmDatasetFormat):
    """Text format that keeps packing semantics but matches the array cache-ledger field.

    Belt-and-braces against the marin cache-ledger reader bug (#6008/#6014) on the
    TPU worker; a no-op for a correctly-built fresh cache. Verbatim from exp120.
    """

    def build_preprocessor(
        self, tokenizer: Any, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> ArrayExemplarBatchTokenizer:
        return ArrayExemplarBatchTokenizer(
            tokenizer,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            text_field=self.text_key,
        )


TextLmDatasetFormat.register_subclass("array_exemplar_text", ArrayExemplarTextLmDatasetFormat)


def _tokenize_step(name: str, glob: str, *, is_validation: bool) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=glob,
        tokenizer=CROPS_TOKENIZER_REPO,
        format=TextLmDatasetFormat(text_key="document"),
        is_validation=is_validation,
    )


# One tokenize step per corpus. All share the tokenizer + prefix; the executor
# versions each by its (name, dataset) so they materialize distinct caches.
CROPS_TRAIN_TOK = _tokenize_step("contacts-and-crops-v1-train", CROPS_TRAIN_GLOB, is_validation=False)
CROPS_VAL_TOK = _tokenize_step("contacts-and-crops-v1-val", CROPS_VAL_GLOB, is_validation=True)
CONTACTS_V1_VAL_TOK = _tokenize_step("contacts-v1-val", CONTACTS_V1_VAL_GLOB, is_validation=True)
# Built only when the mix-in variant is requested (contacts_v1_mix > 0).
CONTACTS_V1_TRAIN_TOK = _tokenize_step("contacts-v1-train", CONTACTS_V1_TRAIN_GLOB, is_validation=False)
# Built only when the ESM-Atlas mix-in is requested (esm_atlas_mix > 0).
ESM_ATLAS_TRAIN_TOK = _tokenize_step("contacts-v1-esm-atlas-train", ESM_ATLAS_TRAIN_GLOB, is_validation=False)


def _as_array_exemplar_component(component: DatasetComponent) -> DatasetComponent:
    """Read a token cache with an array-backed ``input_ids`` exemplar + packing."""
    return _as_array_exemplar(dataclasses.replace(component, pack=True))


def _as_array_exemplar(component: DatasetComponent) -> DatasetComponent:
    return dataclasses.replace(
        component, format=ArrayExemplarTextLmDatasetFormat(text_key="document")
    )


def _component(step: ExecutorStep) -> DatasetComponent:
    return _as_array_exemplar_component(
        step_to_lm_mixture_component(step, include_raw_paths=True)
    )


# The two validation components monitored every eval. Keys become the W&B metric
# namespace: ``eval/<key>/loss``.
#   contacts-and-crops-v1-val  -- primary (the corpus we train on)
#   contacts-v1-val            -- secondary, comparable to Eric's exp117 (2.7112)
VAL_COMPONENTS = {
    "contacts-and-crops-v1-val": _component(CROPS_VAL_TOK),
    "contacts-v1-val": _component(CONTACTS_V1_VAL_TOK),
}


def build_train_step(
    *,
    name: str,
    learning_rate: float = 3.1623e-3,
    lr_schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
    num_train_steps: int,
    train_batch_size: int = 128,
    train_seq_len: int = 8192,
    weight_decay: float = 0.2,
    warmup: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    steps_per_eval: int = 1000,
    steps_per_export: int = 5000,
    max_eval_batches: int | None = None,
    data_seed: int = CROPS_DATA_SEED,
    contacts_v1_mix: float = 0.0,
    esm_atlas_mix: float = 0.0,
    extra_tags: Sequence[str] = (),
    wandb_name: str | None = None,
    resources: ResourceConfig = PROTEIN_RESOURCES,
    override_output_path: str | None = None,
) -> ExecutorStep:
    """Build the exp137 from-scratch crops train step (Eric's exp117 recipe).

    Defaults reproduce exp117's best config: LR 3.1623e-3 (sqrt(10)*1e-3), WD 0.2,
    global batch 128, seq 8192, AdamW betas 0.9/0.95, cosine decay to
    ``min_lr_ratio`` over the run, 10% warmup, unmasked, ``pack=True``, Feistel
    ``data_seed=0``. Only ``num_train_steps`` (and, if swept, ``learning_rate`` /
    ``lr_schedule`` / ``data_seed``) are ``versioned()`` so tuning the other knobs
    doesn't needlessly bust the cache.
    """
    # WANDB_API_KEY is forwarded into the pod (it does NOT inherit the launch
    # shell). Set it at launch (``-e WANDB_API_KEY <key>`` or os.environ).
    env_vars = {"WANDB_ENTITY": "open-athena"}
    _wandb_key = os.environ.get("WANDB_API_KEY")
    if _wandb_key:
        env_vars["WANDB_API_KEY"] = _wandb_key

    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=train_batch_size,
        num_train_steps=versioned(num_train_steps),
        learning_rate=versioned(learning_rate),
        lr_schedule=versioned(lr_schedule),
        min_lr_ratio=min_lr_ratio,
        weight_decay=weight_decay,
        warmup=warmup,
        beta1=beta1,
        beta2=beta2,
        train_seq_len=train_seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        max_eval_batches=max_eval_batches,
        data_seed=versioned(data_seed),
        env_vars=env_vars,
        # From scratch: no initialize_from_checkpoint_path, no pad_tokenizer.
    )

    # Train mixture. Default: crops-only (weight 1.0). With contacts_v1_mix > 0,
    # add a token-minority of standalone contacts-v1 documents (exp53, same
    # proteins) so that fraction of TRAINING tokens comes from contacts-v1 and the
    # rest from crops (a la #121, to keep pure-contacts capability sharp). The two
    # runs share the same num_train_steps (token budget), so they A/B cleanly.
    if contacts_v1_mix < 0 or esm_atlas_mix < 0 or contacts_v1_mix + esm_atlas_mix >= 1.0:
        raise ValueError(
            f"contacts_v1_mix ({contacts_v1_mix}) + esm_atlas_mix ({esm_atlas_mix}) must be "
            f"in [0, 1) so crops keeps a positive weight"
        )
    crops_key = "contacts-and-crops-v1-train"
    components = {crops_key: _component(CROPS_TRAIN_TOK), **VAL_COMPONENTS}
    weights: dict[str, float] = {crops_key: 1.0 - contacts_v1_mix - esm_atlas_mix}
    if contacts_v1_mix > 0.0:
        components["contacts-v1-train"] = _component(CONTACTS_V1_TRAIN_TOK)
        weights["contacts-v1-train"] = contacts_v1_mix
    if esm_atlas_mix > 0.0:
        components["contacts-v1-esm-atlas-train"] = _component(ESM_ATLAS_TRAIN_TOK)
        weights["contacts-v1-esm-atlas-train"] = esm_atlas_mix
    train_weights = {**weights, **{k: 0.0 for k in VAL_COMPONENTS}}

    crops_data = LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=CROPS_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=CROPS_SHUFFLE,  # Eric's exp117 block shuffle (was shuffle=True == full perm)
        block_cross_document_attention=True,  # == levanter default == Eric's exp117
    )

    if contacts_v1_mix <= 0.0 and esm_atlas_mix <= 0.0:
        mix_tag = "crops-only"
    else:
        mix_tag = f"cv1mix{contacts_v1_mix:g}-esmmix{esm_atlas_mix:g}"
    return default_train(
        name=name,
        tokenized=crops_data,
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=[
            "protein", "contacts-and-crops-v1", "qwen3", "unmasked", "exp137",
            "from-scratch", mix_tag, *extra_tags,
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp137-contacts-and-crops-v1",
        wandb_name=wandb_name or name,
        override_output_path=override_output_path,
    )


def build_hf_export_step(
    *,
    train_step: ExecutorStep,
    checkpoint_step: int,
    name_prefix: str,
    checkpoint_path_override: str | None = None,
) -> ExecutorStep:
    """CPU-only HF export of a ``step-{checkpoint_step}`` checkpoint (tokenizer co-located)."""
    from copy import deepcopy

    from levanter.trainer import TrainerConfig
    from marin.export import convert_checkpoint_to_hf_step

    trainer = train_step.config.train_config.trainer
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"Expected TrainerConfig on train_step, got {type(trainer)!r}")

    if checkpoint_path_override is not None:
        checkpoint_path: str | object = checkpoint_path_override
    else:
        checkpoint_path = output_path_of(train_step, f"checkpoints/step-{checkpoint_step}")

    return convert_checkpoint_to_hf_step(
        name=f"hf/{name_prefix}-step-{checkpoint_step}",
        checkpoint_path=checkpoint_path,  # pyrefly: ignore
        trainer=deepcopy(trainer),
        model=MODEL_CONFIG,
        tokenizer=CROPS_TOKENIZER,
        use_cpu=True,
        discover_latest=False,
    )


__all__ = [
    "CROPS_TOKENIZER",
    "MARIN_PREFIX",
    "MODEL_CONFIG",
    "PROTEIN_RESOURCES",
    "VAL_COMPONENTS",
    "build_hf_export_step",
    "build_train_step",
]
