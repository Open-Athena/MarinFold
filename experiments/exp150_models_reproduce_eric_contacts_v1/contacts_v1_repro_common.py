# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared config for exp150: reproduce Eric's best contacts-v1 1.5B run exactly (issue #150).

Target: his exp117 point ``e16-lr3p162e-3-wd0p2-bs128`` (W&B run
``prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs128-us-east5`` in
``eric-czech/marin``), final ``eval/tokenized/contacts-v1-val/loss`` = **2.7112**.

Everything below was read out of ``marin@origin/eac/plm-exp117:experiments/protein/
exp117_sweep.py`` (``e0f3da1``) -- not inferred from issue prose. Where his script
relies on a library default rather than setting a value, we set it explicitly to
that same default and say so, so the whole recipe is legible in one place.

**What is deliberately NOT held fixed: the training harness.** That is the thing
under test. Eric drives marin's newer ``marin.experiment.train.train_lm`` +
``StepRunner`` + ``fray``; this module drives MarinFold's own
``marinfold_models.defaults.default_train`` + ``executor_main`` on marin ``<0.3``.
Both configure a levanter ``TrainLmConfig``, so the losses should agree; whether
they actually do is the experiment.

Adapted from exp137's ``crops_train_common.py`` (same from-scratch native path,
same exp117 recipe) with the corpus and tokenizer swapped back to contacts-v1.
Differences from exp137 worth knowing:

* **contacts-v1 tokenizer** (2845 vocab) instead of the crops 3848 superset.
* **One val component** (``contacts-v1-val``) -- the metric being reproduced.
* **num_train_steps = 71,360**, recomputed with Eric's own formula rather than
  copied (exp137 used 71,359; see ``steps_for_epochs`` below).
* **Block shuffle only.** exp137 kept a ``full`` escape hatch to resume runs
  launched before the block-shuffle fix was found; exactness is the whole point
  here, so this module offers no such override.
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

# Region bucket for ALL executor output (token caches, checkpoints, HF exports).
# us-east5 is where exp53's contacts-v1 corpus lives AND the region Eric's target
# run used, so nothing needs mirroring. marin HARD-BLOCKS training when the TPU
# zone's region != the data region, so the slice must be co-located (see README).
_BUCKET = os.environ.get("EXP150_BUCKET", "gs://marin-us-east5").rstrip("/")
_ROOT = f"{_BUCKET}/protein-structure/MarinFold"
MARIN_PREFIX = f"{_ROOT}/exp150_reproduce_eric_contacts_v1"
os.environ["MARIN_PREFIX"] = MARIN_PREFIX

# contacts-v1 tokenizer (2845 vocab). Eric pins ``…@5d68a24a899f``; MarinFold's
# tokenizer-load path rejects the ``repo@rev`` form (huggingface_hub's
# validate_repo_id -- the recurring exp85/exp120 gotcha), so we pass the BARE repo
# id. The repo is single-revision, so this is the same bytes; the pin below is
# documentation. Verified equal by the token-count check in the README.
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"  # Eric's TOKENIZER pin; not passed as repo@rev
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO
VOCAB_SIZE = 2845  # exp117 VOCAB_SIZE

# --- Corpus (exp53 contacts-v1, one text column ``document``) ----------------
# These are the same shards Eric's region-relative TRAIN_DOCS / VAL_DOCS resolve
# to under gs://marin-us-east5. Explicit ``*.parquet`` globs so neither marin's
# expand_tokenize_paths nor levanter's URL globber falls back to the default
# ``**/*.json.gz`` pattern.
CONTACTS_V1_DATA_PREFIX = f"{_ROOT}/exp53_contacts_v1_5x/documents"
CONTACTS_V1_TRAIN_GLOB = f"{CONTACTS_V1_DATA_PREFIX}/train/*.parquet"
CONTACTS_V1_VAL_GLOB = f"{CONTACTS_V1_DATA_PREFIX}/val/*.parquet"

# Exact train-corpus token count at this tokenizer -- exp117 ``TRAIN_TOKENS``.
# Steps/epoch are DERIVED from this (Eric's formula), never estimated. Our
# tokenize step must report exactly this number; if it doesn't, the corpus or
# tokenizer diverged and the comparison is void (see README "cache equality").
TRAIN_TOKENS = 4_676_753_425

# --- Model: Qwen3 1.47B, exp117 MODEL_CONFIG verbatim ------------------------
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# --- Fixed recipe constants (exp117 module-level values) ---------------------
SEQ_LEN = 8192                 # exp117 SEQ_LEN
DATA_SEED = 0                  # exp117 DATA_SEED
WARMUP = 0.1                   # exp117 WARMUP
LR_SCHEDULE = "cosine"         # exp117 LR_SCHEDULE
NUM_EVALS_PER_EPOCH = 2        # exp117 NUM_EVALS_PER_EPOCH
# exp117 SHUFFLE: a hierarchical Feistel BLOCK shuffle (also levanter's
# LmDataConfig default). NOT ``shuffle=True``, which levanter routes to a FULL
# Feistel permutation -- a real divergence on this round/pLDDT-ordered corpus
# (full perm mixes all rounds uniformly; block preserves macro round-order,
# mixing within ~131k-sequence windows). exp67/85/120 used the full permutation;
# exp137 found the discrepancy. Reproducing Eric means block.
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")

# The exp117 target point.
TARGET_EPOCHS = 16
TARGET_LR = 3.1623e-3
TARGET_WD = 0.2
TARGET_BATCH_SIZE = 128
TARGET_VAL_LOSS = 2.7112  # what we are trying to reproduce


def steps_per_epoch(batch_size: int = TARGET_BATCH_SIZE, seq_len: int = SEQ_LEN) -> int:
    """Steps for one pass over the train corpus -- exp117 ``Point.steps_per_epoch``.

    ``round(TRAIN_TOKENS / (batch_size * seq_len))``. At bs128/seq8192 this is
    **4,460** (4,460.0996 before rounding). Cross-check: #75's 8-epoch run ended
    at ``step-35679``, i.e. 35,680 = 8 x 4,460.
    """
    return round(TRAIN_TOKENS / (batch_size * seq_len))


def steps_for_epochs(
    epochs: int = TARGET_EPOCHS,
    batch_size: int = TARGET_BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> int:
    """exp117 ``Point.num_train_steps`` = ``epochs * steps_per_epoch``.

    16 epochs at bs128 -> **71,360** steps = 74.86B tokens. (exp137 used 71,359,
    apparently taking the final step INDEX for the count; a 1-step difference
    that is immaterial there but pointless to inherit here.)
    """
    return epochs * steps_per_epoch(batch_size, seq_len)


def evals_per_epoch_steps(batch_size: int = TARGET_BATCH_SIZE) -> int:
    """exp117 ``Point.steps_per_eval`` = ``round(steps_per_epoch / 2)`` -> 2,230."""
    return max(1, round(steps_per_epoch(batch_size) / NUM_EVALS_PER_EPOCH))


# --- TPU slice ---------------------------------------------------------------
# Must be co-located with us-east5 (marin blocks cross-region training). Placement
# can be a single ZONE or a REGION list; v5p gangs only register autoscaler demand
# under the region form (a zone= v5p request sits in "coscheduling" with Demand=0
# forever), which is what exp117_sweep.py itself uses -- ResourceConfig.with_tpu(
# tpu, regions=singleton_region_list(region)). Default to the region form here.
_TPU_TYPE = os.environ.get("EXP150_TPU", "v5p-128")
_TPU_REGION = os.environ.get("EXP150_REGION", "us-east5")
_TPU_ZONE = os.environ.get("EXP150_ZONE")  # set to force the zone form instead
_TPU_SLICES = int(os.environ.get("EXP150_SLICES", "1"))
_PLACEMENT = {"zone": _TPU_ZONE} if _TPU_ZONE else {"regions": [_TPU_REGION.lower()]}
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
    TPU worker; a no-op for a correctly-built fresh cache. Verbatim from exp120/137.
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
        tokenizer=CONTACTS_V1_TOKENIZER_REPO,
        format=TextLmDatasetFormat(text_key="document"),
        is_validation=is_validation,
    )


TRAIN_TOK = _tokenize_step("contacts-v1-train", CONTACTS_V1_TRAIN_GLOB, is_validation=False)
VAL_TOK = _tokenize_step("contacts-v1-val", CONTACTS_V1_VAL_GLOB, is_validation=True)


def _component(step: ExecutorStep) -> DatasetComponent:
    """Read a token cache with an array-backed ``input_ids`` exemplar + packing.

    ``pack=True`` on every component is exp117's ``_apply_recipe_overrides``
    (pack-prefix-only: documents are never concat-and-split).
    """
    component = step_to_lm_mixture_component(step, include_raw_paths=True)
    component = dataclasses.replace(component, pack=True)
    return dataclasses.replace(
        component, format=ArrayExemplarTextLmDatasetFormat(text_key="document")
    )


# The single validation component. Its key is the W&B metric namespace, so this
# logs ``eval/contacts-v1-val/loss``. (Eric's component key is
# ``tokenized/contacts-v1-val`` -> ``eval/tokenized/contacts-v1-val/loss``; the
# storage-path prefix is baked into his key. Same series, cosmetically renamed.)
VAL_COMPONENTS = {"contacts-v1-val": _component(VAL_TOK)}


def build_train_step(
    *,
    name: str,
    learning_rate: float = TARGET_LR,
    weight_decay: float = TARGET_WD,
    num_train_steps: int | None = None,
    train_batch_size: int = TARGET_BATCH_SIZE,
    train_seq_len: int = SEQ_LEN,
    lr_schedule: str = LR_SCHEDULE,
    warmup: float = WARMUP,
    steps_per_eval: int | None = None,
    steps_per_export: int | None = None,
    max_eval_batches: int | None = None,
    data_seed: int = DATA_SEED,
    extra_tags: Sequence[str] = (),
    wandb_name: str | None = None,
    resources: ResourceConfig = PROTEIN_RESOURCES,
    override_output_path: str | None = None,
) -> ExecutorStep:
    """Build the exp150 from-scratch train step -- Eric's exp117 recipe, verbatim.

    All defaults reproduce ``e16-lr3p162e-3-wd0p2-bs128``. ``num_train_steps``,
    ``steps_per_eval`` and ``steps_per_export`` default to ``None`` and are then
    DERIVED with Eric's own formulas (71,360 / 2,230 / 4,460) rather than
    hard-coded, so changing the batch size or epoch count stays self-consistent.

    Optimizer values below that Eric leaves to levanter's ``AdamConfig`` defaults
    -- betas 0.9/0.95, epsilon 1e-8, max_grad_norm 1.0, min_lr_ratio 0.1 -- are
    passed explicitly at those same default values. Verified against
    ``levanter/optim/config.py`` (``AdamConfig``/``OptimizerConfig``); stating
    them means a future levanter default change can't silently move the recipe.
    """
    if num_train_steps is None:
        num_train_steps = steps_for_epochs(TARGET_EPOCHS, train_batch_size, train_seq_len)
    if steps_per_eval is None:
        steps_per_eval = evals_per_epoch_steps(train_batch_size)
    if steps_per_export is None:
        # exp117 production_shape: one permanent checkpoint per epoch, alongside
        # the rolling resumption checkpoint.
        steps_per_export = steps_per_epoch(train_batch_size, train_seq_len)

    # WANDB_API_KEY is forwarded into the pod (it does NOT inherit the launch
    # shell). Set it at launch (``-e WANDB_API_KEY <key>`` or os.environ).
    env_vars = {"WANDB_ENTITY": "open-athena"}
    if _wandb_key := os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = _wandb_key

    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=train_batch_size,
        num_train_steps=versioned(num_train_steps),
        learning_rate=versioned(learning_rate),
        lr_schedule=versioned(lr_schedule),
        weight_decay=weight_decay,
        warmup=warmup,
        # --- levanter AdamConfig defaults, stated explicitly (see docstring) ---
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        min_lr_ratio=0.1,
        z_loss_weight=None,  # exp117 train_lm(z_loss_weight=None)
        # ----------------------------------------------------------------------
        train_seq_len=train_seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        # Eric runs no in-training HF export; -1 disables it (None would inherit
        # steps_per_export and burn pod time exporting every epoch). We export the
        # final checkpoint separately with export_checkpoints.py.
        steps_per_hf_export=-1,
        # exp117 pins max_eval_batches=None explicitly: the FULL held-out val
        # split, no downsampling. Also levanter's default; pinned for the same
        # reason he pins it.
        max_eval_batches=max_eval_batches,
        data_seed=versioned(data_seed),
        env_vars=env_vars,
        # From scratch: no initialize_from_checkpoint_path.
    )

    train_key = "contacts-v1-train"
    data = LmDataConfig(
        components={train_key: _component(TRAIN_TOK), **VAL_COMPONENTS},
        train_weights={train_key: 1.0, **{k: 0.0 for k in VAL_COMPONENTS}},
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=SHUFFLE,
        # levanter's default; == exp117 (which leaves it unset).
        block_cross_document_attention=True,
    )

    return default_train(
        name=name,
        tokenized=data,
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=[
            "protein", "contacts-v1", "qwen3", "unmasked", "exp150",
            "from-scratch", "reproduction", "exp117-e16-lr3p162e-3-wd0p2-bs128",
            *extra_tags,
        ],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp150-reproduce-eric-contacts-v1",
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
        tokenizer=CONTACTS_V1_TOKENIZER,
        use_cpu=True,
        discover_latest=False,
    )


__all__ = [
    "CONTACTS_V1_TOKENIZER",
    "MARIN_PREFIX",
    "MODEL_CONFIG",
    "PROTEIN_RESOURCES",
    "TARGET_VAL_LOSS",
    "TRAIN_TOKENS",
    "VAL_COMPONENTS",
    "VOCAB_SIZE",
    "build_hf_export_step",
    "build_train_step",
    "steps_for_epochs",
    "steps_per_epoch",
]
