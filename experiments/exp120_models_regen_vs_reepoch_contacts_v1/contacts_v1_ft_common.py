# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the exp120 regenerated-vs-re-epoch fine-tune (issue #120).

We **continue-train** eric-czech's tuned contacts-v1 1.5B (the #75 sweep winner,
``prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084``, eval loss 2.7566) under two
matched arms that differ **only in the training document content**:

* **Arm A (baseline / re-epoch):** the ORIGINAL contacts-v1 round-0 documents.
* **Arm B (treatment / regenerated):** the exp100 only-correct regenerated
  documents for the SAME proteins.

Both arms share this module: same base checkpoint, same Qwen3 config, same
tokenizer, same optimizer / LR schedule, same seed handling, same held-out
validation sets and eval protocol. The per-arm launcher
(``train_regen_vs_reepoch_sweep.py``) picks only the train corpus + run name.

Key differences from exp85 (the #67 Llama warm-restart this is modelled on):

* **Model is Qwen3, not Llama.** The base checkpoint is Eric's Qwen3 1.47B, so
  ``MODEL_CONFIG`` below is the exp75 ``Qwen3Config`` (verbatim from
  exp89's ``export_contacts_v1_best_to_hf.py``). ``default_train`` is
  architecture-agnostic, so passing a ``Qwen3Config`` is the only change needed.
* **Warm-start from Eric's checkpoint**, not #67's — see ``INIT_CHECKPOINT``.
* **Fresh caches under an exp120 prefix.** exp85 reused exp67's existing caches;
  exp120 tokenizes NEW corpora (Arm A aligned round-0, Arm B regenerated, and a
  regenerated val split), so we build fresh caches under our own ``MARIN_PREFIX``.
  We keep exp85's ``ArrayExemplarTextLmDatasetFormat`` read wrapper anyway: it
  makes the cache reader derive the ledger field ``input_ids`` (belt-and-braces
  against the marin cache-ledger reader bug #6008/#6014 on the TPU worker), and
  is a no-op for a correctly-built fresh cache.
* **Three validation sets monitored every eval** (issue #120 + Tim's comment):
  the full original val split (canonical, anchors the step-0 ≈ 2.7566 sanity),
  the regenerated round-0 val, and the original round-0 val (the apples-to-apples
  partner of the regenerated one — same proteins, one realization each).

The training stream is **shuffled** (full Feistel permutation, fixed ``data_seed``)
because the corpus shards are round/pLDDT-ordered; validation is read
sequentially and we eval the FULL split each time (``max_eval_batches=None``) —
the val splits are small.
"""

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

from fray import ResourceConfig
import numpy as np
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution import ExecutorStep, output_path_of, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

# All exp120 marin-executor output (token caches, checkpoints, HF exports) lands
# under this prefix (per AGENTS.md). Unlike exp85 (which reused exp67's prefix to
# resolve existing caches) exp120 builds FRESH caches for new corpora, so it gets
# its own prefix. The data-prep scripts write the corpora under the same tree.
CONTACTS_V1_MARIN_PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp120_regen_vs_reepoch_contacts_v1"
)
os.environ["MARIN_PREFIX"] = CONTACTS_V1_MARIN_PREFIX

# contacts-v1 tokenizer (2845 vocab). Use the BARE repo id everywhere — the
# ``repo@rev`` form is rejected by huggingface_hub's validate_repo_id on the
# training tokenizer-load path (this marin/levanter build does not split @rev
# there). The repo is a stable single-revision tokenizer; the pinned revision is
# documentation only. (Verbatim rationale from exp85.)
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"  # documented pin; not passed as repo@rev
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO

# --- Corpora (region-local GCS working copies, one text column ``document``) ---
# The regenerated docs published to the HF *bucket* are NOT levanter-addressable
# (HfFileSystem 404s them on the worker), so the data-prep step mirrors them to
# GCS. Arm A / the regen-val are built by the data-prep scripts in this dir.
# Explicit ``*.parquet`` globs so neither marin's expand_tokenize_paths nor
# levanter's URL globber falls back to the default ``**/*.json.gz`` pattern.
ARM_A_TRAIN_GLOB = f"{CONTACTS_V1_MARIN_PREFIX}/data/orig_r0_train/*.parquet"
ARM_B_TRAIN_GLOB = f"{CONTACTS_V1_MARIN_PREFIX}/data/regen_train/*.parquet"
# Full original val split — read straight from exp53's published corpus (all
# rounds; the canonical split Eric's 2.7566 was reported on).
VAL_FULL_GLOB = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/val/*.parquet"
)
VAL_ORIG_R0_GLOB = f"{CONTACTS_V1_MARIN_PREFIX}/data/orig_r0_val/*.parquet"
VAL_REGEN_GLOB = f"{CONTACTS_V1_MARIN_PREFIX}/data/regen_val/*.parquet"

# Eric's #75 tuned checkpoint (Qwen3 1.47B, eval loss 2.7566 @ step 35679). We
# warm-start model weights from here (fresh step-0 / optimizer / LR schedule /
# data loader). Levanter full training-state checkpoint dir:
INIT_CHECKPOINT = (
    "gs://marin-us-east5/checkpoints/"
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679"
)

# exp75 MODEL_CONFIG — Qwen3 1.47B (exp44 dims + Llama3 rope), verbatim from
# exp89's export. MUST match the checkpoint we warm-start from.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

CONTACTS_V1_DATA_SEED = 0

# Pin to us-east5-a: co-locate the TPU with the marin-us-east5 checkpoint bucket
# and Eric's INIT_CHECKPOINT. v5p-8 is enough for a short continue-train; the
# launcher can override to a bigger slice.
PROTEIN_RESOURCES_USE5 = ResourceConfig.with_tpu(
    "v5p-8", slice_count=1, cpu=32, ram="128g", disk="50g", zone="us-east5-a",
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
    """Text format that keeps packing semantics but matches the array cache-ledger field."""

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


# One tokenize step per corpus. All share the tokenizer + prefix; the executor
# versions each by its (name, dataset) so they materialize distinct caches.
ARM_A_TRAIN_TOK = _tokenize_step("contacts-v1-orig-r0-train", ARM_A_TRAIN_GLOB, is_validation=False)
ARM_B_TRAIN_TOK = _tokenize_step("contacts-v1-regen-train", ARM_B_TRAIN_GLOB, is_validation=False)
VAL_FULL_TOK = _tokenize_step("contacts-v1-val-full", VAL_FULL_GLOB, is_validation=True)
VAL_ORIG_R0_TOK = _tokenize_step("contacts-v1-val-orig-r0", VAL_ORIG_R0_GLOB, is_validation=True)
VAL_REGEN_TOK = _tokenize_step("contacts-v1-val-regen", VAL_REGEN_GLOB, is_validation=True)


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


# The three validation components monitored every eval (issue #120 + Tim's
# comment). Keys become the W&B metric namespace: ``eval/<key>/loss``.
#   contacts-v1-val-full   — full original val split (canonical; step-0 ≈ 2.7566)
#   contacts-v1-val-orig   — original round-0 val (matched partner of regen)
#   contacts-v1-val-regen  — regenerated round-0 val ("the regenerated val set")
VAL_COMPONENTS = {
    "contacts-v1-val-full": _component(VAL_FULL_TOK),
    "contacts-v1-val-orig": _component(VAL_ORIG_R0_TOK),
    "contacts-v1-val-regen": _component(VAL_REGEN_TOK),
}
TRAIN_TOK_BY_ARM = {"A": ARM_A_TRAIN_TOK, "B": ARM_B_TRAIN_TOK}


def build_train_step(
    *,
    name: str,
    arm: str,
    learning_rate: float,
    lr_schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
    num_train_steps: int,
    train_batch_size: int = 512,
    train_seq_len: int = 8192,
    weight_decay: float = 0.2,
    warmup: float = 0.1,
    steps_per_eval: int = 100,
    steps_per_export: int = 250,
    max_eval_batches: int | None = None,
    data_seed: int = CONTACTS_V1_DATA_SEED,
    extra_tags: Sequence[str] = (),
    wandb_name: str | None = None,
    resources: ResourceConfig = PROTEIN_RESOURCES_USE5,
    override_output_path: str | None = None,
) -> ExecutorStep:
    """Build one matched-arm continue-train step (warm-start from Eric's #75).

    ``arm`` selects the train corpus ("A" = original round-0, "B" = regenerated);
    everything else is held fixed across arms. Weights warm-start from
    ``INIT_CHECKPOINT`` (fresh step-0 / optimizer / LR schedule / data loader —
    a continue-train, not a resume), so ``learning_rate`` + ``warmup`` +
    ``lr_schedule`` define this run's schedule. Only ``learning_rate``,
    ``num_train_steps``, ``data_seed``, ``lr_schedule`` are ``versioned()`` so
    tuning the other knobs doesn't needlessly bust the cache.
    """
    if arm not in TRAIN_TOK_BY_ARM:
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")

    # WANDB_API_KEY is forwarded into the pod (it does NOT inherit the launch
    # shell). Set it at launch (`-e WANDB_API_KEY <key>` or in os.environ). If
    # unset we omit it and the run fails fast with marin's clear message.
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
        train_seq_len=train_seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        max_eval_batches=max_eval_batches,
        data_seed=versioned(data_seed),
        env_vars=env_vars,
        initialize_from_checkpoint_path=INIT_CHECKPOINT,
        # Eric's Qwen checkpoint very likely padded its vocab to a multiple of 4
        # (2845 -> 2848) for TPU efficiency, so its embedding matrix is wider than
        # the 2845-token tokenizer. This pads the tokenizer up to the checkpoint's
        # vocab so the warm-started model shape matches; it is a no-op if the
        # checkpoint vocab is already 2845. VERIFY at launch via the step-0
        # sanity eval (loss ~2.75); a vocab shape error there means flip this.
        pad_tokenizer_to_match_model=True,
    )

    train_key = f"contacts-v1-train-{'orig' if arm == 'A' else 'regen'}"
    components = {train_key: _component(TRAIN_TOK_BY_ARM[arm]), **VAL_COMPONENTS}
    train_weights = {train_key: 1.0, **{k: 0.0 for k in VAL_COMPONENTS}}

    contacts_v1_data = LmDataConfig(
        components=components,
        train_weights=train_weights,
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
    )

    return default_train(
        name=name,
        tokenized=contacts_v1_data,
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=["protein", "contacts-v1", "qwen3", "unmasked", "exp120", f"arm-{arm}", *extra_tags],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp120-regen-vs-reepoch",
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
    "CONTACTS_V1_MARIN_PREFIX",
    "CONTACTS_V1_TOKENIZER",
    "INIT_CHECKPOINT",
    "MODEL_CONFIG",
    "PROTEIN_RESOURCES_USE5",
    "VAL_COMPONENTS",
    "build_hf_export_step",
    "build_train_step",
]
