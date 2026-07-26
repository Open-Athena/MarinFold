# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the exp163 rollout-refinement fine-tune (issue #163).

We **continue-train** eric-czech's tuned contacts-v1 1.5B (the #75 sweep winner,
``prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084``, eval loss 2.7566 — "E8") to
turn it into a *refiner*: given a protein's sequence plus K noisy candidate
rollouts (the "recon" evidence), emit the TRUE contact set.

A refinement document (built by ``build_refinement_corpus.py``) is a normal
contacts-v1 document with K candidate-rollout blocks prepended as context::

    <contacts-v1> <begin_sequence> ...sequence...
      <CAND> <contact> pi pj ...          candidate block 1  (subsampled/flipped)
      ...                                 0..K such blocks, unordered
    <begin_statements> <contact> ...TRUE contacts... <end>

where the candidate marker ``<CAND>`` reuses the spare token
``<contacts-and-distances-v1>`` (the *other* format's doc-type sentinel — never
emitted inside a contacts-v1 document, so it is collision-proof and distinct
from every begin/end marker). See ``README.md`` / ``build_refinement_corpus.py``.

This module is adapted from exp120's ``contacts_v1_ft_common.py``. Everything
about the warm-start (E8 checkpoint, Qwen3 config, tokenizer, the
``ArrayExemplarTextLmDatasetFormat`` cache-ledger workaround, the TPU resource
config) is carried over verbatim. The exp163-specific changes are:

* a NEW ``MARIN_PREFIX`` (exp163 builds fresh caches for the refinement corpus);
* the train corpus is the refinement corpus (a single arm, no A/B);
* **THE KEY ADDITION** — a packing-aware answer-span ``loss_weight_fn`` applied
  to the TRAIN component only: the LM loss is armed on each document's
  ``<begin_statements> … <end>`` answer span and zeroed everywhere else (the
  sequence header AND every ``<CAND>`` candidate block). The model is trained to
  emit the TRUE contacts from the noisy evidence — never to reproduce the
  sequence or the candidates. See :func:`answer_span_loss_weight`.

The held-out validation component is exp53's canonical original contacts-v1 val
split, monitored **unmasked** every eval (issue #163). It is NOT a refinement
set: it anchors the step-0 warm-start sanity (loss ≈ 2.7566, confirming E8
loaded) and tracks base-task retention (catastrophic-forgetting monitor) while
the model learns to refine.

The training stream is **shuffled** (fixed ``data_seed``); validation is read
sequentially and we eval the FULL split each time (``max_eval_batches=None``).
"""

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fray import ResourceConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution import ExecutorStep, output_path_of, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

# All exp163 marin-executor output (token caches, checkpoints, HF exports) lands
# under this NEW prefix (per AGENTS.md). exp163 builds FRESH caches for the
# refinement corpus, so it gets its own prefix; the data-prep / mirror scripts
# write the corpus shards under the same tree (see SCALE_PLAN.md section B).
REFINE_MARIN_PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp163_rollout_refinement"
)
os.environ["MARIN_PREFIX"] = REFINE_MARIN_PREFIX

# contacts-v1 tokenizer (see exp120 for the full rationale). Use the BARE repo id
# everywhere — the ``repo@rev`` form is rejected by huggingface_hub's
# validate_repo_id on the training tokenizer-load path. The pinned revision is
# documentation only.
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"  # documented pin; not passed as repo@rev
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO

# --- Corpora (region-local GCS working copies, one text column ``document``) ---
# The refinement corpus published to the HF *bucket* is NOT levanter-addressable
# (HfFileSystem 404s it on the worker), so a mirror step stages it to GCS under
# this prefix (template: exp120's ``mirror_regen_train.py``). Explicit
# ``*.parquet`` globs so neither marin's expand_tokenize_paths nor levanter's
# URL globber falls back to the default ``**/*.json.gz`` pattern.
REFINEMENT_TRAIN_GLOB = f"{REFINE_MARIN_PREFIX}/data/refinement_corpus/*.parquet"
# Held-out val = exp53's canonical original contacts-v1 val split (all rounds;
# the split Eric's 2.7566 was reported on). Read straight from exp53's published
# corpus. Kept UNMASKED (see module docstring): step-0 warm-start anchor +
# base-task retention monitor, NOT a refinement metric.
VAL_FULL_GLOB = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/val/*.parquet"
)

# Eric's #75 tuned checkpoint (Qwen3 1.47B, eval loss 2.7566 @ step 35679). We
# warm-start model weights from here (fresh step-0 / optimizer / LR schedule /
# data loader — a continue-train, not a resume). Levanter-native full
# training-state checkpoint dir:
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

# Train on v6e in us-east5-b (verbatim from exp120): the v5p-8 preemptible pool
# is contended and thrashes on short runs, while v6e-preemptible in us-east5-b is
# abundant and in-region with the marin-us-east5 checkpoint bucket. v6e-8 = 8
# chips, per-chip batch 16 at global batch 128. Bump to v6e-16 if a gang OOMs.
PROTEIN_RESOURCES_USE5 = ResourceConfig.with_tpu(
    "v6e-8", slice_count=1, cpu=32, ram="128g", disk="50g", zone="us-east5-b",
)


# ============================================================================
# Answer-span loss mask (the exp163-specific addition; issue #163)
# ============================================================================
#
# Token ids in the contacts-v1 tokenizer are fully determined by
# ``marinfold.document_structures.core.build_tokenizer``: it prepends
# ``<pad>``=0, ``<eos>``=1, then ``all_domain_tokens()`` at id 2+. That fixes:
#
#       <begin_statements>  (BEGIN_STRUCTURE_TOKEN)  ->  9
#       <end>               (END_TOKEN)              -> 10
#
# These are baked in as module constants below so this module imports WITHOUT
# ``marinfold``. That matters: ``marinfold`` pins ``transformers<5`` /
# ``huggingface_hub<1.0``, which clash with the marin training stack (hf>=1.5),
# so it is deliberately NOT a training-env dependency and must not be imported
# on the TPU data-loading worker (the ``loss_weight_fn`` is unpickled there by
# module reference). Where ``marinfold`` IS importable (local step-build, tests,
# the corpus-builder env) the guarded block below ASSERTS the baked-in ids still
# match the authoritative vocab — a loud tripwire on any contacts-v1 vocab drift.
BEGIN_STATEMENTS_ID = 9  # <begin_statements>  (BEGIN_STRUCTURE_TOKEN)
END_ID = 10              # <end>               (END_TOKEN)

try:  # authoritative cross-check; skipped on envs without marinfold (e.g. the TPU worker)
    from marinfold.document_structures.contacts_v1.vocab import (  # noqa: E402
        BEGIN_STRUCTURE_TOKEN as _BEGIN_TOKEN,
        END_TOKEN as _END_TOKEN,
        all_domain_tokens as _all_domain_tokens,
    )

    _SPECIALS = 2  # <pad>=0, <eos>=1 (core.build_tokenizer prepends these)
    _domain = _all_domain_tokens()
    _resolved = (
        _SPECIALS + _domain.index(_BEGIN_TOKEN),
        _SPECIALS + _domain.index(_END_TOKEN),
    )
    assert _resolved == (BEGIN_STATEMENTS_ID, END_ID), (
        f"contacts-v1 vocab drift: resolved <begin_statements>/<end> ids "
        f"{_resolved} != baked-in ({BEGIN_STATEMENTS_ID}, {END_ID}). "
        f"Update refine_ft_common.BEGIN_STATEMENTS_ID / END_ID."
    )
except ImportError:
    pass


def answer_span_loss_weight(tokens: jax.Array) -> jax.Array:
    """Per-position float LM loss weight: 1.0 on each answer span, 0.0 elsewhere.

    An exp163 refinement document is::

        <contacts-v1> <begin_sequence> ...sequence...      <- header (0.0)
          <CAND> <contact> pi pj ...                       <- candidate block(s) (0.0)
          ...
        <begin_statements> <contact> ...TRUE contacts... <end>   <- answer span (1.0)

    and MANY such documents are packed into one 8192-token training sequence
    (``pack=True``). We want loss ONLY where the model must PRODUCE the true
    answer — from the first true ``<contact>`` through the terminal ``<end>``
    (so it also learns to STOP) — and nowhere in the sequence header or the
    noisy ``<CAND>`` candidate blocks.

    Convention (from exp0): ``loss_weight[i]`` weights the loss on predicting
    ``tokens[i+1]``. So to train "predict ``tokens[i+1]``" we set
    ``loss_weight[i] = 1``.

    Packing-safe indicator via two running counts (NOT a single python index —
    there are many spans per packed sequence, and each must re-arm itself)::

        opened[i] = # of <begin_statements> in tokens[:i+1]   (inclusive cumsum)
        closed[i] = # of <end>              in tokens[:i+1]   (inclusive cumsum)
        inside[i] = opened[i] > closed[i]

    Walk ONE document, ``<begin_statements>`` at index b, ``<end>`` at index e
    (b < e). The header and the candidate blocks contain NEITHER token — the
    ``<CAND>`` marker is a different token id — so neither count moves there:

    ======  ==================  ======  =============================================
    pos i   (opened, closed)    inside  weights predicting tokens[i+1] = ...
    ======  ==================  ======  =============================================
    < b     equal               0.0     header / candidate tokens  -> excluded  [req]
    b - 1   equal               0.0     tokens[b] = <begin_statements> -> excluded
    b       closed + 1          1.0     tokens[b+1] = FIRST true <contact>      [ (a) ]
    b..e-1  closed + 1          1.0     each <contact> <pX> <pY> triple         [ (a) ]
    e - 1   closed + 1          1.0     tokens[e] = <end>  -> STOP is trained   [ (b) ]
    e       equal               0.0     tokens[e+1] = packed <eos>/next doc/pad -> excluded
    > e     equal               0.0     excluded
    ======  ==================  ======  =============================================

    Because both cumsums keep climbing across the packed sequence, the strict
    ``>`` re-arms at every document's ``<begin_statements>`` and disarms at its
    ``<end>`` — no per-document index, fully vectorised, jit/TPU-safe. The counts
    stay balanced because each refinement document emits exactly one
    ``<begin_statements>`` and one ``<end>`` (the header emits neither; a
    ``<CAND>`` block emits neither), so ``opened`` never leads ``closed`` by more
    than 1.

    Note the inclusive cumsum already supplies the exp0 ``loss_weight[i] ->
    tokens[i+1]`` shift *implicitly*: counting ``<begin_statements>`` AT index b
    arms the mask exactly at ``loss_weight[b]`` (which targets ``tokens[b+1]``,
    the first contact), and counting ``<end>`` AT index e disarms it at
    ``loss_weight[e]`` (which targets ``tokens[e+1]``) while leaving
    ``loss_weight[e-1]`` armed (predicting ``<end>``). No extra ``jnp.roll`` is
    needed. ``axis=-1`` keeps each packed sequence's counts independent whether
    levanter passes this a 1-D sequence or a 2-D (batch, seq) array.
    """
    opened = jnp.cumsum(tokens == BEGIN_STATEMENTS_ID, axis=-1)
    closed = jnp.cumsum(tokens == END_ID, axis=-1)
    return (opened > closed).astype(jnp.float32)


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
REFINEMENT_TRAIN_TOK = _tokenize_step(
    "exp163-refinement-train", REFINEMENT_TRAIN_GLOB, is_validation=False
)
VAL_FULL_TOK = _tokenize_step("contacts-v1-val-full", VAL_FULL_GLOB, is_validation=True)


def _as_array_exemplar_component(component: DatasetComponent) -> DatasetComponent:
    """Read a token cache with an array-backed ``input_ids`` exemplar + packing."""
    return _as_array_exemplar(dataclasses.replace(component, pack=True))


def _as_array_exemplar(component: DatasetComponent) -> DatasetComponent:
    return dataclasses.replace(
        component, format=ArrayExemplarTextLmDatasetFormat(text_key="document")
    )


def _component(
    step: ExecutorStep,
    *,
    loss_weight_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> DatasetComponent:
    """Array-exemplar + ``pack=True`` component, optionally answer-span masked.

    ``loss_weight_fn`` is attached only for the TRAIN component; val components
    leave it ``None`` (unmasked — every position contributes to the val loss).
    """
    component = _as_array_exemplar_component(
        step_to_lm_mixture_component(step, include_raw_paths=True)
    )
    if loss_weight_fn is not None:
        component = dataclasses.replace(component, loss_weight_fn=loss_weight_fn)
    return component


# The single held-out validation component monitored every eval (issue #163).
# The key becomes the W&B metric namespace: ``eval/contacts-v1-val-full/loss``.
# UNMASKED: step-0 ≈ 2.7566 warm-start sanity + base-task retention monitor.
VAL_COMPONENTS = {
    "contacts-v1-val-full": _component(VAL_FULL_TOK),
}


def build_train_step(
    *,
    name: str,
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
    """Build the exp163 refinement continue-train step (warm-start from E8/#75).

    The single train corpus is the refinement corpus, masked to its
    ``<begin_statements> … <end>`` answer span via
    :func:`answer_span_loss_weight`; the val corpus stays unmasked. Weights
    warm-start from ``INIT_CHECKPOINT`` (fresh step-0 / optimizer / LR schedule /
    data loader — a continue-train, not a resume), so ``learning_rate`` +
    ``warmup`` + ``lr_schedule`` define this run's schedule. Only
    ``learning_rate``, ``num_train_steps``, ``data_seed``, ``lr_schedule`` are
    ``versioned()`` so tuning the other knobs doesn't needlessly bust the cache.
    """
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
        # Eric's Qwen checkpoint padded its vocab to a multiple of 4 (2846 ->
        # 2848) for TPU efficiency, so its embedding matrix is wider than the
        # tokenizer. This pads the tokenizer up to the checkpoint's vocab so the
        # warm-started model shape matches; no-op if the checkpoint vocab already
        # equals the tokenizer's. VERIFY at launch via the step-0 sanity eval
        # (loss ~2.75); a vocab shape error there means flip this.
        pad_tokenizer_to_match_model=True,
    )

    train_key = "exp163-refinement-train"
    components = {
        train_key: _component(REFINEMENT_TRAIN_TOK, loss_weight_fn=answer_span_loss_weight),
        **VAL_COMPONENTS,
    }
    train_weights = {train_key: 1.0, **{k: 0.0 for k in VAL_COMPONENTS}}

    refinement_data = LmDataConfig(
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
        tokenized=refinement_data,
        model_config=MODEL_CONFIG,
        train_config=train_config,
        tags=["protein", "contacts-v1", "qwen3", "answer-masked", "exp163", "refinement", *extra_tags],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="exp163-rollout-refinement",
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
    "BEGIN_STATEMENTS_ID",
    "END_ID",
    "INIT_CHECKPOINT",
    "MODEL_CONFIG",
    "PROTEIN_RESOURCES_USE5",
    "REFINE_MARIN_PREFIX",
    "REFINEMENT_TRAIN_GLOB",
    "VAL_COMPONENTS",
    "answer_span_loss_weight",
    "build_hf_export_step",
    "build_train_step",
]
