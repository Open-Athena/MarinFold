# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the contacts-v1 1.5B **warm-restart** run (issue #85).

Copied from ``exp67_models_contacts_v1_1_5b/contacts_v1_train_common.py`` (the
quick #67 run) and extended with an ``initialize_from_checkpoint_path`` knob so
``build_train_step`` can warm-start weights from the #67 ``step-11999``
checkpoint and re-heat the LR for another epoch (see ``build_train_step``'s
docstring). Everything else — corpus, tokenizer, token caches (reused via the
shared ``MARIN_PREFIX``), TPU recipe — is identical to #67.

This mirrors ``exp0_models_protein_docs_initial_port/protein_train_common.py``
but targets the new, smaller ``contacts-v1`` corpus and uses an **unmasked**
next-token loss (no distance-bin loss mask — contacts-v1 has no ``<distance>``
statements). Every training script in this directory shares:

* The contacts-v1 tokenizer (2845 tokens), pinned to a commit.
* The ``contacts_v1`` train/val parquets — read from their region-local GCS
  working copy (the HF *bucket* publish isn't levanter-addressable; see
  ``GCS_CORPUS_BASE`` below) — and the marin tokenize steps that materialize
  their token caches (sharing the steps here means all runs reuse one cache).
* The TPU resource config pinned to ``us-east5-a`` (co-located with the
  ``marin-us-east5`` checkpoint bucket).
* **Shuffling** of the training data: the corpus shards are physically
  round-descending (highest-pLDDT last), so an unshuffled stream would be
  badly biased. ``LmDataConfig.shuffle=True`` enables a full Feistel
  permutation (not the windowed block-shuffle default) over the whole train
  set, with a fixed ``data_seed``.

* **Full validation each eval.** ``LmDataConfig.shuffle`` applies *only* to the
  train stream — levanter reads the validation set sequentially. The published
  ``val`` shards are round-segregated (shard 0 is all round-4 / lowest-pLDDT),
  so a ``max_eval_batches`` head would be biased to the lowest-pLDDT structures,
  NOT a shuffled sample. The val split is small (~42K docs ≈ ~45 eval batches),
  so we eval the *entire* held-out val split each time (``max_eval_batches=None``)
  — unbiased and only ~6% compute overhead. (The issue's "downsample to ~5000"
  was premised on a large val set; it isn't one. See README "Implementation
  notes".)

Per-experiment scripts choose the model config, learning rate, output path,
and schedule on top of these shared bits.
"""

import dataclasses
import os
from collections.abc import Sequence

from fray import ResourceConfig
from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution import ExecutorStep, output_path_of, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

from marinfold_models.defaults import default_tokenize, default_train
from marinfold_models.simple_train_config import SimpleTrainConfig

# Pin ALL marin-executor output — token caches AND checkpoints AND HF exports —
# under the MarinFold protein-structure prefix, per AGENTS.md. The executor
# takes its output prefix from the MARIN_PREFIX env var; left unset it defaults
# to the top level of the bucket, scattering artifacts into
# gs://marin-us-east5/{tokenized,checkpoints}/... which belong to the marin
# protein-experiments convention, NOT ours. We force-set it here (at import,
# before any default_tokenize / default_train constructs an output path) so
# every entry point in this dir — train and export — lands under one prefix.
#
# exp85 deliberately KEEPS exp67's prefix (not an exp85-specific one): the
# tokenize-step output path is derived from MARIN_PREFIX, so reusing it makes
# marin resolve exp67's already-materialized 4.7B-token train cache + val cache
# instead of re-tokenizing. That also sidesteps the known marin-latest fresh
# tokenize→train cache-ledger bug (#6008/#6014). The warm-restart run's
# checkpoints/HF exports are namespaced under their own W&B-run-name subdir, so
# they don't collide with exp67's artifacts despite the shared prefix.
CONTACTS_V1_MARIN_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp67_contacts_v1_1_5b"
os.environ["MARIN_PREFIX"] = CONTACTS_V1_MARIN_PREFIX

# contacts-v1 tokenizer (2845 vocab tokens; verified loadable by
# transformers/levanter). The training and export paths use the immutable
# ``repo@revision`` identifier. Marin's tokenize step currently passes the value
# directly to huggingface_hub, which rejects that syntax, so tokenization uses
# the bare repo id and includes the revision in its executor-step name. The
# already-materialized caches below were created from this exact revision.
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"
CONTACTS_V1_TOKENIZER = f"{CONTACTS_V1_TOKENIZER_REPO}@{CONTACTS_V1_TOKENIZER_REVISION}"

# contacts-v1 corpus tokenize INPUT. We read the parquet directly from its
# region-local GCS working copy (written by exp53), NOT the published HF bucket
# at hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1/.
# Reason: HF *buckets* are NOT HfFileSystem-addressable, so levanter's fsspec on
# the iris worker resolves `open-athena/MarinFold` as a dataset/model repo and
# 404s ("repository not found") — the tokenize step fails before reading a byte.
# The GCS copy is byte-identical, co-located with the TPU (us-east5), and plain
# gcsfs-globbable. Splits train / val / test; one text column ``document``.
# The ``*.parquet`` glob is explicit (not a bare dir) so neither marin's
# expand_tokenize_paths nor levanter's URL globber falls back to the default
# ``**/*.json.gz`` pattern (which matches nothing here).
GCS_CORPUS_BASE = "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents"

# Fixed seed for the training-data shuffle (and the marin executor versions it,
# so changing it forces a fresh run rather than silently reusing a cache).
CONTACTS_V1_DATA_SEED = 0

# Pin to us-east5-a so the TPU is co-located with the `marin-us-east5`
# checkpoint bucket. The v5p-preemptible pool spans {us-central1-a, us-east5-a};
# without this pin a worker can land in us-central1 and pay cross-region I/O
# latency on every checkpoint write. (Verbatim from exp0.)
PROTEIN_RESOURCES_USE5 = ResourceConfig.with_tpu(
    "v5p-8",
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
    zone="us-east5-a",
)

# Reuse the caches created for the recorded run. Keep the original step names so
# Marin resolves the existing 4.7B-token train cache and held-out validation
# cache rather than tokenizing again.
contacts_v1_tokenized = default_tokenize(
    name="contacts-v1",
    dataset=f"{GCS_CORPUS_BASE}/train/*.parquet",
    tokenizer=CONTACTS_V1_TOKENIZER_REPO,
    format=TextLmDatasetFormat(text_key="document"),
)
contacts_v1_val_tokenized = default_tokenize(
    name="contacts-v1-val",
    dataset=f"{GCS_CORPUS_BASE}/val/*.parquet",
    tokenizer=CONTACTS_V1_TOKENIZER_REPO,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)


def unmasked_components() -> dict[str, object]:
    """Train+val DatasetComponents with ``pack=True`` and NO loss mask.

    Suitable for splatting into ``LmDataConfig.components``. ``pack=True`` avoids
    concat-and-split, which would create partial documents — protein-contact
    docs are nonsensical without their header. Unlike exp0's distance-masked
    components, no ``loss_weight_fn`` is applied: every token position
    contributes to the loss.
    """
    train_component = dataclasses.replace(
        step_to_lm_mixture_component(contacts_v1_tokenized, include_raw_paths=True),
        pack=True,
    )
    val_component = dataclasses.replace(
        step_to_lm_mixture_component(contacts_v1_val_tokenized, include_raw_paths=True),
        pack=True,
    )
    return {"contacts-v1": train_component, "contacts-v1-val": val_component}


def build_train_step(
    *,
    name: str,
    model_config: LlamaConfig,
    learning_rate: float,
    extra_tags: Sequence[str] = (),
    num_train_steps: int = 12_000,
    train_batch_size: int = 128,
    train_seq_len: int = 8192,
    weight_decay: float = 0.01,
    warmup: float = 0.1,
    steps_per_eval: int = 250,
    steps_per_export: int = 2000,
    max_eval_batches: int | None = None,
    data_seed: int = CONTACTS_V1_DATA_SEED,
    wandb_name: str | None = None,
    resources: ResourceConfig = PROTEIN_RESOURCES_USE5,
    override_output_path: str | None = None,
    initialize_from_checkpoint_path: str | None = None,
) -> ExecutorStep:
    """Build an unmasked contacts-v1 training step with the quick #67 recipe.

    Mirrors the 1.5B run from ``marin/protein-training-1b``: TPU pinned to
    ``us-east5-a``, no in-training distogram benchmark, no Paloma validation, no
    eval-harness tasks. The train stream is **shuffled** (``LmDataConfig.shuffle
    =True``, full Feistel permutation) with a fixed ``data_seed`` because the
    corpus shards are round-descending. Validation evaluates the full held-out
    val split (``max_eval_batches=None``) — see the module docstring for why we
    do NOT use a ``max_eval_batches`` head (round-segregated val → biased sample).

    Only ``learning_rate``, ``num_train_steps`` and ``data_seed`` are wrapped in
    ``versioned()``, so the other knobs can be tuned without busting the cache —
    bump them via ``versioned()`` in the caller if a fresh run is needed.

    Args:
        name: Used to derive the output path and as the run name.
        model_config: ``LlamaConfig`` for the model architecture.
        learning_rate: Peak learning rate (wrapped in ``versioned()``).
        extra_tags: Additional W&B tags merged with the standard set.
        max_eval_batches: Cap on validation batches; ``None`` (default) evals
            the full val split.
        data_seed: Seed for the train shuffle (wrapped in ``versioned()``).
        wandb_name: Explicit W&B run name (defaults to ``name`` when None).
        override_output_path: When set, pins the output directory.
        initialize_from_checkpoint_path: When set, warm-starts model weights from
            this levanter checkpoint (e.g. a ``…/checkpoints/step-N`` dir) and
            begins a FRESH training run — step 0, fresh optimizer state, fresh LR
            schedule (so ``learning_rate``/``warmup`` define a re-heat), and a
            fresh shuffled data loader. This is a warm restart, NOT a resume:
            ``defaults.default_train`` routes it to
            ``TrainLmConfig.initialize_from_checkpoint_path`` because
            ``reset_data_loader_on_init`` is left True. Used by the exp85
            continuation of the #67 run.
    """
    # The training pod logs to W&B (entity open-athena). marin's training entry
    # raises unless WANDB_API_KEY is present in the pod's env_vars or os.environ
    # — and the pod does NOT inherit the launching shell's env, so we forward the
    # key explicitly here. It's read from the driver's environment (set it at
    # launch with `iris ... -e WANDB_API_KEY <key>`), never hard-coded, so no
    # secret lands in git. If unset, we omit it (and the run will fail fast with
    # marin's clear "WANDB_API_KEY must be set" message rather than silently
    # logging nowhere).
    env_vars = {"WANDB_ENTITY": "open-athena"}
    _wandb_key = os.environ.get("WANDB_API_KEY")
    if _wandb_key:
        env_vars["WANDB_API_KEY"] = _wandb_key

    train_config = SimpleTrainConfig(
        resources=resources,
        train_batch_size=train_batch_size,
        num_train_steps=versioned(num_train_steps),
        learning_rate=versioned(learning_rate),
        weight_decay=weight_decay,
        warmup=warmup,
        train_seq_len=train_seq_len,
        steps_per_eval=steps_per_eval,
        steps_per_export=steps_per_export,
        max_eval_batches=max_eval_batches,
        data_seed=versioned(data_seed),
        env_vars=env_vars,
        initialize_from_checkpoint_path=initialize_from_checkpoint_path,
    )

    contacts_v1_data = LmDataConfig(
        components=unmasked_components(),
        train_weights={"contacts-v1": 1.0, "contacts-v1-val": 0.0},
        tokenizer=CONTACTS_V1_TOKENIZER,
        cache_dir=None,
        shuffle=True,
        block_cross_document_attention=True,
    )

    return default_train(
        name=name,
        tokenized=contacts_v1_data,
        model_config=model_config,
        train_config=train_config,
        tags=["protein", "contacts-v1", "llama", "unmasked", *extra_tags],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group="protein-training",
        wandb_name=wandb_name or name,
        override_output_path=override_output_path,
    )


def build_hf_export_step(
    *,
    train_step: ExecutorStep,
    model_config: LlamaConfig,
    checkpoint_step: int,
    name_prefix: str,
    checkpoint_path_override: str | None = None,
) -> ExecutorStep:
    """Build a CPU-only HF export step for the contacts-v1 training run.

    The export reads the requested ``step-{checkpoint_step}`` checkpoint inside
    ``train_step``'s ``checkpoints/`` subdirectory. The contacts-v1 tokenizer is
    co-located with the exported weights (hard rule: tokenizer travels with the
    model).

    Set ``checkpoint_path_override`` to a literal gs:// path to snapshot a
    checkpoint while training is still in progress (bypasses the dependency on
    ``train_step`` reaching SUCCEEDED).
    """
    from copy import deepcopy

    from levanter.trainer import TrainerConfig
    from marin.export import convert_checkpoint_to_hf_step

    trainer = train_step.config.train_config.trainer
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"Expected TrainerConfig on train_step, got {type(trainer)!r}")

    checkpoint_path: str | object
    if checkpoint_path_override is not None:
        checkpoint_path = checkpoint_path_override
    else:
        checkpoint_path = output_path_of(train_step, f"checkpoints/step-{checkpoint_step}")

    return convert_checkpoint_to_hf_step(
        name=f"hf/{name_prefix}-step-{checkpoint_step}",
        checkpoint_path=checkpoint_path,  # pyrefly: ignore
        trainer=deepcopy(trainer),
        model=model_config,
        tokenizer=CONTACTS_V1_TOKENIZER,
        use_cpu=True,
        discover_latest=False,
    )


__all__ = [
    "CONTACTS_V1_TOKENIZER",
    "CONTACTS_V1_TOKENIZER_REPO",
    "CONTACTS_V1_TOKENIZER_REVISION",
    "GCS_CORPUS_BASE",
    "PROTEIN_RESOURCES_USE5",
    "build_hf_export_step",
    "build_train_step",
    "contacts_v1_tokenized",
    "contacts_v1_val_tokenized",
    "unmasked_components",
]
