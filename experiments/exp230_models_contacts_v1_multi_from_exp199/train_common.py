# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Training config for exp230's ``<contacts-v1.multi>`` fine-tune of exp199.

Adapted from #163's ``refine_ft_common.py``.  The model, optimiser and batch are
#163's arm-F recipe unchanged -- that is the point, since #163 is what
established that this recipe both produces many candidates and holds the base
task.  Four things are exp230's:

1. **Warm start is `contacts-v1-exp199-1.5B`, not E8.**  #163's architecture
   block was cross-checked against E8's ``config.json``; it matches exp199's
   byte for byte (hidden 2048 / inter 8192 / 32 heads / 8 KV / 24 layers /
   Llama3 rope at 500000 / ctx 8192 / vocab 2845), so nothing about the model
   definition changes.
2. **Storage is GCS**, not CoreWeave S3 -- see ``dispatch_rollouts.py`` for why
   this run is on marin at all.
3. **The tokenizer is the renamed one** (``timodonnell/contacts-v1-multi-tokenizer``,
   id 7 = ``<contacts-v1.multi>``).  Ids are identical to the published
   tokenizer, so this changes nothing about training -- but it is what levanter
   exports alongside the weights, and a checkpoint that ships the *published*
   tokenizer cannot be prompted into multi mode by anyone writing the literal
   token.
4. **~2,500 steps over 2 epochs, checkpointing every 250.**  #163's arm F ran
   **405** steps and leaked the multi-draft habit into plain mode (~2.94 sections
   under the plain sentinel).  #175 got a completely clean token-0 mode switch
   from the same kind of marker after **2,070** steps.  The intermediate
   checkpoints are what turn "more steps fixes the leak" from an assumption into
   a measurement, and let the earliest clean checkpoint be selected.

**No microbatching, deliberately.**  levanter re-normalises per-token loss
weights *per microbatch*; with drafts and finals carrying different weights that
silently changes the objective.  Gradient accumulation is not a free memory
lever here.

Warm-start verification uses **bpb, not loss**: per-token loss is not comparable
across harnesses (levanter's bytes-per-token bookkeeping is packing- and
version-dependent), and there is no step-0 eval on this path.
"""
from __future__ import annotations

import dataclasses
import os

from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.training.training import TrainLmOnPodConfig

from marinfold_models import build_train_lm_on_pod_config

GCS_PREFIX = os.environ.get(
    "EXP230_GCS_PREFIX",
    "gs://marin-us-central1/protein-structure/MarinFold/exp230_contacts_v1_multi",
)

#: Pre-tokenized, pre-packed sequences carrying the profile-F loss weights
#: (``input_ids`` + ``loss_weights``), written by ``tokenize_corpus.py``.
CORPUS_GLOB = os.environ.get("EXP230_CORPUS", f"{GCS_PREFIX}/tokenized/*.parquet")

#: Held-out validation = exp53's canonical contacts-v1 val split, raw text,
#: monitored UNMASKED. It is not a multi-draft set: it anchors the warm start and
#: tracks base-task retention, which is Gate A's early-warning signal.
VAL_GLOB = os.environ.get("EXP230_VAL", f"{GCS_PREFIX}/val/*.parquet")

#: exp199, bf16 on disk. bf16 rather than the fp32 export because the TPU
#: inference path requires bf16 weights anyway, and evaluating the BASE from the
#: same artifact is what makes Gate A's paired comparison apples-to-apples.
INIT_FROM_HF = os.environ.get("EXP230_INIT_HF", f"{GCS_PREFIX}/model/exp199_bf16")
INIT_FROM_LEVANTER = os.environ.get("EXP230_INIT_CKPT") or None

CACHE_BASE = f"{GCS_PREFIX}/cache"

#: Bare repo id. The ``repo@revision`` form is rejected by huggingface_hub's
#: validate_repo_id on the training tokenizer-load path (#85's note).
MULTI_TOKENIZER = os.environ.get("EXP230_TOKENIZER", "timodonnell/contacts-v1-multi-tokenizer")

# Qwen3 1.47B. Cross-checked against exp199's published config.json.
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

SEQ_LEN = 8192
TRAIN_BATCH = int(os.environ.get("EXP230_TRAIN_BATCH", "128"))
WEIGHT_DECAY = 0.2
WARMUP = 0.1
MIN_LR_RATIO = 0.1
LR_SCHEDULE = "cosine"
CONTACTS_V1_DATA_SEED = 0

PROTEIN_RESOURCES_TPU = ResourceConfig.with_tpu(
    os.environ.get("EXP230_TPU_TYPE", "v5p-16"),
    slice_count=int(os.environ.get("EXP230_TPU_SLICES", "1")),
    # REQUIRED: with_tpu leaves regions unset and the scheduler has picked
    # regions with no v5p groups at all. Pinned to where the data is.
    regions=tuple(os.environ.get("EXP230_TPU_REGIONS", "us-central1").split(",")),
)

TRAIN_COMPONENT_KEY = "multi-train"
VAL_COMPONENT_KEY = "contacts-v1-val"


def build_data_config(*, corpus_glob: str = CORPUS_GLOB, val_glob: str = VAL_GLOB,
                      cache_base: str = CACHE_BASE) -> LmDataConfig:
    """Train on pre-packed weighted rows; validate unmasked on raw contacts-v1.

    The two components use different formats on purpose.  Train is
    ``PrebuiltLmDatasetFormat``: the rows already hold packed ``input_ids`` plus
    per-token ``loss_weights``, which is the only route to a weighted loss in
    current levanter, and it is why ``pack`` is off -- the rows are already
    ``SEQ_LEN`` and ``PrebuiltLmDataset`` has no packing mode.  Cross-document
    attention is still blocked: segment ids come from the ``<eos>`` the packer
    wrote after each document.
    """
    train_format = PrebuiltLmDatasetFormat(input_ids_key="input_ids",
                                           loss_weights_key="loss_weights")
    train_source = UrlDatasetSourceConfig(
        train_urls=[corpus_glob], validation_urls=[],
        cache_dir=f"{cache_base}/{TRAIN_COMPONENT_KEY}", format=train_format,
    )
    val_format = TextLmDatasetFormat(text_key="document")
    val_source = UrlDatasetSourceConfig(
        train_urls=[], validation_urls=[val_glob],
        cache_dir=f"{cache_base}/{VAL_COMPONENT_KEY}", format=val_format,
    )
    return LmDataConfig(
        tokenizer=MULTI_TOKENIZER,
        cache_dir=None,
        auto_build_caches=True,
        shuffle=True,
        block_cross_document_attention=True,
        components={
            TRAIN_COMPONENT_KEY: DatasetComponent(
                source=train_source, cache_dir=train_source.cache_dir,
                format=train_format, pack=False, split="train"),
            VAL_COMPONENT_KEY: DatasetComponent(
                source=val_source, cache_dir=val_source.cache_dir,
                format=val_format, pack=True, split="validation"),
        },
        train_weights={TRAIN_COMPONENT_KEY: 1.0, VAL_COMPONENT_KEY: 0.0},
    )


def build_on_pod_config(
    *,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    corpus_glob: str = CORPUS_GLOB,
    val_glob: str = VAL_GLOB,
    init_from_hf: str | None = INIT_FROM_HF,
    init_from_levanter: str | None = INIT_FROM_LEVANTER,
    resources: ResourceConfig = PROTEIN_RESOURCES_TPU,
    env_vars: dict[str, str] | None = None,
    steps_per_eval: int = 250,
    steps_per_checkpoint: int | None = 250,
    hf_save_steps: int | None = 250,
    tags: tuple[str, ...] = (),
    wandb_group: str = "exp230-contacts-v1-multi",
) -> TrainLmOnPodConfig:
    """The concrete ``TrainLmOnPodConfig`` for one LR.

    Warm-starts weights only -- fresh optimiser, fresh schedule, step 0 -- so
    ``learning_rate`` + ``WARMUP`` + ``LR_SCHEDULE`` define this run's schedule.

    ``hf_save_steps`` is armed by default here, unlike #163: the leak-vs-steps
    curve is a *result* of this experiment, and measuring it needs a
    transformers-readable export at each checkpoint, not only at the end.  Each
    export is a full copy of the weights (~2.9 GB), which is the price of that
    measurement.
    """
    if init_from_levanter:
        init_hf: str | bool = False
        init_ckpt: str | None = init_from_levanter
    else:
        init_hf = init_from_hf or False
        init_ckpt = None

    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule=LR_SCHEDULE,
    )

    cfg = build_train_lm_on_pod_config(
        run_name=run_name,
        model=MODEL_CONFIG,
        optimizer=optimizer,
        data=build_data_config(corpus_glob=corpus_glob, val_glob=val_glob),
        resources=resources,
        output_path=output_path,
        num_train_steps=num_train_steps,
        train_batch_size=TRAIN_BATCH,
        seq_len=SEQ_LEN,
        # Overrides LmDataConfig.auto_build_caches, so it must be set here too.
        auto_build_caches=True,
        steps_per_eval=steps_per_eval,
        max_eval_batches=None,
        steps_per_checkpoint=steps_per_checkpoint,
        data_seed=CONTACTS_V1_DATA_SEED,
        initialize_from_checkpoint_path=init_ckpt,
        initialize_from_hf=init_hf,
        pad_tokenizer_to_match_model=True,
        wandb_project="MarinFold",
        wandb_group=wandb_group,
        wandb_name=run_name,
        tags=tags,
        env_vars=env_vars,
    )
    if hf_save_steps:
        cfg = dataclasses.replace(
            cfg, train_config=dataclasses.replace(cfg.train_config,
                                                  hf_save_steps=hf_save_steps))
    return cfg


__all__ = [
    "CACHE_BASE", "CONTACTS_V1_DATA_SEED", "CORPUS_GLOB", "GCS_PREFIX",
    "INIT_FROM_HF", "INIT_FROM_LEVANTER", "LR_SCHEDULE", "MIN_LR_RATIO",
    "MODEL_CONFIG", "MULTI_TOKENIZER", "PROTEIN_RESOURCES_TPU", "SEQ_LEN",
    "TRAIN_BATCH", "VAL_GLOB", "WARMUP", "WEIGHT_DECAY",
    "build_data_config", "build_on_pod_config",
]
