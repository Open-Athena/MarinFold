# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the exp163 rollout-refinement fine-tune (issue #163).

We **continue-train** eric-czech's tuned contacts-v1 1.5B (the #75 sweep winner,
``prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084``, eval loss 2.7566 — "E8") to
turn it into a *refiner*: given a protein's sequence plus K noisy candidate
rollouts, emit the TRUE contact set.

A refinement document (built by ``build_refinement_corpus.py``) is a normal
contacts-v1 document with K candidate-rollout blocks prepended as context::

    <contacts-v1> <begin_sequence> ...sequence...
      <CAND> <contact> pi pj ...          candidate block 1  (subsampled/flipped)
      ...                                 0..K such blocks, unordered
    <begin_statements> <contact> ...TRUE contacts... <end>

where the candidate marker ``<CAND>`` reuses the spare token
``<contacts-and-distances-v1>`` (the *other* format's doc-type sentinel — never
emitted inside a contacts-v1 document, so it is collision-proof and distinct from
every begin/end marker). Zero vocab change; see ``build_refinement_corpus.py``.

The LM loss is armed on the ``<begin_statements> … <end>`` answer span only —
never on the sequence header or the noisy candidate blocks. See ``loss_mask.py``.

--------------------------------------------------------------------------------
RE-BASE NOTE (2026-07-27). This module used to be a copy of exp120's
``contacts_v1_ft_common.py``, i.e. **executor-era marin**: ``default_train`` /
``default_tokenize`` / ``ExecutorStep`` / ``versioned`` / ``this_output_path``
plus ``DatasetComponent.loss_weight_fn``. None of that survives in the marin the
exp163 environment resolves (0.2.57 / levanter 1.2), which is the environment we
need because it is the one whose ``fray.types.JobRequest`` carries ``priority``
(batch-band dispatch). Three things changed:

1. ``default_train`` -> :func:`marinfold_models.build_train_lm_on_pod_config` —
   the StepContext-free builder exp108 introduced, which returns a concrete
   ``TrainLmOnPodConfig`` the caller dispatches itself.
2. ``DatasetComponent.loss_weight_fn`` is **gone**. Per-token loss weights now
   come out of the cache via ``PrebuiltLmDatasetFormat(input_ids_key=…,
   loss_weights_key=…)``, so the answer-span mask is materialized offline by
   ``tokenize_refinement_corpus.py``. This also removes the old plan's second
   blocker: nothing cloudpickles by module reference any more, so the GPU worker
   never has to import exp163 code.
3. Warm-start is ``initialize_from_hf`` against the **HF export already staged on
   S3** (``exp163/model/step-35679`` — the same artifact ``dispatch_rollouts.py``
   feeds to vLLM). levanter's HF converter reads fsspec URLs, so an ``s3://``
   directory works directly, and nothing has to be staged. Weights only: fresh
   optimizer / LR schedule / step 0 — a continue-train, not a resume.

   The Levanter-native E8 checkpoint DOES exist, but not where exp120 pointed —
   the path is missing a ``protein/`` segment. It is at::

       gs://marin-us-east5/checkpoints/protein/
           prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679

   (with ``hf/`` and ``hf_bf16/`` exports as siblings — ``hf/step-35679`` is what
   was copied to S3). We do NOT use it: cw-rno2a pods carry no GCS credentials, so
   it would first have to be staged GCS -> S3 over the ~2.5 MB/s workstation
   uplink, to warm-start from *the same weights* the HF export already holds. It
   stays reachable via ``EXP163_INIT_CKPT`` if a full-state restore is ever wanted.
--------------------------------------------------------------------------------

Storage is **all CoreWeave S3** (``s3://marin-us-east-02a/MarinFold/exp163``):
iris injects ONE S3 endpoint/credential set into each pod and the pods have no
GCS, so the warm-start weights, both corpora, the token caches and the
checkpoints all live under that one prefix.

The held-out validation component is exp53's canonical original contacts-v1 val
split, monitored **unmasked** every eval. It is NOT a refinement set: it anchors
the step-0 warm-start sanity (loss ≈ 2.7566, confirming E8 loaded) and tracks
base-task retention (catastrophic-forgetting monitor) while the model learns to
refine.
"""

from __future__ import annotations

import os

from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.training.training import TrainLmOnPodConfig

from marinfold_models import build_train_lm_on_pod_config

# ---------------------------------------------------------------------------
# Storage — one exp163 prefix on CoreWeave S3 (pods have no GCS; #163).
# ---------------------------------------------------------------------------
EXP163_S3_PREFIX = os.environ.get("EXP163_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp163")

# Train corpus: PRE-TOKENIZED, pre-packed refinement sequences carrying the
# answer-span mask (`input_ids` + `loss_weights`), written by
# `tokenize_refinement_corpus.py` from the raw `val10k/refinement_corpus/`.
REFINEMENT_TOKENIZED_GLOB = os.environ.get(
    "EXP163_CORPUS", f"{EXP163_S3_PREFIX}/val10k/refinement_tokenized/*.parquet"
)
# Unmasked val = the original contacts-v1 val split (raw text parquet, one
# `document` column), tokenized on the fly. exp108 already staged this tree.
VAL_GLOB = os.environ.get(
    "EXP163_VAL",
    "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/val/*.parquet",
)
# E8 warm-start — the fp32 HF export (`config.json` + sharded safetensors).
# levanter's HFCheckpointConverter reads fsspec URLs, so an s3:// dir works.
INIT_FROM_HF = os.environ.get("EXP163_INIT_HF", f"{EXP163_S3_PREFIX}/model/step-35679")
# Optional escape hatch: a Levanter-NATIVE checkpoint dir, if one is ever staged.
# When set it wins over INIT_FROM_HF (they are mutually exclusive in levanter).
INIT_FROM_LEVANTER = os.environ.get("EXP163_INIT_CKPT") or None

# Token caches (built on the training workers on first read; auto_build_caches).
CACHE_BASE = f"{EXP163_S3_PREFIX}/tokenized"

# contacts-v1 tokenizer. BARE repo id everywhere — the `repo@revision` form is
# rejected by huggingface_hub's validate_repo_id on the training tokenizer-load
# path (exp85's note). The pinned revision is documentation only.
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"  # documented pin; not passed as repo@rev
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO

# ---------------------------------------------------------------------------
# Model — exp75's Qwen3 1.47B (exp44 dims + Llama3 rope). MUST match the
# checkpoint we warm-start from; cross-checked against the S3 export's
# config.json (hidden 2048 / inter 8192 / 32 heads / 8 kv / 24 layers / llama3
# rope / vocab 2845).
# ---------------------------------------------------------------------------
MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# ---------------------------------------------------------------------------
# Recipe — a dedicated 1-epoch cosine continue-train at low LR, batch 128
# (== Eric's #75 training batch, so there is no batch-scaling confound), seq 8192.
# ---------------------------------------------------------------------------
SEQ_LEN = 8192
TRAIN_BATCH = int(os.environ.get("EXP163_TRAIN_BATCH", "128"))
WEIGHT_DECAY = 0.2
WARMUP = 0.1
MIN_LR_RATIO = 0.1
LR_SCHEDULE = "cosine"
# Fixed seed for the training-data Feistel shuffle.
CONTACTS_V1_DATA_SEED = 0

# ---------------------------------------------------------------------------
# Device — a single 8xH100 node on cw-rno2a, FSDP over the 8 GPUs (exp108's
# PROTEIN_RESOURCES_H100). The 1.5B refiner shards comfortably on one node;
# scale out via EXP163_REPLICAS once a single-node smoke run is green.
# ---------------------------------------------------------------------------
PROTEIN_RESOURCES_H100 = ResourceConfig.with_gpu(
    "H100",
    count=8,          # GPUs per node (gd-8xh100ib-i128 = 8xH100)
    cpu=32,
    ram="256g",
    disk="256g",
    replicas=1,       # nodes; 1 node = 8 H100.
)

TRAIN_COMPONENT_KEY = "refinement-train"
VAL_COMPONENT_KEY = "contacts-v1-val"


def build_data_config(
    *,
    corpus_glob: str = REFINEMENT_TOKENIZED_GLOB,
    val_glob: str = VAL_GLOB,
    cache_base: str = CACHE_BASE,
) -> LmDataConfig:
    """Concrete-path LM data config: train answer-span MASKED, val UNMASKED.

    The two components deliberately use DIFFERENT formats:

    * **train** — ``PrebuiltLmDatasetFormat``: the shards already hold packed
      ``input_ids`` plus the per-token ``loss_weights`` produced by
      ``tokenize_refinement_corpus.py``, so the LM loss is armed only on each
      document's ``<begin_statements> … <end>`` answer span. This is the only
      route to a masked loss in current levanter (``DatasetComponent`` no longer
      takes a ``loss_weight_fn``), and it is also why ``pack`` is off here: the
      rows are ALREADY packed to ``SEQ_LEN``, and ``PrebuiltLmDataset`` has no
      packing mode. Cross-document attention is still blocked — the dataset
      derives segment ids from the ``<eos>`` the packer wrote after each document.
    * **val** — ``TextLmDatasetFormat`` over the raw ``document`` column of the
      original contacts-v1 val split, packed and tokenized on the fly, with NO
      loss weights (every position counts). Step-0 should read ≈ 2.7566.

    Caches land at ``<cache_dir>/train`` and ``<cache_dir>/validation``.
    """
    train_format = PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")
    train_source = UrlDatasetSourceConfig(
        train_urls=[corpus_glob],
        validation_urls=[],
        cache_dir=f"{cache_base}/refinement-train",
        format=train_format,
    )
    val_format = TextLmDatasetFormat(text_key="document")
    val_source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[val_glob],
        cache_dir=f"{cache_base}/contacts-v1-val",
        format=val_format,
    )
    train_component = DatasetComponent(
        source=train_source,
        cache_dir=train_source.cache_dir,
        format=train_format,
        pack=False,      # rows are pre-packed to SEQ_LEN; see the docstring
        split="train",
    )
    val_component = DatasetComponent(
        source=val_source,
        cache_dir=val_source.cache_dir,
        format=val_format,
        pack=True,       # never concat-and-split: partial protein docs are nonsense
        split="validation",
    )
    return LmDataConfig(
        tokenizer=CONTACTS_V1_TOKENIZER,  # bare id: timodonnell/contacts-v1-tokenizer
        cache_dir=None,                   # each component sets its own concrete cache_dir
        auto_build_caches=True,           # workers build the caches on first read
        shuffle=True,                     # full Feistel permutation (data_seed below)
        block_cross_document_attention=True,
        components={TRAIN_COMPONENT_KEY: train_component, VAL_COMPONENT_KEY: val_component},
        train_weights={TRAIN_COMPONENT_KEY: 1.0, VAL_COMPONENT_KEY: 0.0},  # val weight 0
    )


def build_on_pod_config(
    *,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    corpus_glob: str = REFINEMENT_TOKENIZED_GLOB,
    val_glob: str = VAL_GLOB,
    init_from_hf: str | None = INIT_FROM_HF,
    init_from_levanter: str | None = INIT_FROM_LEVANTER,
    resources: ResourceConfig = PROTEIN_RESOURCES_H100,
    env_vars: dict[str, str] | None = None,
    steps_per_eval: int = 100,
    steps_per_checkpoint: int | None = None,
    tags: tuple[str, ...] = (),
    wandb_group: str = "exp163-rollout-refinement",
) -> TrainLmOnPodConfig:
    """Build the concrete ``TrainLmOnPodConfig`` for one LR of the refiner run.

    Warm-starts weights from E8 with a fresh optimizer / LR schedule / step 0, so
    ``learning_rate`` + :data:`WARMUP` + :data:`LR_SCHEDULE` define this run's
    schedule. ``init_from_levanter`` (a Levanter-native checkpoint dir) wins over
    ``init_from_hf`` when both are given — levanter rejects specifying both.

    ``max_eval_batches=None`` evaluates the FULL val split every time (issue #163):
    it is the warm-start anchor and the base-task-retention monitor, so a noisy
    subsample would defeat the point.
    """
    if init_from_levanter:
        init_hf: str | bool = False
        init_ckpt: str | None = init_from_levanter
    else:
        init_hf = init_from_hf or False
        init_ckpt = None

    # AdamW, cosine decay to the min_lr_ratio floor, `warmup` as a fraction of
    # steps. Tracks Eric's #75 tuned recipe with the peak LR dialled down for a
    # continue-train.
    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        min_lr_ratio=MIN_LR_RATIO,
        lr_schedule=LR_SCHEDULE,
    )

    return build_train_lm_on_pod_config(
        run_name=run_name,
        model=MODEL_CONFIG,
        optimizer=optimizer,
        data=build_data_config(corpus_glob=corpus_glob, val_glob=val_glob),
        resources=resources,
        output_path=output_path,  # replicate_path + checkpointer/HF/run-id derive from this
        num_train_steps=num_train_steps,
        train_batch_size=TRAIN_BATCH,
        seq_len=SEQ_LEN,
        # Both corpora are cached on the training workers on first read; this
        # OVERRIDES `LmDataConfig.auto_build_caches`, so it must be set here too.
        auto_build_caches=True,
        steps_per_eval=steps_per_eval,
        max_eval_batches=None,
        steps_per_checkpoint=steps_per_checkpoint,
        data_seed=CONTACTS_V1_DATA_SEED,
        initialize_from_checkpoint_path=init_ckpt,
        initialize_from_hf=init_hf,
        # The E8 HF export carries the full 2845-token vocab, so this is a no-op
        # today; kept armed because Eric's TPU-padded Levanter checkpoints (2846
        # -> 2848) would need it if `init_from_levanter` is ever used.
        pad_tokenizer_to_match_model=True,
        wandb_project="MarinFold",
        wandb_group=wandb_group,
        wandb_name=run_name,
        tags=tags,
        env_vars=env_vars,
    )


__all__ = [
    "CACHE_BASE",
    "CONTACTS_V1_DATA_SEED",
    "CONTACTS_V1_TOKENIZER",
    "EXP163_S3_PREFIX",
    "INIT_FROM_HF",
    "INIT_FROM_LEVANTER",
    "LR_SCHEDULE",
    "MIN_LR_RATIO",
    "MODEL_CONFIG",
    "PROTEIN_RESOURCES_H100",
    "REFINEMENT_TOKENIZED_GLOB",
    "SEQ_LEN",
    "TRAIN_BATCH",
    "VAL_GLOB",
    "WARMUP",
    "WEIGHT_DECAY",
    "build_data_config",
    "build_on_pod_config",
]
