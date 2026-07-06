# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants + HF-export helper for the exp108 3B Qwen3 sweep (issue #108).

First MarinFold training on **GPU** (CoreWeave RNO2A H100 cluster, ``cw-rno2a``)
and first to read its corpus / write its checkpoints on **S3** (CoreWeave AI
Object Storage) rather than GCS.

The training itself is submitted by ``dispatch_train.py`` via a **direct
batch-priority Fray dispatch** (NOT the marin executor's ``default_train`` /
``executor_main``), because #108 requires iris ``batch`` priority for all work
and the executor submits child jobs without a priority band. This module holds
only the pieces shared across the sweep + export: storage/tokenizer/device
constants and the HF-export helper. The recipe (which tracks Eric's #75 tuning
sweep — Qwen3, AdamW cosine, 10% warmup, wd 0.2) is realized in
``dispatch_train.build_on_pod_config`` and the entry point
``train_qwen_3b_contacts_v1_sweep.py``.
"""

import os

from fray import ResourceConfig

# ---------------------------------------------------------------------------
# Storage: everything under one `MarinFold/` prefix on the CoreWeave bucket, so
# the objects can be bulk-removed later (#108). Task pods carry ONE S3 endpoint/
# credential set (injected by `iris cluster start` from CW_KEY_ID/SECRET), so all
# inputs the job touches must live under this one bucket — hence the HF→S3
# staging step (`stage_data_to_coreweave.py`).
# ---------------------------------------------------------------------------
CONTACTS_V1_S3_PREFIX = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1"
# Set MARIN_PREFIX for any marin internals that read it (the direct-dispatch path
# sets concrete output/cache paths explicitly, but keep this consistent).
os.environ.setdefault("MARIN_PREFIX", CONTACTS_V1_S3_PREFIX)

# contacts-v1 tokenizer (2845 vocab tokens). BARE repo id everywhere — the
# `repo@revision` form is rejected by huggingface_hub's validate_repo_id on the
# training tokenizer-load path (see exp85's note). Loaded from HF directly
# (public, small); not staged to S3 (only the corpus parquet is).
CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"  # documented pin; not passed as repo@rev
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO

# contacts-v1 corpus — parquet shards staged from the (byte-identical) HF publish
# / GCS mirror to the CoreWeave bucket by `stage_data_to_coreweave.py`. Splits
# train / val; one text column `document`.
CONTACTS_V1_S3_CORPUS_BASE = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"

# Fixed seed for the training-data Feistel shuffle (corpus shards are round-
# descending / pLDDT-biased, so an unshuffled stream would be badly biased).
CONTACTS_V1_DATA_SEED = 0

# ---------------------------------------------------------------------------
# Device: a single 8×H100 node on cw-rno2a, FSDP over the 8 GPUs.
# ---------------------------------------------------------------------------
# A ~2.9B dense model shards comfortably across one node's 8×80GB with FSDP, so
# the default is a SINGLE node (replicas=1) — matches the validated marin
# train_tiny_model `with_gpu("H100", count=8)` pattern and avoids the (untested
# for this path) multi-host GPU gang. Scale out via the entry point's
# EXP108_REPLICAS knob after a single-node smoke run is green.
PROTEIN_RESOURCES_H100 = ResourceConfig.with_gpu(
    "H100",
    count=8,          # GPUs per node (gd-8xh100ib-i128 = 8×H100)
    cpu=32,
    ram="256g",
    disk="256g",
    replicas=1,       # nodes; 1 node = 8 H100. Override to scale out.
)


# HF export lives in `export_qwen_3b_contacts_v1.py`. It is a WIP against modern
# marin (0.2.38 moved/renamed the old `marin.export.convert_checkpoint_to_hf_step`
# + `executor_main`); not needed until a run produces a checkpoint to export.

__all__ = [
    "CONTACTS_V1_DATA_SEED",
    "CONTACTS_V1_S3_CORPUS_BASE",
    "CONTACTS_V1_S3_PREFIX",
    "CONTACTS_V1_TOKENIZER",
    "CONTACTS_V1_TOKENIZER_REPO",
    "CONTACTS_V1_TOKENIZER_REVISION",
    "PROTEIN_RESOURCES_H100",
]
