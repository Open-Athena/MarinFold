# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the exp112 NeMo Qwen3-3B MFU benchmark (issue #112).

exp112 asks the same question as #108 — train the 3B Qwen3 contacts-v1 model on
the CoreWeave RNO2A H100 cluster — but on the **NVIDIA NeMo / Megatron-Core**
stack instead of JAX/Levanter, to see whether we beat exp108's poor **~15% MFU**.

Unlike exp108, the *training* stack does NOT live in this Python environment: it
runs inside the ``nvcr.io/nvidia/nemo`` container. This module + ``dispatch_nemo``
are the **launcher** only — they submit a single 8xH100 iris job (batch priority)
whose container runs ``torchrun`` over the in-container training script
(``nemo_train_qwen3.py``). See the README.

The model geometry mirrors exp108 exactly (depth-doubled #75 1.5B width -> ~2.9B),
so the MFU comparison is apples-to-apples: same params / seq / global batch /
hardware, only the framework differs.
"""

import os

from fray import ResourceConfig

# ---------------------------------------------------------------------------
# Container: the NeMo NGC image carries Megatron-Core + Transformer Engine +
# NeMo 2.0. Pinned to an exact patch tag for reproducibility; override with
# EXP112_IMAGE. (nvcr.io/nvidia/nemo:25.04.02 is anonymously pullable; whether
# the CoreWeave nodes can pull nvcr.io without a pull secret is verified by the
# smoke run — see README "Watch-items".)
# ---------------------------------------------------------------------------
NEMO_IMAGE = os.environ.get("EXP112_IMAGE", "nvcr.io/nvidia/nemo:25.04.02")

# ---------------------------------------------------------------------------
# Storage: everything under one removable `MarinFold/` prefix (issue #108's
# convention, inherited). The benchmark writes only W&B + a tiny JSON summary;
# the deferred real run would put checkpoints here.
# ---------------------------------------------------------------------------
EXP112_S3_PREFIX = "s3://marin-us-east-02a/MarinFold/exp112_qwen_3b_nemo_mfu"
# Megatron .bin/.idx corpus (produced by prepare_megatron_data.py); the bootstrap
# downloads these to node-local disk before torchrun.
EXP112_DATA_S3 = f"{EXP112_S3_PREFIX}/tokenized_megatron"
# Sharded checkpoints (synced per-node; the only durable store — no shared FS).
EXP112_CKPT_S3_BASE = f"{EXP112_S3_PREFIX}/checkpoints"
# Multi-node torchrun rendezvous: rank-0 publishes its IP here (attempt-scoped).
EXP112_RDZV_S3_BASE = f"{EXP112_S3_PREFIX}/rdzv"

# contacts-v1 tokenizer (2845 real tokens; Megatron pads the embedding up to a
# multiple of `make_vocab_size_divisible_by`). Loaded from HF inside the
# container (public, tiny) so even the mock-data benchmark uses the REAL vocab
# size -> faithful embedding/LM-head FLOPs. Bare id (no @rev) on the HF load path.
CONTACTS_V1_TOKENIZER = "timodonnell/contacts-v1-tokenizer"

# W&B routing (matches exp108 / the MarinFold project).
WANDB_ENTITY = "open-athena"
WANDB_PROJECT = "MarinFold"

# ---------------------------------------------------------------------------
# Model — Qwen3 ~2.9B, IDENTICAL geometry to exp108 (#75's 1.5B width, depth
# doubled 24 -> 48). These are the FLOP-determining dims, so keeping them exact
# is what makes the MFU number comparable to exp108's Levanter run.
#   hidden 2048 / ffn 8192 (=4h) / 32 heads / 8 KV groups (GQA) / head_dim 64 /
#   48 layers / seq 8192. Qwen3 specifics: RMSNorm, SwiGLU, QK-layernorm,
#   Llama/GPT-NeoX-noninterleaved RoPE, untied embeddings.
# ---------------------------------------------------------------------------
MODEL = dict(
    num_layers=48,
    hidden_size=2048,
    ffn_hidden_size=8192,
    num_attention_heads=32,
    num_query_groups=8,   # GQA; Megatron's name for HF num_key_value_heads
    kv_channels=64,       # head_dim (= 2048/32); set explicit to be safe
    seq_length=8192,
)

# Recipe (tracks #75 / exp108). Not all matter for MFU, but keep them faithful
# for the deferred real run.
GLOBAL_BATCH_SIZE = 128   # sequences; 128 * 8192 ~= 1.05M tokens/step (= exp108)
WEIGHT_DECAY = 0.2
WARMUP_FRACTION = 0.1
LEARNING_RATE = 1e-3      # #75's winner

# 16-epoch schedule (identical to exp108). Computed from the contacts-v1 train
# token count so it stays correct if the batch changes.
TRAIN_TOKENS = 4_672_623_743  # exp53 generation_stats: 4,129,682 docs
EPOCHS = 16
SEQ_LEN = MODEL["seq_length"]


def num_train_steps(global_batch: int = GLOBAL_BATCH_SIZE, epochs: int = EPOCHS) -> int:
    import math
    return epochs * math.ceil(TRAIN_TOKENS / (global_batch * SEQ_LEN))  # ≈71,312 @ bs128

# ---------------------------------------------------------------------------
# Device: a single 8xH100 node on cw-rno2a. torchrun --standalone drives the 8
# local GPUs (no multi-node rendezvous -> sidesteps iris's missing coordinator
# API). One iris task == one pod == one node; replicas=1.
# ---------------------------------------------------------------------------
def h100_resources(image: str = NEMO_IMAGE, replicas: int = 1) -> ResourceConfig:
    """`replicas` × 8xH100 nodes in the NeMo container (gang-scheduled). The full
    run uses replicas=4 (32 GPU); the benchmark used replicas=1."""
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        replicas=replicas,
        image=image,
        cpu=64,
        ram="512g",
        disk="512g",
    )


# H100 SXM bf16 dense peak (TFLOP/s), for MFU = achieved_TFLOPs / PEAK. NVIDIA's
# convention (their perf tables quote model-TFLOP/s/GPU; divide by this for MFU).
H100_BF16_PEAK_TFLOPS = 989.0

__all__ = [
    "NEMO_IMAGE",
    "EXP112_S3_PREFIX",
    "CONTACTS_V1_TOKENIZER",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "MODEL",
    "GLOBAL_BATCH_SIZE",
    "WEIGHT_DECAY",
    "WARMUP_FRACTION",
    "LEARNING_RATE",
    "h100_resources",
    "H100_BF16_PEAK_TFLOPS",
]
