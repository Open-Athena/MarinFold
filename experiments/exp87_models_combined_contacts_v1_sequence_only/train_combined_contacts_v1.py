# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~1.47B-param Llama on the COMBINED contacts-v1 + sequence-only corpus — issue #87.

One epoch over a token-proportional mixture of the with-structure ``contacts-v1``
corpus (~4.7B train tok) and the ~7× larger sequence-only ``contacts-v1.sequence_only``
corpus (~32.65B train tok). Each batch mixes both kinds and is ~87% sequence-only
(mixture weights ∝ corpus size). The train stream is fully shuffled (Feistel).
Per-document-type eval losses (``eval/contacts-v1-val/loss`` and
``eval/sequence-only-val/loss``) are reported separately on W&B.

Recipe = the #67 1.5B recipe scaled to a bigger, less-contended slice (the v5p-8
preemptible pool thrashed for #85). v5p-32 = 4× #67's v5p-8, so the global batch
is 4× (128→512; per-chip batch stays 32 ⇒ same memory, no OOM) and the peak LR is
√4 = 2× #67's 3.5e-4 ⇒ 7.0e-4 (standard LR-vs-batch sqrt scaling, per issue #87).

Fresh init (random weights) — this is a from-scratch one-epoch run, NOT a warm
restart. The unified 2846-token tokenizer also makes #67/#85 checkpoints
vocab-compatible, but the issue asks for a fresh epoch on the combined data.

All executor output (the reused contacts-v1 caches, the fresh sequence-only
cache, checkpoints, HF exports) lives under exp67's ``MARIN_PREFIX`` (set in
``contacts_v1_train_common``), namespaced by this run's name.

Usage::

    WANDB_API_KEY=<key> uv run iris --cluster marin job run --no-wait \\
        --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \\
        --extra=tpu --zone=us-east5-a \\
        -- python -m train_combined_contacts_v1
"""

import math

from fray import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.execution import executor_main

from contacts_v1_train_common import COMBINED_TRAIN_TOKENS, build_train_step

# 1.5B shape — matches Pythia-1.4B (h=2048, l=24, dff=8192, heads=32), identical
# to exp0/exp67's ``protein_llama_1_5b``.
protein_llama_1_5b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
)

# Bigger, less-contended slice (v5p-8 preemptible pool thrashed for #85). v5p-32
# = 4× #67's chips ⇒ batch 4× (128→512, per-chip 32, same memory) and LR √4 = 2×.
RESOURCES_V5P32 = ResourceConfig.with_tpu(
    "v5p-32", slice_count=1, cpu=32, ram="128g", disk="50g", zone="us-east5-a",
)
TRAIN_BATCH = 512
TRAIN_SEQ_LEN = 8192
PEAK_LR = 7.0e-4  # 3.5e-4 (#67 @ batch 128) × √(512/128)

# One epoch over the combined corpus. tokens/step = 512 × 8192 = 4.19M;
# COMBINED_TRAIN_TOKENS ≈ 37.36B ⇒ ~8,910 steps. (Mixture sampling is with
# replacement, so "one epoch" is in expectation: each corpus is drawn in
# proportion to its size over this many steps.)
EPOCH_STEPS = math.ceil(COMBINED_TRAIN_TOKENS / (TRAIN_BATCH * TRAIN_SEQ_LEN))

RUN_NAME = "protein-contacts-1_5b-combined-seqonly-7e-4"

protein_model_1_5b_combined = build_train_step(
    name=RUN_NAME,
    model_config=protein_llama_1_5b,
    learning_rate=PEAK_LR,
    num_train_steps=EPOCH_STEPS,
    train_batch_size=TRAIN_BATCH,
    train_seq_len=TRAIN_SEQ_LEN,
    data_seed=0,
    extra_tags=("1_5b", "combined", "v5p32", "bs512", "epoch1"),
    wandb_name=RUN_NAME,
    resources=RESOURCES_V5P32,
    # Long-ish run on a preemptible pool — keep periodic permanent checkpoints so
    # a late preemption can't lose much.
    steps_per_export=1000,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1_5b_combined])
