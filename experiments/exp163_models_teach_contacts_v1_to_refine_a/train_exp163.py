# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp163 refinement fine-tune launcher (issue #163).

Warm-start Eric's tuned contacts-v1 1.5B (E8, eval loss 2.7566) and continue-
train it on the rollout-refinement corpus with the answer-span loss mask (see
``refine_ft_common``): the model learns to emit the TRUE contacts given a
sequence + K noisy candidate rollouts, with loss on the
``<begin_statements> … <end>`` answer span only.

Schedule matches exp120's *headline* arm — a dedicated **1-epoch cosine** run
(peak LR decays to the ``min_lr_ratio`` floor over exactly one epoch), at the
same low LRs {1e-4, 3e-4} (interpretable as "below Eric's 1e-3"),
``weight_decay=0.2``, ``warmup=0.1``, batch **128** (== Eric's #75 training
batch, no batch-scaling confound). Val (exp53 original contacts-v1 val) is
monitored unmasked every eval: the step-0 value should read ≈ 2.7566, confirming
the E8 warm-start loaded.

Knobs (env; ``learning_rate`` / ``num_train_steps`` are ``versioned()`` inside
``build_train_step``):

* ``EXP163_LRS``             default ``1e-4,3e-4`` (comma-separated peak LRs)
* ``EXP163_EPOCHS``          default ``1``
* ``EXP163_STEPS_PER_EPOCH`` REQUIRED unless ``EXP163_TRAIN_TOKENS`` is set —
                             one full pass = ceil(train_tokens / (128 * 8192)).
* ``EXP163_TRAIN_TOKENS``    alternative to STEPS_PER_EPOCH: total tokens in the
                             mirrored refinement corpus; steps/epoch is derived.
* ``WANDB_API_KEY``          forwarded into the pod.

Usage::

    cd experiments/exp163_models_teach_contacts_v1_to_refine_a
    uv venv && uv sync --extra tpu

    # (steps/epoch: print the mirrored corpus token count during the
    #  HF-bucket -> GCS mirror step, then ceil(tokens / (128 * 8192)).)
    EXP163_STEPS_PER_EPOCH=<N> WANDB_API_KEY=<key> \\
      uv run iris --cluster marin job run --no-wait --enable-extra-resources \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-b \\
        -- python -m train_exp163
"""
from __future__ import annotations

import math
import os

from marin.execution import executor_main

from refine_ft_common import build_train_step

TRAIN_BATCH = 128
TRAIN_SEQ_LEN = 8192


def steps_per_epoch(
    train_tokens: int, *, batch: int = TRAIN_BATCH, seq_len: int = TRAIN_SEQ_LEN
) -> int:
    """One full pass over the corpus: ceil(train_tokens / (batch * seq_len))."""
    return math.ceil(train_tokens / (batch * seq_len))


def _steps_per_epoch() -> int:
    steps = os.environ.get("EXP163_STEPS_PER_EPOCH")
    if steps:
        return int(steps)
    tokens = os.environ.get("EXP163_TRAIN_TOKENS")
    if tokens:
        return steps_per_epoch(int(tokens))
    raise SystemExit(
        "EXP163_STEPS_PER_EPOCH is required (or set EXP163_TRAIN_TOKENS and it "
        "is derived). steps/epoch = ceil(train_tokens / (128 * 8192)); print the "
        "corpus token count during the HF-bucket -> GCS mirror step."
    )


def _lrs() -> list[float]:
    return [float(x) for x in os.environ.get("EXP163_LRS", "1e-4,3e-4").split(",") if x.strip()]


def _epochs() -> int:
    return int(os.environ.get("EXP163_EPOCHS", "1"))


def _lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("-0", "-")


def build_steps() -> list:
    spe = _steps_per_epoch()
    epochs = _epochs()
    steps = []

    for lr in _lrs():
        lr_tag = _lr_tag(lr)
        # 1-epoch cosine headline run (decays peak -> min_lr_ratio floor over the
        # single epoch), matching exp120's primary comparison point.
        name = f"exp163-refine-cv1-1_5b-{lr_tag}-e{epochs}-cos"
        steps.append(
            build_train_step(
                name=name,
                learning_rate=lr,
                lr_schedule="cosine",
                min_lr_ratio=0.1,
                num_train_steps=spe * epochs,
                train_batch_size=TRAIN_BATCH,
                # weight_decay=0.2 / warmup=0.1 come from build_train_step defaults.
                steps_per_eval=max(1, spe // 4),
                steps_per_export=max(1, spe // 2),
                extra_tags=("headline", f"e{epochs}", lr_tag),
                wandb_name=name,
            )
        )
    return steps


if __name__ == "__main__":
    executor_main(steps=build_steps())
