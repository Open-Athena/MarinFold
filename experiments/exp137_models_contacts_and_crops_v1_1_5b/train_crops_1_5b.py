# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp137 launcher: train a 1.5B protein LM on contacts-and-crops-v1 from scratch.

Reproduces Eric's exp117 best config (LR 3.1623e-3, WD 0.2, global batch 128,
seq 8192, cosine + 10% warmup, betas 0.9/0.95, unmasked, pack=True, Feistel
data_seed=0) on the 8k crops corpus. See ``crops_train_common`` for the shared
config.

**Token budget.** Eric's exp117 leader (contacts-v1-val 2.7112) ran 16 epochs of
contacts-v1 = **71,359 steps** @ batch 128 x seq 8192 = 74.8B tokens. crops is
~33.82B train tokens (exp132 stats.json), so 71,359 steps == matching Eric's
*token budget* (~2.21 crops epochs) rather than his epoch count -- the right
apples-to-apples reference for a long from-scratch run. This is the default.

Knobs (env):
* ``EXP137_STEPS``          num_train_steps (default 71359 = Eric 16ep-equiv)
* ``EXP137_LR``             peak learning rate (default 3.1623e-3)
* ``EXP137_WD``             weight decay (default 0.2)
* ``EXP137_STEPS_PER_EVAL`` default 2000
* ``EXP137_STEPS_PER_EXPORT`` permanent-checkpoint interval (default 10000)
* ``EXP137_MAX_EVAL_BATCHES`` cap eval batches per val component (default all)
* ``EXP137_NAME``           override the run/W&B name
* ``EXP137_TPU`` / ``EXP137_ZONE`` / ``EXP137_SLICES``  TPU slice (see common)
* ``WANDB_API_KEY``         forwarded into the pod

Usage (from the fresh marin checkout's iris, bundling this dir's old-marin
pyproject -- see README for the iris-freshness rationale)::

    cd experiments/exp137_models_contacts_and_crops_v1_1_5b
    uv venv && uv sync --extra tpu

    WANDB_API_KEY=<key> \\
      /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \\
        --enable-extra-resources --cpu=1 --memory=16GB --disk=16GB \\
        --extra=tpu --zone=us-east5-a \\
        -e WANDB_API_KEY <key> -e WANDB_ENTITY open-athena \\
        -- python -m train_crops_1_5b
"""
from __future__ import annotations

import os

from marin.execution import executor_main

from crops_train_common import build_train_step


# Exact TRAIN-token counts (for epoch-based mixture budgeting).
CROPS_TRAIN_TOKENS = 33_824_241_857  # exp132 stats.json (contacts-and-crops-v1 train)
CV1_TRAIN_TOKENS = 4_676_753_425     # Eric exp117 TRAIN_TOKENS (contacts-v1 train)
_TOK_PER_STEP = 128 * 8192


def _lr_tag(lr: float) -> str:
    # 3.1623e-3 -> "lr3p162e-3" (matches Eric's exp117 naming)
    s = f"{lr:.3e}".replace("-0", "-")
    mant, exp = s.split("e")
    return "lr" + mant.replace(".", "p") + "e" + exp


def build_steps() -> list:
    steps = int(os.environ.get("EXP137_STEPS", "71359"))  # Eric 16ep-equiv
    lr = float(os.environ.get("EXP137_LR", "3.1623e-3"))
    wd = float(os.environ.get("EXP137_WD", "0.2"))
    steps_per_eval = int(os.environ.get("EXP137_STEPS_PER_EVAL", "2000"))
    steps_per_export = int(os.environ.get("EXP137_STEPS_PER_EXPORT", "10000"))
    _mev = os.environ.get("EXP137_MAX_EVAL_BATCHES")
    max_eval_batches = int(_mev) if _mev else None
    # contacts-v1 mix-in fraction of TRAIN tokens (0 = crops-only, the default run;
    # e.g. 0.05 = 5% standalone contacts-v1 docs alongside the crops bulk, a la #121).
    cv1_mix = float(os.environ.get("EXP137_CV1_MIX", "0"))

    # Epoch-based mixture (overrides EXP137_STEPS + EXP137_CV1_MIX). Set both
    # EXP137_CROPS_EPOCHS and EXP137_CV1_EPOCHS to train an exact number of passes
    # over each corpus (e.g. 1 crops epoch + 8 contacts-v1 epochs). The sampling
    # weights are the token fractions, and num_train_steps is the summed token
    # budget / (batch*seq), so each corpus is seen exactly its requested # of times.
    _ce = os.environ.get("EXP137_CROPS_EPOCHS")
    _ve = os.environ.get("EXP137_CV1_EPOCHS")
    epoch_tag = ""
    if _ce is not None and _ve is not None:
        crops_ep, cv1_ep = float(_ce), float(_ve)
        crops_tok = crops_ep * CROPS_TRAIN_TOKENS
        cv1_tok = cv1_ep * CV1_TRAIN_TOKENS
        total_tok = crops_tok + cv1_tok
        steps = round(total_tok / _TOK_PER_STEP)
        cv1_mix = cv1_tok / total_tok
        epoch_tag = f"crops{crops_ep:g}ep-cv1{cv1_ep:g}ep"

    slug = epoch_tag or ("cc1" if cv1_mix <= 0 else f"cc1mix{round(cv1_mix * 100)}")
    default_name = f"exp137-{slug}-1_5b-{_lr_tag(lr)}-wd{wd:g}".replace(".", "p") + "-bs128"
    name = os.environ.get("EXP137_NAME", default_name)

    return [
        build_train_step(
            name=name,
            learning_rate=lr,
            weight_decay=wd,
            num_train_steps=steps,
            train_batch_size=128,
            steps_per_eval=steps_per_eval,
            steps_per_export=steps_per_export,
            max_eval_batches=max_eval_batches,
            contacts_v1_mix=cv1_mix,
            extra_tags=(f"steps{steps}",),
            wandb_name=name,
        )
    ]


if __name__ == "__main__":
    executor_main(steps=build_steps())
