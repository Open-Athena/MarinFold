# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp120 training sweep: regenerated (Arm B) vs re-epoch original (Arm A), matched.

Continue-train Eric's tuned contacts-v1 1.5B (eval loss 2.7566) under two arms
that differ ONLY in the training document content (see ``contacts_v1_ft_common``):

* **Arm A (re-epoch):** original contacts-v1 round-0 documents.
* **Arm B (regenerated):** exp100 only-correct documents, same proteins.

Two schedule modes per (arm, LR):

* **headline** — a dedicated **1-epoch cosine** run (peak LR -> ``min_lr_ratio``
  floor over exactly one epoch). This is issue #120's primary comparison point:
  a properly-decayed 1-epoch model per arm.
* **curve** — a **4-epoch flat-LR** run (cosine with ``min_lr_ratio=1.0`` == peak
  held constant after warmup) with a permanent checkpoint at **every epoch
  boundary**, so the 1/2/3/4-epoch checkpoints are directly comparable (no
  cosine-phase confound) and we can watch the regenerated arm for
  overfitting/collapse across passes (issue #120's "report the full curve").

Both arms use the SAME num_train_steps for a given epoch count (their token
counts are identical — same proteins, same contact sets, order-only difference),
so the budget is matched.

Batch is **128** (== Eric's #75 training batch), so the swept LRs
{1e-4, 3e-4} are directly interpretable as "below his 1e-3" with no batch-scaling
confound. v5p-8 @ us-east5-a; marin's checkpoint-resume covers preemptions.

Knobs (env):
* ``EXP120_ARMS``            default ``A,B``
* ``EXP120_LRS``             default ``1e-4,3e-4``
* ``EXP120_MODE``            ``headline`` | ``curve`` | ``both`` (default ``headline``)
* ``EXP120_STEPS_PER_EPOCH`` REQUIRED — measured token count / (128*8192); print
                             it from ``build_arm_a_aligned.py`` (both arms match).
* ``EXP120_MAX_EPOCHS``      default ``4`` (curve mode)
* ``WANDB_API_KEY``          forwarded into the pod.

Usage (headline first, then the curve)::

    cd experiments/exp120_models_regen_vs_reepoch_contacts_v1
    uv venv && uv sync --extra tpu

    EXP120_MODE=headline EXP120_STEPS_PER_EPOCH=<N> WANDB_API_KEY=<key> \\
      uv run iris --cluster marin job run --no-wait --enable-extra-resources \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a \\
        -- python -m train_regen_vs_reepoch_sweep
"""
from __future__ import annotations

import os

from marin.execution import executor_main

from contacts_v1_ft_common import build_train_step

TRAIN_BATCH = 128


def _steps_per_epoch() -> int:
    v = os.environ.get("EXP120_STEPS_PER_EPOCH")
    if not v:
        raise SystemExit(
            "EXP120_STEPS_PER_EPOCH is required. Run build_arm_a_aligned.py — it "
            "prints total train tokens; steps/epoch = ceil(tokens / (128*8192)). "
            "Arm A and Arm B share the same value (identical token counts)."
        )
    return int(v)


def _arms() -> list[str]:
    return [a.strip() for a in os.environ.get("EXP120_ARMS", "A,B").split(",") if a.strip()]


def _lrs() -> list[float]:
    return [float(x) for x in os.environ.get("EXP120_LRS", "1e-4,3e-4").split(",") if x.strip()]


def _lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("-0", "-")


def build_steps() -> list:
    mode = os.environ.get("EXP120_MODE", "headline")
    max_epochs = int(os.environ.get("EXP120_MAX_EPOCHS", "4"))
    spe = _steps_per_epoch()
    steps = []

    for arm in _arms():
        arm_slug = "orig" if arm == "A" else "regen"
        for lr in _lrs():
            lr_tag = _lr_tag(lr)

            if mode in ("headline", "both"):
                # 1-epoch cosine (decays peak -> 0.1*peak over the single epoch).
                name = f"exp120-cv1-1_5b-{arm_slug}-{lr_tag}-e1-cos"
                steps.append(
                    build_train_step(
                        name=name,
                        arm=arm,
                        learning_rate=lr,
                        lr_schedule="cosine",
                        min_lr_ratio=0.1,
                        num_train_steps=spe,
                        train_batch_size=TRAIN_BATCH,
                        steps_per_eval=max(1, spe // 4),
                        steps_per_export=max(1, spe // 2),
                        extra_tags=("headline", "e1", lr_tag),
                        wandb_name=name,
                    )
                )

            if mode in ("curve", "both"):
                # 4-epoch flat LR (min_lr_ratio=1.0 == constant after warmup) with
                # a permanent checkpoint at each epoch boundary.
                name = f"exp120-cv1-1_5b-{arm_slug}-{lr_tag}-e{max_epochs}-flat"
                steps.append(
                    build_train_step(
                        name=name,
                        arm=arm,
                        learning_rate=lr,
                        lr_schedule="cosine",
                        min_lr_ratio=1.0,  # flat: peak held after warmup
                        num_train_steps=spe * max_epochs,
                        train_batch_size=TRAIN_BATCH,
                        steps_per_eval=max(1, spe // 4),
                        steps_per_export=spe,  # checkpoint every epoch boundary
                        extra_tags=("curve", f"e{max_epochs}", lr_tag),
                        wandb_name=name,
                    )
                )
    return steps


if __name__ == "__main__":
    executor_main(steps=build_steps())
