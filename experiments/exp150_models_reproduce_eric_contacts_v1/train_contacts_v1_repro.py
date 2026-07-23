# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp150 launcher: reproduce Eric's best contacts-v1 1.5B run on MarinFold's own path.

Defaults reproduce exp117 point ``e16-lr3p162e-3-wd0p2-bs128`` exactly:
LR 3.1623e-3, WD 0.2, global batch 128, seq 8192, 16 epochs = **71,360 steps**
(74.86B tokens), cosine + 10% warmup, AdamW betas 0.9/0.95, unmasked,
``pack=True``, hierarchical Feistel block shuffle at ``data_seed=0``, full-val
eval every 2,230 steps, one permanent checkpoint every 4,460 steps.

Target: final ``eval/contacts-v1-val/loss`` = **2.7112** (+-0.01).

See ``contacts_v1_repro_common`` for the full recipe and its provenance.

Knobs (env) -- all default to the target point; override only to debug:
* ``EXP150_PREVIEW``      ``yes`` -> print the resolved config and exit, submit nothing
* ``EXP150_SMOKE``        ``yes`` -> tiny isolated run (``EXP150_SMOKE_STEPS``, default 20)
* ``EXP150_EPOCHS``       epoch count (default 16); steps are derived from it
* ``EXP150_STEPS``        override ``num_train_steps`` outright (default: derived)
* ``EXP150_LR`` / ``EXP150_WD`` / ``EXP150_BATCH_SIZE``
* ``EXP150_NAME``         override the run / W&B name
* ``EXP150_TPU`` / ``EXP150_REGION`` / ``EXP150_ZONE`` / ``EXP150_SLICES``  (see common)
* ``EXP150_BUCKET``       region bucket for caches + checkpoints (default gs://marin-us-east5)
* ``WANDB_API_KEY``       forwarded into the pod

Usage (from the fresh marin checkout's iris, bundling this dir's pinned
pyproject -- see README for the iris-freshness rationale)::

    cd experiments/exp150_models_reproduce_eric_contacts_v1
    uv venv && uv sync --extra tpu

    # 1. verify the resolved config locally, submit nothing
    EXP150_PREVIEW=yes uv run python -m train_contacts_v1_repro

    # 2. smoke on a small slice
    EXP150_SMOKE=yes EXP150_TPU=v5p-8 WANDB_API_KEY=<key> \\
      /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \\
        --enable-extra-resources --cpu=1 --memory=16GB --disk=16GB \\
        --extra=tpu --region=us-east5 \\
        -e WANDB_API_KEY <key> -e WANDB_ENTITY open-athena -e EXP150_SMOKE yes \\
        -e EXP150_TPU v5p-8 -- python -m train_contacts_v1_repro

    # 3. the real run
    WANDB_API_KEY=<key> \\
      /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \\
        --enable-extra-resources --cpu=1 --memory=16GB --disk=16GB \\
        --extra=tpu --region=us-east5 \\
        -e WANDB_API_KEY <key> -e WANDB_ENTITY open-athena \\
        -- python -m train_contacts_v1_repro
"""
from __future__ import annotations

import os

from marin.execution import executor_main

import contacts_v1_repro_common as common
from contacts_v1_repro_common import (
    SEQ_LEN,
    TARGET_BATCH_SIZE,
    TARGET_EPOCHS,
    TARGET_LR,
    TARGET_VAL_LOSS,
    TARGET_WD,
    TRAIN_TOKENS,
    build_train_step,
    evals_per_epoch_steps,
    steps_for_epochs,
    steps_per_epoch,
)

SMOKE_STEPS_DEFAULT = 20


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"yes", "true", "1"}


def _fmt_lr(lr: float) -> str:
    """exp117's ``_fmt_lr``: 3.1623e-3 -> ``3p162e-3``, 1e-3 -> ``1e-3``."""
    mantissa, exponent = f"{lr:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def _fmt_wd(wd: float) -> str:
    """exp117's ``_fmt_wd``: 0.2 -> ``0p2``."""
    return f"{wd:g}".replace(".", "p")


def resolve() -> dict:
    """Resolve the run parameters from the environment (all defaulted to the target)."""
    smoke = _env_flag("EXP150_SMOKE")
    epochs = int(os.environ.get("EXP150_EPOCHS", str(TARGET_EPOCHS)))
    lr = float(os.environ.get("EXP150_LR", str(TARGET_LR)))
    wd = float(os.environ.get("EXP150_WD", str(TARGET_WD)))
    batch_size = int(os.environ.get("EXP150_BATCH_SIZE", str(TARGET_BATCH_SIZE)))

    steps = int(os.environ.get("EXP150_STEPS", "0")) or steps_for_epochs(epochs, batch_size)
    per_epoch = steps_per_epoch(batch_size)
    steps_per_eval = evals_per_epoch_steps(batch_size)
    steps_per_export = per_epoch

    # exp117's Point.point_id, so the run name carries the same point label he uses.
    point = f"e{epochs}-lr{_fmt_lr(lr)}-wd{_fmt_wd(wd)}-bs{batch_size}"

    if smoke:
        # Tiny, identity-isolated end-to-end validation: real caches, real data
        # path, real checkpoint write, under a name that can never collide with
        # (or resume) the production run. Mirrors exp117's smoke_shape.
        steps = int(os.environ.get("EXP150_SMOKE_STEPS", str(SMOKE_STEPS_DEFAULT)))
        steps_per_eval = max(1, steps // 2)
        steps_per_export = steps_per_eval
        default_name = f"exp150-smoke-cv1repro-1_5b-{point}"
    else:
        default_name = f"exp150-cv1repro-1_5b-{point}"

    return {
        "name": os.environ.get("EXP150_NAME", default_name),
        "smoke": smoke,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": wd,
        "train_batch_size": batch_size,
        "num_train_steps": steps,
        "steps_per_eval": steps_per_eval,
        "steps_per_export": steps_per_export,
        "steps_per_epoch": per_epoch,
    }


def _preview(r: dict) -> None:
    tokens = r["train_batch_size"] * SEQ_LEN * r["num_train_steps"]
    mode = "SMOKE" if r["smoke"] else "PRODUCTION"
    print(
        f"PREVIEW exp150 [{mode}] -- nothing submitted\n"
        f"  run/W&B name   {r['name']}\n"
        f"  reproducing    prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs128-us-east5\n"
        f"  target loss    eval/contacts-v1-val/loss = {TARGET_VAL_LOSS} (+-0.01)\n"
        f"\n"
        f"  epochs         {r['epochs']}\n"
        f"  learning rate  {r['learning_rate']:g}\n"
        f"  weight decay   {r['weight_decay']:g}\n"
        f"  batch / seq    {r['train_batch_size']} x {SEQ_LEN} = {r['train_batch_size'] * SEQ_LEN:,} tok/step\n"
        f"  steps/epoch    {r['steps_per_epoch']:,}   (round({TRAIN_TOKENS:,} / {r['train_batch_size'] * SEQ_LEN:,}))\n"
        f"  num_train_steps{r['num_train_steps']:>10,}\n"
        f"  total tokens   {tokens / 1e9:.2f}B\n"
        f"  steps_per_eval {r['steps_per_eval']:,}   (full val, max_eval_batches=None)\n"
        f"  perm ckpt every{r['steps_per_export']:>10,}\n"
        f"\n"
        f"  shuffle        {common.SHUFFLE}\n"
        f"  data_seed      {common.DATA_SEED}\n"
        f"  tokenizer      {common.CONTACTS_V1_TOKENIZER} (pin {common.CONTACTS_V1_TOKENIZER_REVISION}, vocab {common.VOCAB_SIZE})\n"
        f"  train glob     {common.CONTACTS_V1_TRAIN_GLOB}\n"
        f"  val glob       {common.CONTACTS_V1_VAL_GLOB}\n"
        f"  MARIN_PREFIX   {common.MARIN_PREFIX}\n"
        f"  resources      {common.PROTEIN_RESOURCES}",
        flush=True,
    )


def build_steps() -> list:
    r = resolve()
    extra_tags = [f"steps{r['num_train_steps']}", f"epochs{r['epochs']}"]
    if r["smoke"]:
        extra_tags.append("smoke")
    return [
        build_train_step(
            name=r["name"],
            learning_rate=r["learning_rate"],
            weight_decay=r["weight_decay"],
            num_train_steps=r["num_train_steps"],
            train_batch_size=r["train_batch_size"],
            steps_per_eval=r["steps_per_eval"],
            steps_per_export=r["steps_per_export"],
            extra_tags=tuple(extra_tags),
            wandb_name=r["name"],
        )
    ]


if __name__ == "__main__":
    if _env_flag("EXP150_PREVIEW"):
        _preview(resolve())
    else:
        executor_main(steps=build_steps())
