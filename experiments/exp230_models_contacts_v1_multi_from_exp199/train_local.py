# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Run exp230's fine-tune IN THIS PROCESS on a standalone multi-GPU node.

``run_levanter_train_lm`` calls levanter's ``train_lm.main`` directly, so it does
not need an iris scheduler -- which is what makes this possible. The alternative,
``dispatch_train.py``, submits a TPU gang and is the right path when the v5p pool
has capacity; it has had none all day (zero ready, zero booting, and zero
registered demand, so the autoscaler never even saw the request).

Everything is local: the corpus is written on this node by ``build_corpus.py`` /
``tokenize_corpus.py``, and the warm-start weights are already here from
generation. Nothing crosses the workstation's ~2.4 MB/s uplink.

The recipe is unchanged from ``train_common.py`` -- #163's arm F against exp199 --
except that resources are 8 local GPUs rather than a TPU slice, and it is a
**single epoch**, so ``--steps`` should be exactly the ``STEPS_PER_EPOCH``
``tokenize_corpus.py`` printed.

    WANDB_API_KEY=... ./.venv/bin/python train_local.py \\
        --corpus '/home/ubuntu/exp230_data/tokenized/*.parquet' \\
        --val '/home/ubuntu/exp230_data/val/*.parquet' \\
        --init /home/ubuntu/exp230_data/model/exp199 \\
        --out /home/ubuntu/exp230_data/checkpoints --steps <STEPS_PER_EPOCH>
"""
from __future__ import annotations

import argparse
import os
import sys

from fray.types import ResourceConfig
from marin.training.training import run_levanter_train_lm

import train_common


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="glob of tokenized parquets")
    ap.add_argument("--val", required=True, help="glob of raw contacts-v1 val parquets")
    ap.add_argument("--init", required=True, help="exp199 HF export to warm-start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True,
                    help="ONE epoch: use tokenize_corpus.py's printed STEPS_PER_EPOCH")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps-per-eval", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        raise SystemExit("WANDB_API_KEY must be set; marin refuses to build the config without it")

    run_name = a.run_name or f"plm-exp230-cv1-multi-1_5b-lr{a.lr:.0e}-e1-cos-a100".replace(
        "e-0", "e-")
    resources = ResourceConfig.with_gpu("A100", count=a.gpus, replicas=1)

    cfg = train_common.build_on_pod_config(
        run_name=run_name,
        learning_rate=a.lr,
        num_train_steps=a.steps,
        output_path=a.out,
        corpus_glob=a.corpus,
        val_glob=a.val,
        init_from_hf=a.init,
        resources=resources,
        env_vars={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
        steps_per_eval=a.steps_per_eval,
        steps_per_checkpoint=250,
        # A readable export every 250 steps is what makes the leak-vs-steps curve
        # measurable rather than assumed -- it is a RESULT of this experiment, not
        # a detail. Each export is a full copy of the weights.
        hf_save_steps=250,
        tags=("exp230", "contacts-v1-multi", "exp199-warmstart", "single-epoch"),
    )

    print(f"[exp230] {run_name}\n"
          f"         {a.steps} steps (ONE epoch) x batch {train_common.TRAIN_BATCH} "
          f"x seq {train_common.SEQ_LEN}\n"
          f"         {a.gpus} x A100 | lr {a.lr} cosine | warm start {a.init}\n"
          f"         corpus {a.corpus}\n         val {a.val}\n         out {a.out}")
    if a.dry_run:
        print("[exp230] DRY RUN -- config built, not training.")
        return 0
    run_levanter_train_lm(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
