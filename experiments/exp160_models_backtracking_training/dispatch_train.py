# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the #160 fine-tune on a marin TPU slice.

Builds and runs the ``iris job run`` command, so the launch is reproducible and
the reasoning below travels with it. Run it from this directory:

    uv run python dispatch_train.py            # submit
    uv run python dispatch_train.py --dry-run  # print the command only

Why a plain CLI submission and not a driver job
-----------------------------------------------
The abandoned CoreWeave revision used a small CPU driver job that submitted the
real training job as a *child* at batch priority. That indirection exists only
because CoreWeave requires an explicit batch band on every child JobRequest
(the band does not propagate from the CLI). On the marin cluster there is no
such rule, so the training job is submitted directly as a root job: fewer
moving parts, and every launch failure in this experiment so far came from the
driver/child plumbing rather than from training itself.

Why v5p-32 in us-east5-a
------------------------
Measured against the live controller DB on 2026-07-28, not assumed:

- **Family.** 1,642 tasks were pending cluster-wide: 896 wanted v6e, 429 v5e,
  and only 101 v5p. Every one of those 101 was band 3 (BATCH) from one user's
  ``extract-lpv11`` sweep, wanting v5p-32/64. We submit at band 2
  (INTERACTIVE), which outranks all of it. Band 1 (PRODUCTION) had 258 tasks
  running and 0 pending, so the queue is not blocking the upper bands.
- **Zone.** ``us-central1-a`` is exhausted: its live ``quota_reason`` is
  ``There is no more capacity in the zone "us-central1-a"``, its 86 v5p-8
  slices are all busy (ages up to 22.7 h) and no new slice had reached ready in
  125 min. ``us-east5-a`` is actively provisioning: 20 v5p-8 and 12 v5p-32
  slices reached ready in the preceding 6 h (newest 14 min old), with 26 + 11
  more booting and no live quota block.
  NB when re-checking: ``scaling_groups.last_scale_up_ms`` is written by
  ``begin_scale_up``, so it timestamps *attempts*, not successes — read actual
  ``ready`` transitions out of the ``slices`` table instead.
- **Region.** ``gs://marin-us-east5`` is a regional bucket in US-EAST5, so the
  corpus, the 17.7 GiB exp120 init checkpoint and the outputs are all local to
  a us-east5-a slice. Nothing needs staging, and marin's TPU-path
  ``check_gcs_paths_same_region`` assertion passes.
- **Size.** ~1.9e19 FLOPs (1.47B params x 2.16B tokens). v5p-32 is 4 VMs x 4
  chips; at ~40% MFU that is ~2 h, which fits inside the 4.5 h maximum slice
  age observed in that group. v5p-8 would be ~7 h and would almost certainly
  span a preemption. Not v6e-8 (exp120's shape): us-east5-b v6e is now the most
  contested pool on the cluster and reports no zone capacity.

Prerequisites, both verified before launch:

  1. corpus + tokenizer at ``GCS_PREFIX/corpus`` — 32 shards, 3.54 GB, byte
     sizes diffed against the local build
  2. exp120's Levanter checkpoint, **vocab-resized to 3849**, at ``INIT`` —
     see ``resize_init_vocab.py``. exp120's own checkpoint lives in us-east5
     already (the CoreWeave S3 copy is now unnecessary) but is 2845 tokens
     wide, which Levanter's strict warm-start load rejects.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess

# Fresh iris client: the workstation's default marin checkout is stale and is
# rejected by the 14-day client-freshness gate.
IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
CLUSTER = "marin"

GCS_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp160_backtracking_training"
DEFAULT_CORPUS = f"{GCS_PREFIX}/corpus"
DEFAULT_OUTPUT = f"{GCS_PREFIX}/runs"
DEFAULT_TOKENIZER = f"{GCS_PREFIX}/corpus/tokenizer"
# exp120's weights grown to the 3849-token superset vocab by
# resize_init_vocab.py — NOT exp120's own checkpoint, which is 2845-wide and
# fails Levanter's strict warm-start load against the superset tokenizer.
DEFAULT_INIT = f"{GCS_PREFIX}/init/exp120-step-1005-vocab3849"

TPU_TYPE = "v5p-32"
TPU_ZONE = "us-east5-a"
SEQ_LEN = 8192


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.environ.get("EXP160_CORPUS", DEFAULT_CORPUS))
    ap.add_argument("--init", default=os.environ.get("EXP160_INIT", DEFAULT_INIT))
    ap.add_argument("--output", default=os.environ.get("EXP160_OUTPUT", DEFAULT_OUTPUT))
    ap.add_argument("--tokenizer", default=os.environ.get("EXP160_TOKENIZER", DEFAULT_TOKENIZER))
    ap.add_argument("--tpu-type", default=os.environ.get("EXP160_TPU", TPU_TYPE))
    ap.add_argument("--tpu-zone", default=os.environ.get("EXP160_ZONE", TPU_ZONE))
    ap.add_argument(
        "--run-name",
        default=os.environ.get("EXP160_RUN_NAME", "exp160-cv1-1_5b-bt50-lr3e-4-e1-cos"),
    )
    # exp120's continue-train LR (3e-4, 1-epoch cosine) rather than #117's
    # 3.16e-3 pretraining LR — this is a fine-tune from a converged model.
    ap.add_argument("--learning-rate", type=float, default=float(os.environ.get("EXP160_LR", "3e-4")))
    ap.add_argument("--weight-decay", type=float, default=float(os.environ.get("EXP160_WD", "0.2")))
    ap.add_argument("--train-batch-size", type=int, default=int(os.environ.get("EXP160_BATCH", "128")))
    ap.add_argument("--warmup", type=float, default=float(os.environ.get("EXP160_WARMUP", "0.1")))
    # ~2.16B tokens over the 2.05M-document mix; one epoch.
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("EXP160_TOKENS", "2160000000")))
    ap.add_argument("--num-train-steps", type=int, default=int(os.environ.get("EXP160_STEPS", "0")))
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    steps = args.num_train_steps or max(1, args.tokens // (args.train_batch_size * SEQ_LEN))
    worker = (
        "uv run python train_backtracking.py "
        f"--corpus {args.corpus} "
        f"--init {args.init} "
        f"--output {args.output} "
        f"--tokenizer {args.tokenizer} "
        f"--run-name {args.run_name} "
        f"--learning-rate {args.learning_rate} "
        f"--weight-decay {args.weight_decay} "
        f"--train-batch-size {args.train_batch_size} "
        f"--num-train-steps {steps} "
        f"--warmup {args.warmup} "
        f"--tpu-type {args.tpu_type} "
        f"--tpu-zone {args.tpu_zone}"
    )
    return [
        IRIS, f"--cluster={CLUSTER}", "job", "run", "--no-wait",
        "--tpu", args.tpu_type,
        "--zone", args.tpu_zone,
        # --tpu/--gpu and any large CPU/RAM/disk request need this opt-in.
        "--enable-extra-resources",
        # INTERACTIVE (band 2) — our cap as a non-admin, and enough to outrank
        # the entire pending v5p queue, which is band 3. The CoreWeave
        # always-batch rule is specific to those GPU clusters.
        "--priority", "interactive",
        "--extra", "tpu",
        # Bounded by what the pool advertises per VM (marin.yaml v5p-preemptible:
        # cpu 208, ram 448GB, disk 100GB). Asking for more is not a queue — the
        # controller rejects the submission outright with "no matching scaling
        # group has enough per-VM capacity". 100GB of local disk is ample: the
        # token cache is written to GCS, not to the pod.
        "--cpu", "200", "--memory", "400GB", "--disk", "100GB",
        # The v5p pool is capacity_type: preemptible whatever band we submit
        # at, so the run must be able to come back. Levanter resumes from its
        # own 10-minute rolling checkpoint.
        "--max-retries", "100",
        "--job-name", args.run_name,
        "-e", "WANDB_API_KEY", os.environ.get("WANDB_API_KEY", ""),
        "-e", "TOKENIZERS_PARALLELISM", "false",
        "--", "bash", "-lc", worker,
    ]


def main() -> None:
    args = build_args()
    cmd = build_command(args)
    printable = [("<WANDB_API_KEY>" if c and c == os.environ.get("WANDB_API_KEY") else c) for c in cmd]
    print("+ " + " ".join(shlex.quote(c) for c in printable), flush=True)
    if args.dry_run:
        return
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set; the run would train untracked.")
    # check=False on purpose: CalledProcessError renders the whole argv in its
    # message, which would echo the W&B key into logs and terminal scrollback.
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise SystemExit(f"iris job run failed (exit {result.returncode}); see the error above.")


if __name__ == "__main__":
    main()
