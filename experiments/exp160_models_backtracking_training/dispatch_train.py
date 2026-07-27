# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tune of exp120 on the 50:50 backtracking mix, on CoreWeave H100s (#160).

Runs *inside* a small CPU driver job on iris and submits the training job at
**batch priority** (standing rule for any CoreWeave GPU cluster). Modelled on
exp108's GPU dispatch, which is the working precedent for Levanter training on
`cw-rno2a`.

Decisions and why:

- **Base model `contacts-v1-exp120-1.5B`.** Eric's #117 sweep has better
  validation loss (best 2.7037, bs256) but **every exp117 checkpoint has been
  deleted** — the surviving GCS dirs hold only executor metadata, and none are
  on the HF bucket. exp120 (2.7213) is the best contacts-v1 model that still
  loads, and it is also the model that *generated* the #159 corpus, so this
  fine-tunes a model on corrections of its own mistakes.
- **GPU, not TPU.** The marin TPU cluster was fully subscribed (0 chips free,
  39 jobs pending on "Insufficient TPUs") while `cw-rno2a` had ~239 free H100s.
- **Superset tokenizer** (crops/ccoord vocab, 3849 tokens, `<retract>` at 3848).
  Verified: 0 id mismatches against exp120's 2845 tokens, so this is a +1004
  embedding resize, not a remap; and 200 real backtracking documents tokenize
  with exact token counts, no UNKs, exact round-trip.
- **Unmasked loss + `pack=True`**, exactly as #117 — contacts-v1 has no
  `<distance>` statements to mask. This matters: marin 0.2.57 removed
  `DatasetComponent.loss_weight_fn`, so a masked run would require pre-packing
  a `PrebuiltLmDatasetFormat` cache. Unmasked keeps us on the simple path.
- **Caches build on the training workers** (`auto_build_caches=True`) straight
  from the S3 parquet, so there is no separate tokenize step to route through
  GCS.

Prerequisites (see `README.md`):
  1. the 50:50 mix staged to `EXP160_CORPUS` (S3) — `build_mix.py`
  2. exp120's **Levanter** checkpoint staged to `EXP160_INIT` (S3), 16.4 GiB —
     CoreWeave cannot read GCS
  3. the superset tokenizer published as a HF repo id

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    KUBECONFIG=~/.kube/coreweave-iris-rno2a \\
    /home/bizon/git/marin-freshiris/.venv/bin/iris --cluster=cw-rno2a job run \\
        --no-wait --priority batch --enable-extra-resources \\
        --cpu=2 --memory=8GB --disk=32GB \\
        -e WANDB_API_KEY "$WK" -- python -m dispatch_train
"""

from __future__ import annotations

import dataclasses
import os

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

# iris PriorityBand: PRIORITY_BAND_BATCH = 3 (job.proto).
IRIS_PRIORITY_BAND_BATCH = 3

S3_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp160_backtracking_training"
DEFAULT_CORPUS = f"{S3_PREFIX}/corpus"
DEFAULT_INIT = f"{S3_PREFIX}/init/exp120-step-1005"
DEFAULT_OUTPUT = f"{S3_PREFIX}/runs"

# Superset tokenizer (crops/ccoord vocab incl. <retract>). Bare repo id — the
# training tokenizer-load path does not split `@rev` (exp85's note).
DEFAULT_TOKENIZER = os.environ.get(
    "EXP160_TOKENIZER", "timodonnell/contacts-v1-backtracking-tokenizer"
)

# 1.5B Qwen3 (exp44 dims + Llama3 rope) — the architecture exp75/exp117/exp120
# all use. Kept here as data so the worker needs no experiment imports.
MODEL_KWARGS = dict(
    seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_layers=24,
    num_heads=32,
    num_kv_heads=8,
)


def build_worker_command(args: dict) -> str:
    """The training entrypoint, run under `uv run` inside the workspace venv."""
    return (
        "uv run python train_backtracking.py "
        f"--corpus {args['corpus']} "
        f"--init {args['init']} "
        f"--output {args['output']} "
        f"--tokenizer {args['tokenizer']} "
        f"--run-name {args['run_name']} "
        f"--learning-rate {args['learning_rate']} "
        f"--weight-decay {args['weight_decay']} "
        f"--train-batch-size {args['train_batch_size']} "
        f"--num-train-steps {args['num_train_steps']} "
        f"--warmup {args['warmup']}"
    )


def build_request(args: dict) -> JobRequest:
    """One multi-GPU training job at batch priority."""
    resources = ResourceConfig.with_gpu(
        "H100",
        count=args["gpus_per_node"],
        replicas=args["nodes"],
        cpu=64,
        ram="512g",
        # CoreWeave pods default to 5Gi ephemeral; the 16.4 GiB init checkpoint
        # plus caches need far more.
        disk="512g",
    )
    environment = create_environment(
        env_vars={
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "TOKENIZERS_PARALLELISM": "false",
            "UV_LINK_MODE": "copy",
        },
        extras=["gpu"],
    )
    return JobRequest(
        name=args["run_name"],
        entrypoint=Entrypoint.from_binary("bash", ["-lc", build_worker_command(args)]),
        resources=resources,
        environment=environment,
        replicas=args["nodes"],
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=10,
        # Separate field defaulting to 0 — without it a worker dies on its first
        # failure. GPU reclamation on this cluster arrives as a SIGTERM recorded
        # as a *failure*, not a preemption (see #159).
        max_task_failures=20,
        max_retries_preemption=100,
    )


def main() -> None:
    assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
        "This fray build lacks JobRequest.priority; batch-band dispatch needs "
        "the 0.2.x.dev fray line."
    )
    args = {
        "corpus": os.environ.get("EXP160_CORPUS", DEFAULT_CORPUS),
        "init": os.environ.get("EXP160_INIT", DEFAULT_INIT),
        "output": os.environ.get("EXP160_OUTPUT", DEFAULT_OUTPUT),
        "tokenizer": DEFAULT_TOKENIZER,
        "run_name": os.environ.get("EXP160_RUN_NAME", "exp160-cv1-1_5b-bt50-lr3e-4-e1-cos"),
        # exp120's continue-train LR (3e-4, 1-epoch cosine) rather than #117's
        # 3.16e-3 pretraining LR — this is a fine-tune from a converged model.
        "learning_rate": float(os.environ.get("EXP160_LR", "3e-4")),
        "weight_decay": float(os.environ.get("EXP160_WD", "0.2")),
        "train_batch_size": int(os.environ.get("EXP160_BATCH", "128")),
        "num_train_steps": int(os.environ.get("EXP160_STEPS", "0")) or None,
        "warmup": float(os.environ.get("EXP160_WARMUP", "0.1")),
        "nodes": int(os.environ.get("EXP160_NODES", "1")),
        "gpus_per_node": int(os.environ.get("EXP160_GPUS", "8")),
    }
    if args["num_train_steps"] is None:
        # ~2.16B tokens over the 2.05M-document mix; one epoch.
        tokens = int(os.environ.get("EXP160_TOKENS", "2160000000"))
        args["num_train_steps"] = max(
            1, tokens // (args["train_batch_size"] * MODEL_KWARGS["seq_len"])
        )
    print(f"dispatching training: {args}", flush=True)

    job = current_client().submit(build_request(args))
    print(f"submitted -> {job}", flush=True)
    # The driver must not exit: child jobs are its children and iris finalizes
    # them if it goes away.
    job.wait(raise_on_failure=True)
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
