# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Launch exp230's fine-tune of exp199 on the multi-draft corpus.

Adapted from #163's ``dispatch_refine_train.py``.  This runs as a small CPU
driver job on marin which submits the TPU training gang as its child, so **the
driver must outlive the gang** -- iris kills children when a parent exits.

Priority: **interactive**, not batch.  The v5p pool is dominated by other
people's interactive jobs, so a batch-band TPU job yields indefinitely on
"Insufficient TPUs" and registers no autoscaler demand.  This is the opposite of
the CoreWeave rule and deliberately so; see ``dispatch_rollouts.py``.

Env knobs:

  EXP230_LRS               comma list of peak LRs (default "1e-4"). #163 swept
                           {1e-4, 3e-4}: 3e-4 fit the multi-draft objective 0.3%
                           better and degraded base-task retention by +7.4% bpb,
                           so 1e-4 is the default rather than a guess.
  EXP230_EPOCHS            default 2
  EXP230_STEPS_PER_EPOCH   REQUIRED unless EXP230_TRAIN_SEQUENCES is set;
                           ``tokenize_corpus.py`` PRINTS it.
  EXP230_TRAIN_SEQUENCES   alternative: packed sequences; steps/epoch =
                           ceil(sequences / batch)
  EXP230_MAX_STEPS         cap for a smoke run
  EXP230_STEPS_PER_EVAL    default 250, matching the checkpoint cadence
  EXP230_TPU_TYPE          default v5p-16
  EXP230_CORPUS/VAL/INIT_HF/GCS_PREFIX   see train_common.py
  WANDB_API_KEY            forwarded into the pod (iris tasks do NOT inherit the
                           submitting shell)

Warm-start verification uses **bpb, not per-token loss** -- levanter's
bytes-per-token bookkeeping is packing- and version-dependent, so a loss target
copied from another harness is not comparable.  There is also no step-0 eval on
this path; for an anchor run EXP230_STEPS_PER_EVAL=1 EXP230_MAX_STEPS=1.

    set -a; source ~/marin.env; set +a
    WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    <fresh-iris> --cluster=marin job run --no-wait --enable-extra-resources \
        --cpu=2 --memory=6GB --disk=16GB -e WANDB_API_KEY "$WK" \
        -e EXP230_STEPS_PER_EPOCH <N> -- python -m dispatch_train

    EXP230_DRY_RUN=1 EXP230_STEPS_PER_EPOCH=100 python -m dispatch_train
"""
from __future__ import annotations

import dataclasses
import math
import os

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm

from train_common import (
    CORPUS_GLOB,
    GCS_PREFIX,
    INIT_FROM_HF,
    PROTEIN_RESOURCES_TPU,
    TRAIN_BATCH,
    VAL_GLOB,
    build_on_pod_config,
)

#: iris PriorityBand (iris/rpc/job.proto). Interactive on the marin TPU pool.
IRIS_PRIORITY_BAND_INTERACTIVE = 0

# Fail loudly on the frozen 0.99.dev fray, whose JobRequest has no `priority`
# field -- the band would be silently dropped.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "this fray build lacks JobRequest.priority; use the 0.2.x.dev line"
)

# Runtime-tuning env forwarded from the driver onto the gang: iris tasks do not
# inherit the submitter's shell, and the gang runs in a SEPARATE pod from this
# driver. JAX_PLATFORMS is excluded so the CPU driver's value cannot leak onto
# the TPU gang.
_FORWARD_PREFIXES = ("XLA_", "LIBTPU_INIT_ARGS", "JAX_")
_FORWARD_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if k.startswith(_FORWARD_PREFIXES) and k not in _FORWARD_EXCLUDE}


def _steps_per_epoch() -> int:
    steps = os.environ.get("EXP230_STEPS_PER_EPOCH")
    if steps:
        return int(steps)
    seqs = os.environ.get("EXP230_TRAIN_SEQUENCES")
    if seqs:
        return math.ceil(int(seqs) / TRAIN_BATCH)
    raise SystemExit(
        "EXP230_STEPS_PER_EPOCH is required (or EXP230_TRAIN_SEQUENCES). "
        "tokenize_corpus.py prints 'STEPS_PER_EPOCH at batch 128 = <N>'."
    )


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def run_name_for(lr: float, epochs: int) -> str:
    suffix = os.environ.get("EXP230_RUN_SUFFIX", "").strip().strip("-")
    # W&B-safe: alphanumerics and hyphens, under 64 chars.
    return (f"plm-exp230-cv1-multi-1_5b-lr{_lr_tag(lr)}-e{epochs}-cos"
            + (f"-{suffix}" if suffix else ""))


def build_request(*, run_name: str, learning_rate: float, num_train_steps: int,
                  env_vars: dict[str, str], steps_per_eval: int) -> JobRequest:
    env_vars = {**_forwarded_env(), **env_vars}
    resources = PROTEIN_RESOURCES_TPU
    cfg = build_on_pod_config(
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        output_path=f"{GCS_PREFIX}/checkpoints/{run_name}",
        resources=resources,
        env_vars=env_vars,
        steps_per_eval=steps_per_eval,
        steps_per_checkpoint=250,
        hf_save_steps=250,
        tags=("exp230", "contacts-v1-multi", "exp199-warmstart"),
    )
    return JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[cfg]),
        resources=resources,
        environment=create_environment(
            env_vars=resolve_training_env(base_env=dict(env_vars), resources=resources),
            extras=extras_for_resources(resources),
        ),
        replicas=int(os.environ.get("EXP230_REPLICAS", "1")),
        priority=IRIS_PRIORITY_BAND_INTERACTIVE,
        processes_per_task=1,
        max_retries_failure=3,
    )


def main() -> None:
    lrs = [float(x) for x in os.environ.get("EXP230_LRS", "1e-4").split(",") if x.strip()]
    epochs = int(os.environ.get("EXP230_EPOCHS", "2"))
    spe = _steps_per_epoch()
    num_train_steps = spe * epochs
    if os.environ.get("EXP230_MAX_STEPS"):
        num_train_steps = min(num_train_steps, int(os.environ["EXP230_MAX_STEPS"]))
    steps_per_eval = int(os.environ.get("EXP230_STEPS_PER_EVAL", "250"))

    env_vars = {k: os.environ[k] for k in ("WANDB_API_KEY",) if k in os.environ}
    reqs = [build_request(run_name=run_name_for(lr, epochs), learning_rate=lr,
                          num_train_steps=num_train_steps, env_vars=env_vars,
                          steps_per_eval=steps_per_eval)
            for lr in lrs]

    print(f"[exp230] {len(reqs)} run(s), {spe} steps/epoch x {epochs} epochs "
          f"= {num_train_steps} steps, batch {TRAIN_BATCH}\n"
          f"         corpus={CORPUS_GLOB}\n         val={VAL_GLOB}\n"
          f"         warm start={INIT_FROM_HF}")
    for r in reqs:
        print(f"    {r.name}: priority={r.priority} resources={r.resources.device}")

    if os.environ.get("EXP230_DRY_RUN"):
        print("[exp230] DRY RUN -- JobRequests built, not submitting.")
        return

    client = current_client()
    jobs = [client.submit(r) for r in reqs]
    print(f"[exp230] submitted {len(jobs)} run(s); waiting (the driver must "
          "outlive its children or iris kills them)", flush=True)
    results = []
    for j in jobs:
        try:
            results.append(j.wait())
        except Exception as exc:  # noqa: BLE001 -- report every run, not just the first
            results.append(f"{type(exc).__name__}: {exc}")
    bad = [(r.name, s) for r, s in zip(reqs, results) if s != JobStatus.SUCCEEDED]
    print(f"[exp230] finished: {len(results) - len(bad)}/{len(results)} succeeded")
    for name, status in bad:
        print(f"  FAILED {name}: {status}")


if __name__ == "__main__":
    main()
