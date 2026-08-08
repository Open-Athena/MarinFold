# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp201 Phase 1b launcher: the statement-head-mask arm, its control, and the LR sweep.

Submits one Fray ``JobRequest`` per (arm, learning rate) to the marin **v5p**
pool in us-east5, co-located with #150's token caches. Modelled on exp163's
``dispatch_refine_train.py`` and exp108's ``dispatch_train.py``: we build the
``TrainLmOnPodConfig`` ourselves and submit it, rather than letting the marin
executor dispatch child jobs (it submits them with no priority band).

**Priority band on TPU is INTERACTIVE, deliberately.** The CoreWeave rule
("always batch", #108) is CoreWeave-specific. The shared v5p pool is dominated
by other people's interactive jobs, so a batch-band TPU gang yields indefinitely
on "Insufficient TPUs (need N, available 0)" — the same trap exp163 documented.

**The plan (#201).** Masking changes the gradient-noise scale, so #117's tuned
3.1623e-3 may no longer be optimal and running only there risks a false negative.
So: a 2-epoch LR mini-sweep of the masked arm at {1x, 2x, 3.16x} of #117's LR
alongside a 2-epoch control at 1x, then the winner extended. The control is
re-run under *this* harness rather than read off #150's curve, so the harness is
not a confound.

Two sequential phases, selected with ``--phase``::

    # 1. the sweep: 3 masked LRs + 1 control, 2 epochs each
    uv run python dispatch_arms.py --phase sweep --dry-run
    uv run python dispatch_arms.py --phase sweep

    # 2. once a winner is picked, extend both arms to 4 epochs
    uv run python dispatch_arms.py --phase extend --lr 6.3246e-3

**Submit from a fresh marin client.** iris rejects a marin-iris client older than
14 days; the frozen ``marin-*-latest`` wheels are always rejected. Submit from an
editable marin checkout (``/home/bizon/git/marin``) or a recently-synced venv.

``--dry-run`` builds every JobRequest and prints it without submitting, which
validates the whole config assembly with no cluster time. Do that first.
"""

import argparse
import dataclasses
import os
import sys

from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm

# Fail loudly on a stale fray. The frozen `marin-*-latest` wheels predate both
# `priority` (batch-band dispatch) and `processes_per_task`, and iris separately
# rejects a marin-iris client older than 14 days -- so a submit from the wrong
# venv fails late and confusingly. Submit from the editable marin checkout.
_JOB_REQUEST_FIELDS = {f.name for f in dataclasses.fields(JobRequest)}
_MISSING = {"priority", "processes_per_task"} - _JOB_REQUEST_FIELDS
assert not _MISSING, (
    f"this fray build lacks JobRequest{sorted(_MISSING)}; submit from a recent "
    "marin checkout (e.g. /home/bizon/git/marin/.venv)"
)

from exp201_arm_common import (
    BASE_LR,
    OUTPUT_PREFIX,
    PROTEIN_RESOURCES,
    TRAIN_CACHE_DIR,
    VAL_CACHE_DIR,
    build_on_pod_config,
    evals_per_epoch_steps,
    steps_for_epochs,
    steps_per_epoch,
)

# The mini-sweep: multiples of #117's tuned peak LR. Masking removes 24 % of the
# supervised slots, which lowers gradient noise, and lower noise usually admits a
# larger step -- so the sweep goes up from #117's value, not down.
LR_MULTIPLIERS = (1.0, 2.0, 3.1623)

SWEEP_EPOCHS = 2
EXTEND_EPOCHS = 4


def run_name(arm: str, learning_rate: float, epochs: int) -> str:
    """Fray/iris-safe name (alnum + hyphens) that reads as the experiment point."""
    lr_tag = f"{learning_rate:.4g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"exp201-{arm}-lr{lr_tag}-e{epochs}"


def build_request(
    *,
    arm: str,
    learning_rate: float,
    epochs: int,
    env_vars: dict[str, str],
    max_retries_failure: int,
    max_steps: int | None = None,
) -> JobRequest:
    """One arm at one learning rate, as a self-contained JobRequest."""
    name = run_name(arm, learning_rate, epochs)
    num_train_steps = steps_for_epochs(epochs)
    if max_steps is not None:
        num_train_steps = min(num_train_steps, max_steps)
        name = f"{name}-smoke{num_train_steps}"
    on_pod_config = build_on_pod_config(
        arm=arm,
        run_name=name,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        output_path=f"{OUTPUT_PREFIX}/{name}",
        resources=PROTEIN_RESOURCES,
        env_vars=env_vars,
        steps_per_eval=evals_per_epoch_steps(),
        steps_per_checkpoint=steps_per_epoch(),
        tags=(f"epochs-{epochs}",),
    )
    environment = create_environment(
        # resolve_training_env: hardware defaults + GIT_COMMIT + JAX compile cache.
        env_vars=resolve_training_env(base_env=dict(env_vars), resources=PROTEIN_RESOURCES),
        extras=extras_for_resources(PROTEIN_RESOURCES),  # TpuConfig -> ["tpu"]
    )
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[on_pod_config]),
        resources=PROTEIN_RESOURCES,
        environment=environment,
        replicas=1,
        # Interactive, NOT batch -- see the module docstring.
        priority=int(os.environ.get("EXP201_PRIORITY", "0")),
        processes_per_task=1,
        max_retries_failure=max_retries_failure,
    )


def planned_points(phase: str, lr: float | None) -> list[tuple[str, float, int]]:
    """The (arm, learning_rate, epochs) points a phase submits."""
    if phase == "smoke":
        # One masked job at #117's LR, step-capped by --max-steps. Nothing in
        # this path has ever run on a pod: the config has to deserialize (which
        # means marinfold_models must import and register its draccus plugin),
        # the #150 caches have to be readable with auto_build_caches off, and the
        # mask has to actually engage. Watch `train/kept_slot_fraction` -- it
        # should sit near 0.763, and a value of 1.0 means the mask silently did
        # nothing. Cheaper to learn that here than across four full gangs.
        return [("masked", BASE_LR, SWEEP_EPOCHS)]
    if phase == "sweep":
        points = [("masked", BASE_LR * m, SWEEP_EPOCHS) for m in LR_MULTIPLIERS]
        # One control at #117's own LR: the reference both the masked arm and
        # #150's published curve are read against.
        points.append(("control", BASE_LR, SWEEP_EPOCHS))
        return points
    if phase == "extend":
        if lr is None:
            raise SystemExit("--phase extend needs --lr (the winning masked LR)")
        return [("masked", lr, EXTEND_EPOCHS), ("control", BASE_LR, EXTEND_EPOCHS)]
    raise SystemExit(f"unknown phase {phase!r}")


def training_env(*, dry_run: bool) -> dict[str, str]:
    """W&B routing, forwarded explicitly — the pod does NOT inherit the launch shell.

    Export ``WANDB_API_KEY`` before a real submit. A dry run substitutes a
    placeholder so config assembly can be validated without a credential
    (marin's ``resolve_training_env`` refuses to build an env without one).
    """
    env = {"WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold"}
    key = os.environ.get("WANDB_API_KEY")
    if key:
        env["WANDB_API_KEY"] = key
    elif dry_run:
        env["WANDB_API_KEY"] = "dry-run-placeholder"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "sweep", "extend"), required=True)
    parser.add_argument("--lr", type=float, default=None,
                        help="winning masked LR, for --phase extend")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and print the JobRequests without submitting")
    parser.add_argument("--max-retries-failure", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap num_train_steps (smoke runs); the LR schedule "
                             "still spans the full epoch count, so a capped run "
                             "is NOT a short training run, only a plumbing check")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set; the pod cannot log to W&B without it")
    env_vars = training_env(dry_run=args.dry_run)

    points = planned_points(args.phase, args.lr)
    print(f"[exp201] phase={args.phase}  points={len(points)}")
    print(f"[exp201] train cache : {TRAIN_CACHE_DIR}")
    print(f"[exp201] val cache   : {VAL_CACHE_DIR}")
    print(f"[exp201] steps/epoch : {steps_per_epoch():,}")

    requests = []
    for arm, learning_rate, epochs in points:
        request = build_request(
            arm=arm,
            learning_rate=learning_rate,
            epochs=epochs,
            env_vars=env_vars,
            max_retries_failure=args.max_retries_failure,
            max_steps=args.max_steps,
        )
        requests.append(request)
        steps = min(steps_for_epochs(epochs), args.max_steps or steps_for_epochs(epochs))
        print(
            f"  {request.name:<34} arm={arm:<7} lr={learning_rate:.4e} "
            f"epochs={epochs} steps={steps:,} model={_model_name(arm)}"
        )

    if args.dry_run:
        print("\n[exp201] DRY RUN -- JobRequests built, not submitting.")
        return

    from fray import current_client  # local: only needed on the submit path

    client = current_client()
    jobs = [client.submit(request) for request in requests]
    print(f"\n[exp201] submitted {len(jobs)} jobs:")
    for job in jobs:
        print(f"  {job.id}")
    print("\nMonitor with: uv run iris job list | grep exp201")


def _model_name(arm: str) -> str:
    from exp201_arm_common import model_config

    return type(model_config(arm)).__name__


if __name__ == "__main__":
    sys.exit(main())
