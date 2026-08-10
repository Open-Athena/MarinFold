# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stop exp200 RL arms once training has actually finished — issue #200.

WHY THIS IS NEEDED. marin's RL run cannot terminate itself. The completion
handshake has no safety on either side:

* the trainer calls ``runtime.run_state.mark_completed.remote().result()``
  (``orchestration.py:308``) with **no timeout**, so it blocks forever if that RPC
  does not land; and
* the rollout worker polls ``get_snapshot.remote().result(timeout=5.0)`` inside
  ``except Exception: pass  # best-effort``, so a persistent failure is
  indistinguishable from "run still going" and it loops forever.

Either way the coordinator's ``train_job.wait()`` never returns and nothing reaps
the rollout workers. Observed directly: a nano run logged all 10 training steps
and closed its W&B run, then sat for another 50 minutes generating rollouts until
killed by hand.

W&B is the channel that does work — it is how training completion was established
in the first place — so completion is detected there and the iris job is stopped
from outside. Stopping the driver cascades to the coordinators and their children.

    uv run python reap.py --job exp200-rl-sweep \\
        --runs plm-exp200-rl-cv1-1_5b-lr1em06-s8,plm-exp200-rl-cv1-1_5b-lr3em06-s8 \\
        --steps 150
"""

import argparse
import netrc
import os
import subprocess
import time

from _submit import CLUSTER, IRIS


def wandb_api():
    if "WANDB_API_KEY" not in os.environ:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if not auth or not auth[2]:
            raise SystemExit("no W&B credentials: set WANDB_API_KEY or log in")
        os.environ["WANDB_API_KEY"] = auth[2]
    import wandb

    return wandb.Api()


def max_step(api, project: str, run_name: str) -> tuple[int | None, str | None]:
    """Highest logged step for a run, and its state. (None, None) if absent."""
    runs = list(api.runs(project, filters={"display_name": run_name}, order="-created_at"))
    if not runs:
        return None, None
    run = runs[0]
    # summary["_step"] is W&B's own counter and is enough for a completion test:
    # the trainer logs once per training step through levanter's tracker.
    return int(run.summary.get("_step", -1)), run.state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="iris job to stop (the driver)")
    ap.add_argument("--runs", required=True, help="comma-separated W&B run names, one per arm")
    ap.add_argument("--steps", type=int, required=True, help="num_train_steps; done at steps-1")
    ap.add_argument("--project", default="open-athena/MarinFold")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--grace", type=int, default=420,
                    help="seconds to wait after the last step before stopping, so the "
                         "trainer can finish writing a checkpoint")
    ap.add_argument("--timeout-h", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    api = wandb_api()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    target = a.steps - 1
    deadline = time.time() + a.timeout_h * 3600
    done_since: float | None = None

    while time.time() < deadline:
        status = {}
        for name in runs:
            step, state = max_step(api, a.project, name)
            status[name] = (step, state)
        line = "  ".join(f"{n.split('-lr')[-1]}={s}/{target}({st})" for n, (s, st) in status.items())
        complete = [n for n, (s, st) in status.items()
                    if (s is not None and s >= target) or st in ("finished", "failed", "crashed")]
        print(f"[reap] {len(complete)}/{len(runs)} complete  {line}", flush=True)

        if len(complete) == len(runs):
            if done_since is None:
                done_since = time.time()
                print(f"[reap] all arms complete; waiting {a.grace}s for final checkpoints", flush=True)
            elif time.time() - done_since >= a.grace:
                if a.dry_run:
                    print(f"[reap] DRY RUN would stop {a.job}")
                    return 0
                subprocess.run([IRIS, f"--cluster={CLUSTER}", "job", "stop", f"/bizon/{a.job}"], check=False)
                print(f"[reap] stopped {a.job}")
                return 0
        else:
            done_since = None
        time.sleep(a.poll)

    print(f"[reap] TIMEOUT after {a.timeout_h}h; leaving {a.job} alone rather than killing "
          "a run that may still be training")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
