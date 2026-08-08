# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Phase 1 parity gate to a marin v5p-8 — issue #200.

Copied from exp169's ``dispatch_eval_tpu.py``, which is the settled shape for
running vLLM on the marin v5p pool. The pieces that matter:

* **The submitting CLIENT and the bundled WORKSPACE are two different checkouts.**
  ``iris job run`` bundles the CWD as the job's workspace and runs ``uv sync`` in
  it on the pod, and ``--extra vllm --extra tpu`` are *marin's* extras, so the
  workspace has to be a marin checkout. But iris also rejects a client build more
  than 14 days old, and on 2026-08-07 marin main deleted BOTH the ``vllm`` extra
  and ``marin.rl`` itself (``e7ef104402``, "Externalize the TPU vLLM stack and
  drop legacy marin.rl"). So a checkout new enough to submit no longer contains
  what the job needs. The binary therefore comes from the fresh checkout and the
  bundle from the last RL-capable one, and :func:`check_workspace` asserts the
  bundle still has both before submitting — a missing extra otherwise surfaces
  30 seconds into a pod build, three retries deep.

  This split is temporary by nature. exp200 pins ``marin-core`` 0.2.57 from PyPI,
  which still ships ``marin.rl``; the durable fix is to make this experiment dir
  a self-contained iris workspace the way exp166 is, declaring its own pinned
  vLLM fork so upstream churn cannot reach it.
* **exp200's modules are base64-inlined** into the bootstrap, since the bundle is
  marin rather than this experiment dir.
* **Interactive band, not batch.** The v5p pool is interactive-dominated and a
  batch job there never schedules. This is the opposite of the CoreWeave rule.
* **Region, not zone**, unless capacity says otherwise: exp163 lost three jobs to
  zone-pinning. The data lives in us-east5, so that is the default.
* ``--disk 64GB``: a v5p-8 VM offers 100 GiB ephemeral and asking for more is
  rejected as unschedulable rather than queued.

Usage::

    uv run python dispatch_parity.py --limit 100 --n-generations 4
    EXP200_DRY_RUN=1 uv run python dispatch_parity.py    # print, do not submit
"""

import argparse
import base64
import os
import subprocess
from pathlib import Path

# Fresh checkout: supplies the iris CLI binary, which must be < 14 days old.
MARIN_CLIENT = Path(os.environ.get("MARIN_CLIENT_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN_CLIENT / ".venv/bin/iris"))
# Bundled workspace: the last marin that still has `marin.rl` and the `vllm` extra.
MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin"))
# `--cluster=marin` resolves the cluster YAML from search paths relative to the
# CWD, which is the BUNDLE checkout — and the new client rejects the old
# checkout's config ("cluster name is required"). Name the fresh config outright.
CLUSTER_CONFIG = Path(
    os.environ.get("IRIS_CLUSTER_CONFIG", str(MARIN_CLIENT / "lib/iris/config/marin.yaml"))
)

EXP163 = "gs://marin-us-east5/MarinFold/exp163"
MODEL = os.environ.get("EXP200_MODEL", f"{EXP163}/tpu/tpuF-bf16/step-404")
TARGETS = os.environ.get("EXP200_TARGETS", f"{EXP163}/eval554/targets.parquet")
PROMPTS = os.environ.get("EXP200_PROMPTS", f"{EXP163}/eval554/prompts")
OUT = os.environ.get("EXP200_OUT", "gs://marin-us-east5/MarinFold/exp200/phase1")

HERE = Path(__file__).resolve().parent
# Everything phase1_parity.py imports from this experiment.
MODULES = ("contact_rewards.py", "contacts_env.py", "_exp163_rollout_metrics.py", "phase1_parity.py")
WORKDIR = "/tmp/exp200"


def check_workspace() -> None:
    """Fail at submit time if the bundled marin cannot serve this job.

    marin main deleted the ``vllm`` extra and ``marin.rl`` on 2026-08-07. Without
    this check, either loss shows up as a pod-side ``uv sync`` error repeated once
    per retry, several minutes after submission.
    """
    pyproject = MARIN / "lib/marin/pyproject.toml"
    if not pyproject.exists():
        raise SystemExit(f"{pyproject} not found — MARIN_CHECKOUT does not look like a marin checkout")
    if "\nvllm = [" not in pyproject.read_text():
        raise SystemExit(
            f"{pyproject} defines no `vllm` extra. marin removed it in e7ef104402 (2026-08-07); "
            "point MARIN_CHECKOUT at an older checkout, or give this experiment its own "
            "pinned vLLM fork the way exp166 does."
        )
    if not (MARIN / "lib/marin/src/marin/rl/rl_losses.py").exists():
        raise SystemExit(
            f"{MARIN} has no marin.rl (deleted upstream in e7ef104402). exp200's loss and "
            "environment are built on it; point MARIN_CHECKOUT at an RL-capable checkout."
        )


def build_bootstrap(*, limit: int, n_generations: int, max_sections: int,
                    tensor_parallel_size: int, tag: str) -> str:
    writes = "\n".join(
        f"echo {base64.b64encode((HERE / name).read_bytes()).decode()} | base64 -d > {WORKDIR}/{name}"
        for name in MODULES
    )
    return f"""
set -euo pipefail
echo "[exp200-parity] host=$(hostname) limit={limit} gens={n_generations} cap={max_sections}"

mkdir -p {WORKDIR}
{writes}

# marin's synced venv already carries its vLLM TPU fork, jax, transformers,
# pyarrow, fsspec and gcsfs, plus marin.rl itself. Nothing else to install:
# exp200's modules import only from marin.rl, numpy, pyarrow and fsspec.
# Run from the bundle root (iris sets CWD there) so `uv run --no-sync` picks up
# the synced workspace venv. Python puts the SCRIPT's directory on sys.path, so
# phase1_parity.py finds its siblings in {WORKDIR} without any PYTHONPATH games.
exec uv run --no-sync python {WORKDIR}/phase1_parity.py \\
    --model {MODEL} \\
    --targets {TARGETS} \\
    --prompts {PROMPTS} \\
    --out {OUT} \\
    --limit {limit} \\
    --n-generations {n_generations} \\
    --max-sections {max_sections} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --tag {tag}
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="proteins to sample")
    ap.add_argument("--n-generations", type=int, default=4)
    ap.add_argument("--max-sections", type=int, default=0,
                    help="0 = uncapped, matching the published #163 numbers")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--tpu", default="v5p-8")
    ap.add_argument("--priority", default="interactive",
                    choices=["production", "interactive", "batch"])
    ap.add_argument("--region", default="us-east5", help="region, not zone (exp163 lesson)")
    ap.add_argument("--zone", default=None, help="only if capacity forces it")
    ap.add_argument("--job-name", default="exp200-phase1-parity")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    check_workspace()
    bootstrap = build_bootstrap(
        limit=a.limit, n_generations=a.n_generations,
        max_sections=a.max_sections, tensor_parallel_size=a.tensor_parallel_size,
        tag=a.job_name,
    )
    placement = ["--zone", a.zone] if a.zone else ["--region", a.region]
    command = [
        IRIS, "--config", str(CLUSTER_CONFIG), "job", "run",
        "--job-name", a.job_name, "--no-wait", "--enable-extra-resources",
        "--priority", a.priority, *placement, "--tpu", a.tpu,
        "--extra", "vllm", "--extra", "tpu",
        "--cpu", "8", "--memory", "64GB", "--disk", "64GB",
        "--max-retries", "2",
        "--", "bash", "-lc", bootstrap,
    ]
    if a.dry_run or os.environ.get("EXP200_DRY_RUN"):
        print(f"[exp200-parity] DRY RUN {a.job_name}")
        print(" ".join(command[:-1]))
        print(f"--- bootstrap ({len(bootstrap)} chars) ---")
        print(bootstrap[:900])
        return 0
    subprocess.run(command, cwd=MARIN, check=True)
    print(f"[exp200-parity] submitted {a.job_name}")
    print(f"  monitor: {IRIS} --cluster=marin job list | grep {a.job_name}")
    print(f"  results: {OUT}/{a.job_name}/parity_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
