# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared iris submission for exp208 — issue #208.

THIS DIRECTORY IS THE IRIS WORKSPACE. ``iris job run`` bundles the CWD and runs
``uv sync`` in it on the pod, so submitting from here means the pod resolves
exp208's own pinned manifest: marin 0.2.76 (the last release with ``marin.rl``)
plus the vLLM TPU fork at the SHA marin used at that release. Nothing depends on
a marin source checkout, which is what broke on 2026-08-07 when marin main
deleted the ``vllm`` extra and ``marin.rl`` while iris kept rejecting clients
more than 14 days old.

Two consequences worth knowing:

* The iris CLI comes from this directory's own venv, and the cluster YAML ships
  inside the ``marin-iris`` wheel, so ``--cluster=marin`` resolves with no
  external path.
* The bundle is built from ``git ls-files``, so **uncommitted changes are not
  uploaded**. :func:`check_clean` refuses to submit a dirty tree rather than let
  a job silently run last commit's code.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IRIS = str(HERE / ".venv/bin/iris")
CLUSTER = "marin"


def check_clean(paths: tuple[str, ...] = (".",)) -> None:
    """Refuse to submit if tracked files here differ from HEAD.

    The workspace bundle comes from ``git ls-files``, so an edited-but-uncommitted
    worker would be uploaded in its committed form — the job runs, succeeds, and
    measures the wrong code.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=HERE, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = [line for line in out.splitlines() if not line.startswith("??")]
    if dirty:
        listing = "\n  ".join(dirty)
        raise SystemExit(
            "refusing to submit with uncommitted changes — the iris bundle is built from "
            f"`git ls-files`, so the pod would run HEAD, not your edits:\n  {listing}"
        )


def submit(
    *,
    job_name: str,
    command: list[str],
    extras: tuple[str, ...] = (),
    tpu: str | None = None,
    cpu: int = 8,
    memory: str = "64GB",
    disk: str = "64GB",
    zone: str | None = None,
    region: str | None = None,
    priority: str = "interactive",
    max_retries: int = 2,
    env: dict[str, str] | None = None,
    raw: bool = False,
    dry_run: bool = False,
) -> str:
    """Submit one iris job whose workspace is this directory.

    Args:
        command: argv to run on the pod, e.g. ``["python", "phase1_parity.py", ...]``.
            Prefixed with ``uv run --no-sync`` so it uses the synced venv.
        raw: run ``command`` as-is, without the ``uv run`` prefix. Needed for a
            ``bash -lc`` bootstrap: ``-l`` sources profile files that can rewrite
            PATH, so the shell must call ``uv`` itself rather than inherit an
            activated venv that a login shell may have shadowed.
        extras: uv extras to sync on the pod. Rollout/TPU work wants
            ``("tpu", "vllm")``; CPU work wants ``("cpu",)``.
        tpu: TPU variant, e.g. ``"v5p-8"``. Omit for a CPU job.
        priority: ``interactive`` on the marin v5p pool — it is
            interactive-dominated and a batch job there never schedules. The
            always-batch rule is CoreWeave's, and does not carry over.
        zone: Pin only when capacity has been measured; exp163 lost three jobs to
            speculative zone-pinning. Prefer ``region``.
    """
    if tpu and disk and int(disk.rstrip("GB")) > 100:
        raise ValueError("a v5p-8 VM offers 100 GiB ephemeral; a larger ask is rejected, not queued")

    argv = [
        IRIS, f"--cluster={CLUSTER}", "job", "run",
        "--job-name", job_name, "--no-wait", "--enable-extra-resources",
        "--priority", priority,
        "--cpu", str(cpu), "--memory", memory, "--disk", disk,
        "--max-retries", str(max_retries),
    ]
    if tpu:
        argv += ["--tpu", tpu]
    if zone:
        argv += ["--zone", zone]
    elif region:
        argv += ["--region", region]
    for extra in extras:
        argv += ["--extra", extra]
    for key, value in (env or {}).items():
        argv += ["-e", key, value]
    argv += ["--", *command] if raw else ["--", "uv", "run", "--no-sync", *command]

    if dry_run:
        print(f"[exp208] DRY RUN {job_name}\n  " + " ".join(argv))
        return job_name

    subprocess.run(argv, cwd=HERE, check=True)
    print(f"[exp208] submitted {job_name}")
    print(f"  monitor: {IRIS} --cluster={CLUSTER} job summary /bizon/{job_name}")
    return job_name


def main_guard() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit("exp208 targets python 3.12")


__all__ = ["CLUSTER", "HERE", "IRIS", "check_clean", "submit"]
