"""Append measured pilot milestones to W&B and the repository run history."""

import argparse
import json
import subprocess
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ID = "exp278-proteina-pilot-20260909"
RUN_NAME = "exp278-proteina-pilot-20260909"


def history_command(*args: str) -> None:
    """Run the canonical repository history writer in this locked environment."""
    subprocess.run(
        ["uv", "run", "python", str(ROOT / "scripts/history.py"), *args],
        cwd=HERE,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    run = wandb.init(
        project="MarinFold",
        entity="open-athena",
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        job_type="data-generation",
        config={
            "experiment": 278,
            "phase": "scale" if args.scale else "pilot",
            **({} if args.scale else {"pilot_gpu_hour_cap": 100}),
        },
    )
    existing = [
        path
        for path in (ROOT / "history/runs").glob("*exp278*")
        if f"run_id: {args.run_id}\n" in path.read_text()
    ]
    if not existing:
        history_command(
            "new",
            "--wandb-url",
            run.url,
            "--wandb-name",
            args.run_id,
            "--experiment",
            HERE.name,
            "--kind",
            "data",
            "--short",
            "Proteina scale generation: independent batch GPU workers; preserve all backbones and sequences"
            if args.scale
            else "Proteina structural diversity pilot: sampling, sequence design and refolding",
            "--iris-jobs",
            *args.job,
        )
    else:
        for job in args.job:
            history_command("add-iris-job", args.run_id, job)
    history_command("update-index")
    run.log(metrics)
    run.summary.update(metrics)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
