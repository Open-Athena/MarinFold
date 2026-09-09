"""Capture the 18-hour milestone and run its bounded retention/diversity audit."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from launch import storage_filesystem

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    """Run a checked analysis step using the experiment's locked environment."""
    subprocess.run(["uv", "run", "python", script, *args], cwd=HERE, check=True)


def main() -> None:
    root = HERE / "data/scale-20260909"
    report = root / "review-18h"
    report.mkdir(parents=True, exist_ok=True)
    manifest = str(root / "manifest.json")
    run("scale_snapshot.py", "--manifest", manifest, "--output", str(report))
    run(
        "budget_snapshot.py",
        "--cluster",
        "cw-us-east-02a",
        "--prefix",
        "/bizon/exp278-scale-v1",
        "--output",
        str(report),
        "--scale",
    )
    with (report / "iris-jobs.txt").open("w") as handle:
        subprocess.run(
            [
                "uv",
                "run",
                "iris",
                "--cluster",
                "cw-us-east-02a",
                "job",
                "list",
                "--prefix",
                "/bizon/exp278-scale-v1",
                "--limit",
                "2000",
            ],
            cwd=HERE,
            stdout=handle,
            check=True,
        )
    run(
        "scale_audit.py",
        "--manifest",
        manifest,
        "--work",
        "/data/exp278/scale-review-18h",
        "--report",
        str(report),
        "--per-stratum",
        "64",
        "--threads",
        "8",
    )
    snapshot = json.loads((report / "latest.json").read_text())
    budget = json.loads((report / "iris-resource-time.json").read_text())
    retention = json.loads((report / "retention.json").read_text())
    metrics = {
        f"scale/{key}": snapshot[key]
        for key in ("generated", "refolded", "quality_pass", "quality_retention")
    }
    metrics.update(
        {
            "scale/h100_hours": budget["h100_hours"],
            "scale/audit_candidates": retention["candidates"],
            "scale/audit_retained_fraction": retention["retained_fraction"],
            "scale/review_action": "continue while reviewing",
        }
    )
    (report / "wandb-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    jobs = [
        json.loads(path.read_text())["job_id"]
        for path in sorted(root.glob("worker-*-submission.json"))
    ]
    run(
        "record_metrics.py",
        "--scale",
        "--run-id",
        "exp278-proteina-scale-20260909",
        "--metrics",
        str(report / "wandb-metrics.json"),
        *(arg for job in jobs for arg in ("--job", job)),
    )
    done = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "continued_generation": True,
        "metrics": metrics,
    }
    (report / "complete.json").write_text(json.dumps(done, indent=2) + "\n")
    fs = storage_filesystem("cw-us-east-02a")
    for path in report.iterdir():
        if path.is_file():
            fs.pipe_file(
                f"marin-us-east-02a/MarinFold/exp278-proteina/scale-20260909/reports/review-18h/{path.name}",
                path.read_bytes(),
            )
    body = (
        "🤖 **18-hour scale review** — generation continues while we review.\n\n"
        f"Committed snapshot: {snapshot['generated']:,} backbones, "
        f"{snapshot['sequences_saved']:,} saved designed sequences, "
        f"{snapshot['refolded']:,} refolds, and {snapshot['quality_pass']:,} "
        "quality-passing provisional documents.\n\n"
        f"Iris task-attempt time: {budget['h100_hours']:,.1f} H100-hours "
        "including setup and retries (not a provider invoice).\n\n"
        f"Stratified audit: {retention['candidates']:,} candidates, "
        f"{retention['after_decontamination']:,} after quality/reference exclusion, "
        f"{retention['after_cluster_cap']:,} after capping clusters within the sample. "
        "This is sampled retention, not the final corpus retention or a global "
        "diversity cap. Compare per-length/arm tables and accumulation results "
        "before projecting a final yield.\n\n"
        "[W&B metrics](https://wandb.ai/open-athena/MarinFold/runs/exp278-proteina-scale-20260909). "
        "All detailed reports are saved under "
        "`s3://marin-us-east-02a/MarinFold/exp278-proteina/scale-20260909/reports/review-18h/`."
    )
    subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            "repos/Open-Athena/MarinFold/issues/278/comments",
            "--input",
            "-",
        ],
        input=json.dumps({"body": body}),
        text=True,
        check=True,
        capture_output=True,
    )
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
