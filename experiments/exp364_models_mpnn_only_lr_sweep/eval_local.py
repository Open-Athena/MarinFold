"""Score one checkpoint on one H100, using worker disk and public result storage.

CoreWeave suspended writes to this account on October 21. The checkpoint and
HF export therefore live only on regional worker disk; model bytes never leave
CoreWeave. Only completed score parts, timings and provenance go to the public
HF data bucket. The validated rollout worker is unchanged.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pandas as pd

from experiments.exp364_models_mpnn_only_lr_sweep.eval_run import (
    DATA,
    HERE,
    MARINFOLD_REVISION,
    WORKER,
    WORKER_SHA256,
    markers_for,
    score_checkpoint,
)


def sync_results(directory: Path, remote: str, files: list[Path] | None = None) -> None:
    """Publish result files only; model/native directories are outside this root."""
    command = [
        "hf",
        "buckets",
        "sync",
        str(directory),
        remote,
        "--no-delete",
        "--quiet",
    ]
    if files is not None:
        for path in files:
            command.extend(["--include", str(path.relative_to(directory))])
    subprocess.run(command, check=True)


def completed_files(directory: Path) -> list[Path]:
    """Select only closed score parts whose completion marker has been written."""
    paths = []
    for marker_path in sorted(directory.glob("*/**/complete/*.json")):
        marker = json.loads(marker_path.read_text())
        paths.extend(
            [marker_path, Path(marker["score_uri"]), Path(marker["timing_uri"])]
        )
    for path in paths:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(
                f"Result marker points outside the output directory: {path}"
            )
    return paths


def run_worker(command: list[str], directory: Path, remote: str) -> None:
    """Publish completed groups while sampling, retaining partial results on error."""
    published: set[str] = set()
    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
    environment = {**os.environ, "VLLM_PORT": str(port)}
    with subprocess.Popen(command, env=environment) as process:
        while process.poll() is None:
            paths = completed_files(directory)
            current = {str(p) for p in paths}
            if current != published:
                sync_results(directory, remote, paths)
                published = current
            time.sleep(5)
        returncode = process.wait()
    sync_results(directory, remote)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    """Export locally, gate one protein, then score all 116 requested proteins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--vllm-python", required=True)
    args = parser.parse_args()
    plan = json.loads((DATA / "plan.json").read_text())
    checkpoint = next(c for c in plan["checkpoints"] if c["trial"] == args.trial)
    bundle = json.loads(Path("bundle_manifest.json").read_text())
    for name, digest in bundle.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundled source changed: {name}")
    if hashlib.sha256(WORKER.read_bytes()).hexdigest() != WORKER_SHA256:
        raise ValueError("The validated worker changed")
    scratch = Path("/tmp/exp364-contact-eval") / checkpoint["trial"]
    outputs = scratch / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    remote = plan["artifact_root"] + "/" + checkpoint["label"]
    public_fs, public_path = fsspec.core.url_to_fs(remote)
    if public_fs.exists(public_path):
        subprocess.run(
            ["hf", "buckets", "sync", remote, str(outputs), "--no-delete", "--quiet"],
            check=True,
        )
    (outputs / "plan.json").write_text(json.dumps(plan, indent=2))
    (outputs / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2))
    sync_results(outputs, remote)
    export_start = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp364_models_mpnn_only_lr_sweep.eval_export",
            "--plan",
            str(DATA / "plan.json"),
            "--trial",
            args.trial,
            "--scratch",
            str(scratch / "models"),
        ],
        check=True,
    )
    export_seconds = time.monotonic() - export_start
    export = json.loads((scratch / "models/export.json").read_text())
    (outputs / "export_manifest.json").write_text(json.dumps(export, indent=2))
    local_plan = {**plan, "root": str(outputs)}
    fs = fsspec.filesystem("file")
    truth = [
        json.loads(line)
        for line in (DATA / "ground_truth.jsonl").read_text().splitlines()
    ]
    for phase in ("smoke", "rollout"):
        command = [
            args.vllm_python,
            str(WORKER),
            "--model",
            export["checkpoint"]["hf_export"],
            "--model-manifest-b64",
            base64.b64encode(json.dumps(export).encode()).decode(),
            "--targets",
            str(DATA / "targets.parquet"),
            "--out",
            str(outputs / phase),
            "--label",
            checkpoint["label"],
            "--shard",
            "0/1",
            "--n-rollouts",
            "100",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--top-k",
            "-1",
            "--contact-mult",
            "6",
            "--seed",
            "0",
        ]
        if phase == "smoke":
            command.extend(["--limit", "1"])
        run_worker(command, outputs, remote)
        markers = markers_for(
            fs,
            str(outputs / phase / checkpoint["label"]),
            1 if phase == "smoke" else 116,
        )
        units = {(u["dataset"], u["stem"]) for m in markers for u in m["units"]}
        selected = [r for r in truth if (r["dataset"], r["stem"]) in units]
        precision, timings = score_checkpoint(
            fs, local_plan, checkpoint, selected, phase=phase
        )
        if phase == "smoke":
            print(f"SMOKE PASSED {checkpoint['label']}", flush=True)
    targets = pd.read_csv(DATA / "targets.csv")
    merged = precision.merge(
        targets.drop(columns="L"), on=["dataset", "stem"], validate="many_to_one"
    )
    merged.to_csv(outputs / "contact_precision_all.csv", index=False)
    timings.to_csv(outputs / "timings.csv", index=False)
    aggregate = merged.groupby(
        ["model", "eval_set", "range", "cut"], as_index=False
    ).agg(
        mean=("precision", "mean"),
        valid=("precision", "count"),
        total=("precision", "size"),
    )
    aggregate.to_csv(outputs / "aggregate.csv", index=False)
    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint,
        "git_commit": os.environ["GIT_COMMIT"],
        "worker_sha256": WORKER_SHA256,
        "metric_sha256": hashlib.sha256(
            (
                HERE.parent
                / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
            ).read_bytes()
        ).hexdigest(),
        "marinfold_revision": MARINFOLD_REVISION,
        "units": 116,
        "rollouts": 11600,
        "unfinished_rollouts": 0,
        "sampling": plan["sampling"],
        "model_egress": False,
        "checkpoint_staging_and_export_seconds": export_seconds,
        "artifact_root": remote,
        "runtime": "one regional H100; models on ephemeral disk",
    }
    (outputs / "manifest.json").write_text(json.dumps(result, indent=2))
    sync_results(outputs, remote)
    print(
        aggregate[
            (aggregate.cut == "R") & aggregate.range.isin(["all", "long"])
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
