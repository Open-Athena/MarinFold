"""Score one checkpoint on one H100, using worker disk and public result storage.

Frozen checkpoints and reusable HF exports stay in CoreWeave. Only completed
score parts, timings and provenance go to the public HF data bucket. Initial
sampling and metric construction follow exp277; capped samples continue under
the user's approved extension policy.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
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
        if marker.get("extension_uri"):
            paths.append(Path(marker["extension_uri"]))
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


def select_eval_set(
    targets: pd.DataFrame, inputs: pd.DataFrame, truth: list[dict], eval_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Select a complete requested set without including designs in natural scores."""
    expected = {"all": 116, "eval-val": 97, "eval-denovo": 19}[eval_set]
    selected = (
        targets if eval_set == "all" else targets[targets.eval_set == eval_set].copy()
    )
    units = set(zip(selected.dataset, selected.stem, strict=True))
    selected_inputs = inputs.merge(
        selected[["dataset", "stem"]], on=["dataset", "stem"], validate="one_to_one"
    )
    selected_truth = [r for r in truth if (r["dataset"], r["stem"]) in units]
    if (
        len(units) != expected
        or len(selected) != expected
        or len(selected_inputs) != expected
    ):
        raise ValueError("Selected evaluation inputs have incomplete coverage")
    if (
        len(selected_truth) != expected
        or {(r["dataset"], r["stem"]) for r in selected_truth} != units
    ):
        raise ValueError("Selected evaluation ground truth has incomplete coverage")
    return selected, selected_inputs, selected_truth


def main() -> None:
    """Export locally, gate one protein, then score the selected evaluation sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument(
        "--eval-set", choices=("all", "eval-val", "eval-denovo"), default="all"
    )
    args = parser.parse_args()
    plan = json.loads((DATA / "plan.json").read_text())
    checkpoint = next(c for c in plan["checkpoints"] if c["trial"] == args.trial)
    bundle = json.loads(Path("bundle_manifest.json").read_text())
    for name, digest in bundle.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundled source changed: {name}")
    if hashlib.sha256(WORKER.read_bytes()).hexdigest() != WORKER_SHA256:
        raise ValueError("The validated worker changed")
    worker = HERE / "eval_worker.py"
    if "capped_sample_policy" not in plan["sampling"]:
        raise ValueError("Extended worker requires the approved continuation policy")
    scratch = Path("/tmp/exp364-contact-eval") / checkpoint["trial"]
    if args.eval_set != "all":
        scratch = scratch / "sets" / args.eval_set
    outputs = scratch / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    remote = plan["artifact_root"] + "/" + checkpoint["label"]
    if args.eval_set != "all":
        remote += "/sets/" + args.eval_set
    public_fs, public_path = fsspec.core.url_to_fs(remote)
    if public_fs.exists(public_path):
        subprocess.run(
            ["hf", "buckets", "sync", remote, str(outputs), "--no-delete", "--quiet"],
            check=True,
        )
    (outputs / "plan.json").write_text(json.dumps(plan, indent=2))
    (outputs / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2))
    sync_results(outputs, remote)
    remote_fs = fsspec.filesystem("s3", use_listings_cache=False)
    with remote_fs.open(
        f"{plan['root']}/manifests/{checkpoint['label']}.json"
    ) as handle:
        export = json.load(handle)
    if export["checkpoint"]["label"] != checkpoint["label"]:
        raise ValueError("Shared export identifies another checkpoint")
    (outputs / "export_manifest.json").write_text(json.dumps(export, indent=2))
    local_plan = {**plan, "root": str(outputs)}
    fs = fsspec.filesystem("file")
    truth = [
        json.loads(line)
        for line in (DATA / "ground_truth.jsonl").read_text().splitlines()
    ]
    target_path = DATA / "targets.parquet"
    targets, selected_inputs, truth = select_eval_set(
        pd.read_csv(DATA / "targets.csv"),
        pd.read_parquet(target_path),
        truth,
        args.eval_set,
    )
    if args.eval_set != "all":
        target_path = scratch / "targets.parquet"
        selected_inputs.to_parquet(target_path, index=False)
    expected_units = len(targets)
    for phase in ("smoke", "rollout"):
        command = [
            args.vllm_python,
            str(worker),
            "--model",
            export["checkpoint"]["hf_export"],
            "--model-manifest-b64",
            base64.b64encode(json.dumps(export).encode()).decode(),
            "--targets",
            str(target_path),
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
            1 if phase == "smoke" else expected_units,
        )
        units = {(u["dataset"], u["stem"]) for m in markers for u in m["units"]}
        selected = [r for r in truth if (r["dataset"], r["stem"]) in units]
        precision, timings = score_checkpoint(
            fs, local_plan, checkpoint, selected, phase=phase
        )
        if phase == "smoke":
            print(f"SMOKE PASSED {checkpoint['label']}", flush=True)
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
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "base_worker_sha256": WORKER_SHA256,
        "extension_sha256": hashlib.sha256(
            (HERE / "eval_generation.py").read_bytes()
        ).hexdigest(),
        "metric_sha256": hashlib.sha256(
            (
                HERE.parent
                / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
            ).read_bytes()
        ).hexdigest(),
        "marinfold_revision": MARINFOLD_REVISION,
        "eval_set": args.eval_set,
        "units": expected_units,
        "rollouts": expected_units * 100,
        "unfinished_rollouts": 0,
        "extended_rollouts": int(timings.extended_rollouts.sum()),
        "sampling": plan["sampling"],
        "model_egress": False,
        "artifact_root": remote,
        "runtime": "one regional H100; frozen native checkpoint and reusable HF export in CoreWeave S3",
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
