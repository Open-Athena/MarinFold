"""Verify existing regional inputs, gate on smoke, and wait for H100 shards."""

import argparse
import hashlib
import io
import json
import shlex
import zipfile
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from fray.client import wait_all
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)

ROOT = "s3://marin-us-east-02a/MarinFold/exp357_evals_train_val_rprecision"
IMAGE = "vllm/vllm-openai:v0.9.2"
HERE = Path(__file__).resolve().parent


def put(uri: str, payload: bytes) -> None:
    """Write one durable artifact through the platform's configured S3 client."""
    with fsspec.open(uri, "wb") as handle:
        handle.write(payload)


def validate_sources() -> None:
    """Verify that selected train documents are identical to exp277's S3 source."""
    manifest = pd.read_csv(HERE / "cohort_manifest.csv")
    train = manifest[manifest.dataset == "afdb_train"]
    for source, rows in train.groupby("source"):
        uri = (
            "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data/afdb/"
            + Path(source).name
        )
        with fsspec.open(uri, "rb") as handle:
            table = pq.read_table(handle, columns=["entry_id", "document"])
        documents = {r["entry_id"]: r["document"] for r in table.to_pylist()}
        for row in rows.itertuples():
            digest = hashlib.sha256(documents[row.entry_id].encode()).hexdigest()
            if digest != row.document_sha256:
                raise ValueError(f"Actual training source mismatch: {row.entry_id}")
    print(
        json.dumps({"event": "training_membership_verified", "proteins": len(train)}),
        flush=True,
    )


def payload(checkpoint: dict) -> bytes:
    """Package pinned inference code and a small wheel, never a repository clone."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("worker.py", "protocol.py", "scoring.py", "exp89_metrics.py"):
            archive.write(HERE / name, name)
        for wheel in HERE.glob("*.whl"):
            archive.write(wheel, wheel.name)
        archive.writestr("checkpoint.json", json.dumps(checkpoint))
    return buffer.getvalue()


def submit_worker(
    client,
    checkpoint: dict,
    root: str,
    shard: int,
    n: int,
    *,
    smoke: bool,
    extra_args: list[str] | None = None,
):
    """Submit one independent GPU worker at batch priority."""
    worker_args = [
        "--targets",
        f"{root}/inputs/targets.json",
        "--checkpoint",
        "checkpoint.json",
        "--out",
        f"{root}/{'smoke' if smoke else 'rollout'}",
        "--shard",
        f"{shard}/{n}",
    ]
    if smoke:
        worker_args.append("--smoke")
    worker_args.extend(extra_args or [])
    shell = "\n".join(
        [
            "set -euo pipefail",
            "VLLM_PY=''",
            'for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do if "$candidate" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$candidate"; break; fi; done',
            'test -n "$VLLM_PY"',
            'export VLLM_PORT=$("$VLLM_PY" -c \'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()\')',
            "work_dir=$(mktemp -d)",
            'cd "$work_dir"',
            '"$VLLM_PY" -m pip install --quiet fsspec==2026.1.0 s3fs==2026.1.0 aiobotocore==2.26.0 pyarrow==23.0.1 scikit-learn==1.7.2 pandas==2.2.3 gemmi==0.7.3',
            '"$VLLM_PY" -c '
            + shlex.quote(
                "import fsspec,io,zipfile; "
                "zipfile.ZipFile(io.BytesIO(fsspec.open("
                + repr(f"{root}/inputs/worker-{checkpoint['label']}.zip")
                + ", 'rb').open().read())).extractall('.')"
            ),
            '"$VLLM_PY" -m pip install --quiet --no-deps marinfold-0.1.0-py3-none-any.whl',
            'export EXP89_METRICS_PATH="$work_dir/exp89_metrics.py"',
            'exec "$VLLM_PY" worker.py ' + shlex.join(worker_args),
        ]
    )
    phase = "smoke" if smoke else "full"
    job = client.submit(
        JobRequest(
            name=f"exp357-{phase}-{checkpoint['job_label']}-{shard:02d}",
            entrypoint=Entrypoint.from_binary("bash", ["-lc", shell]),
            environment=create_environment(
                docker_image=IMAGE,
                setup_scripts=[],
                env_vars={"MARIN_PREFIX": "s3://marin-us-east-02a/MarinFold"},
            ),
            resources=ResourceConfig.with_gpu(
                "H100", count=1, image=IMAGE, cpu=8, ram="64Gi", disk="128Gi"
            ),
            replicas=1,
            processes_per_task=1,
            priority=3,
            max_retries_failure=1,
            max_retries_preemption=100,
        )
    )
    print(
        json.dumps({"event": "submitted", "job": job.job_id, "phase": phase}),
        flush=True,
    )
    return job


def wait_jobs(jobs: list) -> None:
    """Wait for every submitted shard and report all failures together."""
    states = wait_all(jobs, raise_on_failure=False)
    failures = [
        (job.job_id, str(state))
        for job, state in zip(jobs, states, strict=True)
        if state != JobStatus.SUCCEEDED
    ]
    if failures:
        raise RuntimeError(f"Failed shards: {failures}")


def collect(root: str, checkpoints: list[dict], targets: list[dict]) -> None:
    """Consolidate compact metrics only after verifying every expected unit."""
    expected = {(r["dataset"], r["stem"]) for r in targets}
    metric_rows, sample_rows, timing_rows = [], [], []
    for checkpoint in checkpoints:
        filesystem, prefix = fsspec.core.url_to_fs(
            f"{root}/rollout/{checkpoint['label']}/complete"
        )
        found = set()
        for path in sorted(filesystem.glob(prefix + "/*.json")):
            with filesystem.open(path, "rt") as handle:
                result = json.load(handle)
            key = (result["dataset"], result["stem"])
            if key in found or key not in expected:
                raise ValueError(f"Unexpected or duplicate output: {key}")
            found.add(key)
            base = dict(
                dataset=key[0],
                stem=key[1],
                L=result["L"],
                model=checkpoint["label"],
                diagnostic=result["diagnostic"],
            )
            metric_rows.extend({**base, **row} for row in result["metrics"])
            sample_rows.extend({**base, **row} for row in result["single_sample"])
            timing_rows.extend(
                {**row, "model": checkpoint["label"]} for row in result["timings"]
            )
        if found != expected:
            raise ValueError(
                f"Incomplete checkpoint {checkpoint['label']}: {len(found)}/{len(expected)}"
            )
    for name, rows in (
        ("per_protein.csv", metric_rows),
        ("single_samples.csv", sample_rows),
        ("timings.csv", timing_rows),
    ):
        put(f"{root}/results/{name}", pd.DataFrame(rows).to_csv(index=False).encode())
    put(
        f"{root}/results/complete.json",
        json.dumps(
            {"units": len(expected), "models": len(checkpoints), "root": root}
        ).encode(),
    )
    print(json.dumps({"event": "complete", "root": root}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--shards", type=int, default=12)
    args = parser.parse_args()
    root = ROOT + "/" + args.run_id
    targets = json.loads((HERE / "targets.json").read_text())
    checkpoints = json.loads((HERE / "checkpoints.json").read_text())
    validate_sources()
    for name in (
        "targets.json",
        "cohort_manifest.csv",
        "input_provenance.json",
        "checkpoints.json",
        "code_manifest.json",
    ):
        put(f"{root}/inputs/{name}", (HERE / name).read_bytes())
    client = current_client()
    for checkpoint in checkpoints:
        put(f"{root}/inputs/worker-{checkpoint['label']}.zip", payload(checkpoint))
    smokes = [submit_worker(client, c, root, 0, 1, smoke=True) for c in checkpoints]
    wait_jobs(smokes)
    for checkpoint in checkpoints:
        fs, prefix = fsspec.core.url_to_fs(
            f"{root}/smoke/{checkpoint['label']}/complete"
        )
        paths = fs.glob(prefix + "/*.json")
        if len(paths) != 1:
            raise ValueError("Smoke did not finish exactly one protein")
        with fs.open(paths[0], "rt") as handle:
            smoke = json.load(handle)
        if any(t["unfinished_rollouts"] for t in smoke["timings"]):
            raise ValueError("Smoke has capped samples; inspect before production")
    jobs = [
        submit_worker(client, c, root, i, args.shards, smoke=False)
        for c in checkpoints
        for i in range(args.shards)
    ]
    wait_jobs(jobs)
    collect(root, checkpoints, targets)


if __name__ == "__main__":
    main()
