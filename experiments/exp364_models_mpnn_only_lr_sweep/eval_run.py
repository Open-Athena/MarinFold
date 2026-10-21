"""Run the requested 116-protein evaluation using the validated exp277 worker."""

import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from fray.client import wait_all
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)

from experiments.exp89_evals_contacts_v1_model_on_eval_set.compute_metrics import (
    metric_rows,
    resolved_pairs,
    stamp,
    true_matrix,
)
from experiments.exp364_models_mpnn_only_lr_sweep.eval_export import freeze_source

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "contact_eval"
WORKER = (
    HERE.parent
    / "exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/score_rollout_worker.py"
)
WORKER_SHA256 = "dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c"
MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
IMAGE = "vllm/vllm-openai:v0.9.2"


def read_json(fs: fsspec.AbstractFileSystem, path: str) -> dict:
    """Read a single result or provenance object."""
    with fs.open(path) as handle:
        return json.load(handle)


def worker_request(
    plan: dict, checkpoint: dict, manifest: dict, shard: int, smoke: bool
) -> JobRequest:
    """Use the existing worker byte-for-byte with explicit sampling arguments."""
    worker = WORKER.read_bytes()
    if hashlib.sha256(worker).hexdigest() != WORKER_SHA256:
        raise ValueError("The validated rollout worker has changed")
    phase = "smoke" if smoke else "rollout"
    args = [
        "--model",
        checkpoint["hf_export"],
        "--model-manifest-b64",
        base64.b64encode(json.dumps(manifest).encode()).decode(),
        "--targets",
        plan["root"] + "/inputs/targets.parquet",
        "--out",
        plan["root"] + "/" + phase,
        "--label",
        checkpoint["label"],
        "--shard",
        f"{shard}/{1 if smoke else 3}",
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
    if smoke:
        args.extend(["--limit", "1"])
    package = f"marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{MARINFOLD_REVISION}#subdirectory=marinfold"
    shell = "\n".join(
        [
            "set -euo pipefail",
            "VLLM_PY=''",
            'for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do if "$candidate" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$candidate"; break; fi; done',
            'test -n "$VLLM_PY"',
            'export VLLM_PORT=$("$VLLM_PY" -c \'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()\')',
            '"$VLLM_PY" -m pip install --quiet "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0" "pyarrow>=23,<24"',
            '"$VLLM_PY" -m pip install --quiet --no-deps ' + shlex.quote(package),
            "work_dir=$(mktemp -d)",
            f'printf %s {shlex.quote(base64.b64encode(worker).decode())} | base64 -d > "$work_dir/worker.py"',
            'exec "$VLLM_PY" "$work_dir/worker.py" ' + shlex.join(args),
        ]
    )
    env = {
        key: os.environ[key] for key in ("FSSPEC_S3", "HF_TOKEN") if key in os.environ
    }
    return JobRequest(
        name=f"exp364-eval-{phase}-{checkpoint['trial']}-s{shard}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", shell]),
        resources=ResourceConfig.with_gpu(
            "H100",
            count=1,
            cpu=8,
            ram="64Gi",
            disk="128Gi",
            image=IMAGE,
            target_cluster=plan["cluster"],
        ),
        environment=create_environment(
            docker_image=IMAGE, env_vars=env, setup_scripts=[]
        ),
        priority=3,
        max_retries_failure=0,
        max_retries_preemption=10,
    )


def markers_for(fs: fsspec.AbstractFileSystem, root: str, expected: int) -> list[dict]:
    """Require complete, uncapped rollouts and unique evaluation units."""
    markers = [
        read_json(fs, path) for path in sorted(fs.glob(root + "/complete/*.json"))
    ]
    units = []
    for marker in markers:
        if marker["unfinished_rollouts"] != 0 or marker["total_rollouts"] != 100 * len(
            marker["units"]
        ):
            raise ValueError(f"Invalid or capped rollouts: {root}: {marker}")
        for unit in marker["units"]:
            if unit["n_rollouts"] != 100 or unit["unfinished_rollouts"] != 0:
                raise ValueError(f"Incomplete unit: {unit}")
            units.append((unit["dataset"], unit["stem"]))
    if len(units) != expected or len(set(units)) != expected:
        raise ValueError(
            f"Expected {expected} unique units at {root}, found {len(units)}"
        )
    return markers


def score_checkpoint(
    fs: fsspec.AbstractFileSystem,
    plan: dict,
    checkpoint: dict,
    truth: list[dict],
    *,
    phase: str = "rollout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct vote matrices and call exp89's unchanged metric functions."""
    root = f"{plan['root']}/{phase}/{checkpoint['label']}"
    markers = markers_for(fs, root, len(truth))
    gt = {(r["dataset"], r["stem"]): r for r in truth}
    units = {(u["dataset"], u["stem"]): u["L"] for m in markers for u in m["units"]}
    if units != {key: record["L"] for key, record in gt.items()}:
        raise ValueError("Completed proteins do not match ground truth")
    matrices = {
        key: np.zeros((length, length), dtype=np.float64)
        for key, length in units.items()
    }
    timing_parts = []
    seen = set()
    for marker in markers:
        with fs.open(marker["score_uri"], "rb") as handle:
            scores = pd.read_parquet(handle)
        for row in scores.itertuples(index=False):
            key = (row.dataset, row.stem)
            pair = (*key, row.i, row.j)
            if (
                key not in matrices
                or pair in seen
                or not (0 <= row.i < row.j < units[key])
                or not (0 < row.votes <= 100)
            ):
                raise ValueError(f"Invalid sparse vote row: {row}")
            if row.L != units[key]:
                raise ValueError(f"Vote-matrix length mismatch: {key}")
            seen.add(pair)
            matrices[key][row.i, row.j] = matrices[key][row.j, row.i] = row.votes
        with fs.open(marker["timing_uri"], "rb") as handle:
            timing_parts.append(pd.read_parquet(handle))
    timings = pd.concat(timing_parts, ignore_index=True)
    if len(timings) != len(truth) or set(
        zip(timings.dataset, timings.stem, strict=True)
    ) != set(gt):
        raise ValueError("Timing coverage differs from scored units")
    if not (
        (timings.n_rollouts == 100)
        & (timings.stopped_rollouts == 100)
        & (timings.unfinished_rollouts == 0)
    ).all():
        raise ValueError("Timing records reveal incomplete rollouts")
    rows = []
    fs.makedirs(f"{plan['root']}/dense/{phase}/{checkpoint['label']}", exist_ok=True)
    for key, record in gt.items():
        matrix = matrices[key]
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"Asymmetric votes: {key}")
        with fs.open(
            f"{plan['root']}/dense/{phase}/{checkpoint['label']}/{key[0]}__{key[1]}.npz",
            "wb",
        ) as handle:
            np.savez_compressed(handle, score=matrix.astype(np.float16))
        pi, pj, separation = resolved_pairs(
            np.asarray(record["resolved"], dtype=np.int64)
        )
        metrics = metric_rows(
            matrix,
            true_matrix(record["L"], record["contacts"]),
            pi,
            pj,
            separation,
            record["L"],
            with_precision=True,
        )
        rows.extend(
            stamp(
                metrics,
                rec=record,
                model=checkpoint["label"],
                mode="single_seq",
                predictor="lm",
            )
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(truth) * 20:
        raise ValueError("Expected 20 metric rows per protein")
    defined = ((frame.cut != "AUC") & (frame.n_top > 0)) | (
        (frame.cut == "AUC") & (frame.n_true > 0) & (frame.n_true < frame.n_candidate)
    )
    if not np.isfinite(frame.loc[defined, "precision"]).all():
        raise ValueError("A mathematically defined metric is nonfinite")
    if not frame.loc[defined, "precision"].between(0, 1).all():
        raise ValueError("A precision/AUC value is outside [0, 1]")
    timings["checkpoint"] = checkpoint["label"]
    return frame, timings


def main() -> None:
    """Freeze/export four models, gate each on a smoke, and score twelve shards."""
    plan = json.loads((DATA / "plan.json").read_text())
    bundle = json.loads(Path("bundle_manifest.json").read_text())
    for name, digest in bundle.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundled source changed: {name}")
    fs = fsspec.filesystem("s3", use_listings_cache=False)
    fs.put_file("bundle_manifest.json", plan["root"] + "/bundle_manifest.json")
    for name, digest in plan["inputs"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed: {name}")
        fs.put_file(str(DATA / name), plan["root"] + "/inputs/" + name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda checkpoint: freeze_source(fs, checkpoint), plan["checkpoints"]
            )
        )
    for checkpoint in plan["checkpoints"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.exp364_models_mpnn_only_lr_sweep.eval_export",
                "--plan",
                str(DATA / "plan.json"),
                "--trial",
                checkpoint["trial"],
            ],
            check=True,
        )
    manifests = {
        c["label"]: read_json(fs, f"{plan['root']}/manifests/{c['label']}.json")
        for c in plan["checkpoints"]
    }
    client = current_client()
    job_ids = []
    truth = [
        json.loads(line)
        for line in (DATA / "ground_truth.jsonl").read_text().splitlines()
    ]
    for smoke in (True, False):
        jobs = [
            client.submit(
                worker_request(
                    plan, checkpoint, manifests[checkpoint["label"]], shard, smoke
                )
            )
            for checkpoint in plan["checkpoints"]
            for shard in range(1 if smoke else 3)
        ]
        job_ids.extend(job.job_id for job in jobs)
        print(
            json.dumps(
                {
                    "phase": "smoke" if smoke else "full",
                    "jobs": [job.job_id for job in jobs],
                }
            ),
            flush=True,
        )
        with fs.open(plan["root"] + "/jobs.json", "wt") as handle:
            json.dump(job_ids, handle)
        statuses = wait_all(jobs, raise_on_failure=False)
        failures = {
            job.job_id: status
            for job, status in zip(jobs, statuses, strict=True)
            if status != JobStatus.SUCCEEDED
        }
        if failures:
            raise RuntimeError(
                f"Evaluation jobs failed; partial outputs retained: {failures}"
            )
        if smoke:
            for checkpoint in plan["checkpoints"]:
                markers = markers_for(
                    fs, f"{plan['root']}/smoke/{checkpoint['label']}", 1
                )
                unit = markers[0]["units"][0]
                smoke_truth = [
                    r
                    for r in truth
                    if (r["dataset"], r["stem"]) == (unit["dataset"], unit["stem"])
                ]
                score_checkpoint(fs, plan, checkpoint, smoke_truth, phase="smoke")
    outputs = [
        score_checkpoint(fs, plan, checkpoint, truth)
        for checkpoint in plan["checkpoints"]
    ]
    precision = pd.concat([p for p, _ in outputs], ignore_index=True)
    timings = pd.concat([t for _, t in outputs], ignore_index=True)
    targets = pd.read_csv(DATA / "targets.csv")
    merged = precision.merge(
        targets.drop(columns="L"), on=["dataset", "stem"], validate="many_to_one"
    )
    aggregate = merged.groupby(
        ["model", "eval_set", "range", "cut"], as_index=False
    ).agg(
        mean=("precision", "mean"),
        valid=("precision", "count"),
        total=("precision", "size"),
    )
    for name, frame in [
        ("contact_precision_all.csv", merged),
        ("timings.csv", timings),
        ("aggregate.csv", aggregate),
    ]:
        with fs.open(plan["root"] + "/results/" + name, "wt") as handle:
            frame.to_csv(handle, index=False)
    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "plan": plan,
        "git_commit": os.environ["GIT_COMMIT"],
        "bundle_manifest": bundle,
        "jobs": job_ids,
        "worker_sha256": WORKER_SHA256,
        "metric_sha256": hashlib.sha256(
            (
                HERE.parent
                / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
            ).read_bytes()
        ).hexdigest(),
        "marinfold_revision": MARINFOLD_REVISION,
        "units_per_checkpoint": 116,
        "rollouts_per_checkpoint": 11600,
        "unfinished_rollouts": 0,
    }
    with fs.open(plan["root"] + "/results/manifest.json", "wt") as handle:
        json.dump(result, handle, indent=2)
    print(
        aggregate[
            (aggregate.cut == "R") & aggregate.range.isin(["all", "long"])
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
