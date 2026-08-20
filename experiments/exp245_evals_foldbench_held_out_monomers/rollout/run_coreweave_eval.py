# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the exp245 FoldBench-monomer evaluation entirely inside CoreWeave.

Adapted from PR #244's exp232 driver, which this experiment is stacked on. The
CPU driver mirrors the immutable public evaluation inputs into the shared
CoreWeave S3 bucket, verifies all checkpoints in place, launches one smoke job
per checkpoint, launches the full 12-way H100 fanout, waits, and finalizes
metrics.

Two things differ from #244's version. The eval set is exp245's published
334-unit FoldBench monomer universe -- mirrored and validated, not assembled
from the legacy targets at run time -- and the reporting cuts are eval-val /
eval-test / eval-denovo with their viral splits instead of the legacy and eval2
cuts.
"""

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fray.client import JobHandle, wait_all
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)

from checkpoint_specs import (
    CHECKPOINT_SUITES,
    EXPECTED_SET_SIZES,
    EXPECTED_UNITS,
    GROUND_TRUTH_SHA256,
    GROUND_TRUTH_SIZE,
    GROUND_TRUTH_URL,
    MARIN_PREFIX,
    MARINFOLD_REVISION,
    SETS_MANIFEST_SHA256,
    SETS_MANIFEST_SIZE,
    SETS_MANIFEST_URL,
    TARGETS_SHA256,
    TARGETS_SIZE,
    TARGETS_URL,
    Checkpoint,
    checkpoint_model_uri,
    run_root,
)
from finalize_coreweave import finalize
from hf_to_s3 import (
    expected_manifest,
    mirror_public_input,
    verify_checkpoint_at_uri,
)

NUM_SHARDS = 12
NUM_ROLLOUTS = 100
GPU_IMAGE = "vllm/vllm-openai:v0.9.2"
CHILD_CPU = 8
CHILD_MEMORY = "64Gi"
CHILD_DISK = "128Gi"
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {field.name for field in dataclasses.fields(JobRequest)}, (
    "This Fray build lacks JobRequest.priority; batch dispatch needs the pinned 0.2.x.dev build."
)


def _read_worker() -> tuple[str, str]:
    worker_path = Path(__file__).with_name("score_rollout_worker.py")
    worker_bytes = worker_path.read_bytes()
    return base64.b64encode(worker_bytes).decode(), hashlib.sha256(
        worker_bytes
    ).hexdigest()


def _child_command(
    *,
    worker_b64: str,
    model_manifest_b64: str,
    model_uri: str,
    targets_uri: str,
    output_uri: str,
    label: str,
    shard_idx: int,
    num_shards: int,
    vllm_port: int,
    seed: int,
    contact_mult: int,
    accept_unfinished: bool,
    limit: int | None = None,
) -> list[str]:
    if any(value.startswith("gs://") for value in (model_uri, targets_uri, output_uri)):
        raise ValueError("GCS sources are forbidden for this evaluation")

    worker_args = [
        "--model",
        model_uri,
        "--model-manifest-b64",
        model_manifest_b64,
        "--targets",
        targets_uri,
        "--out",
        output_uri,
        "--label",
        label,
        "--shard",
        f"{shard_idx}/{num_shards}",
        "--n-rollouts",
        str(NUM_ROLLOUTS),
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "-1",
        "--contact-mult",
        str(contact_mult),
        "--seed",
        str(seed),
    ]
    if limit is not None:
        worker_args.extend(["--limit", str(limit)])
    if accept_unfinished:
        worker_args.append("--accept-unfinished")

    python_candidates = (
        "for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 "
        "/opt/venv/bin/python python3 python; do "
        "if \"$candidate\" -c 'import vllm' >/dev/null 2>&1; then "
        'VLLM_PY="$candidate"; break; fi; done'
    )
    install_storage = (
        "uv pip install --python \"$VLLM_PY\" --quiet 'fsspec==2026.1.0' "
        "'s3fs==2026.1.0' 'aiobotocore==2.26.0' 'pyarrow>=23,<24' "
        "|| \"$VLLM_PY\" -m pip install --quiet 'fsspec==2026.1.0' "
        "'s3fs==2026.1.0' 'aiobotocore==2.26.0' 'pyarrow>=23,<24'"
    )
    marinfold_package = (
        "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@"
        f"{MARINFOLD_REVISION}#subdirectory=marinfold"
    )
    install_marinfold = (
        'uv pip install --python "$VLLM_PY" --quiet --no-deps '
        f"{shlex.quote(marinfold_package)} || "
        '"$VLLM_PY" -m pip install --quiet --no-deps '
        f"{shlex.quote(marinfold_package)}"
    )
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"export MARIN_PREFIX={shlex.quote(MARIN_PREFIX)}",
            f"export VLLM_PORT={vllm_port}",
            "VLLM_PY=''",
            python_candidates,
            "if [ -z \"$VLLM_PY\" ]; then echo 'no Python interpreter imports vLLM'; exit 3; fi",
            install_storage,
            install_marinfold,
            '"$VLLM_PY" -c \'from marinfold.document_structures.contacts_v1 import build_document; print("marinfold import OK")\'',
            "work_dir=$(mktemp -d)",
            "trap 'rm -rf \"$work_dir\"' EXIT",
            f'printf %s {shlex.quote(worker_b64)} | base64 -d > "$work_dir/score_rollout_worker.py"',
            'exec "$VLLM_PY" "$work_dir/score_rollout_worker.py" '
            + " ".join(shlex.quote(arg) for arg in worker_args),
        ]
    )
    return ["bash", "-lc", shell]


def _job_request(*, name: str, command: list[str]) -> JobRequest:
    environment = create_environment(
        docker_image=GPU_IMAGE,
        env_vars={
            "MARIN_PREFIX": MARIN_PREFIX,
        },
        setup_scripts=[],
    )
    resources = ResourceConfig.with_gpu(
        "H100",
        count=1,
        image=GPU_IMAGE,
        cpu=CHILD_CPU,
        ram=CHILD_MEMORY,
        disk=CHILD_DISK,
    )
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary(command[0], command[1:]),
        environment=environment,
        resources=resources,
        replicas=1,
        processes_per_task=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        max_retries_failure=3,
        max_retries_preemption=100,
    )


def _wait_for_jobs(jobs: list[JobHandle], phase: str) -> None:
    job_ids = [job.job_id for job in jobs]
    print(json.dumps({"event": "waiting", "phase": phase, "jobs": job_ids}), flush=True)
    results = wait_all(jobs, raise_on_failure=False)
    failures: list[dict[str, str]] = []
    for job_id, status in zip(job_ids, results, strict=True):
        status_text = str(status.value)
        print(
            json.dumps(
                {
                    "event": "child_finished",
                    "phase": phase,
                    "job_id": job_id,
                    "status": status_text,
                }
            ),
            flush=True,
        )
        if status != JobStatus.SUCCEEDED:
            failures.append({"job_id": job_id, "status": status_text})
    if failures:
        raise RuntimeError(f"{phase} child failures: {failures}")


def _validate_smokes(smoke_root: str, checkpoints: tuple[Checkpoint, ...]) -> None:
    filesystem, root = fsspec.core.url_to_fs(smoke_root)
    for checkpoint in checkpoints:
        pattern = f"{root.rstrip('/')}/{checkpoint.label}/complete/*.json"
        markers = filesystem.glob(pattern)
        if len(markers) != 1:
            raise RuntimeError(
                f"Expected one smoke completion marker for {checkpoint.label}, found {len(markers)}"
            )
        with filesystem.open(markers[0], "rt") as handle:
            record = json.load(handle)
        units = record.get("units", [])
        if (
            len(units) != 1
            or record.get("total_rollouts") != NUM_ROLLOUTS
            or units[0].get("n_rollouts") != NUM_ROLLOUTS
            or record.get("unfinished_rollouts") != 0
        ):
            raise RuntimeError(f"Invalid smoke result for {checkpoint.label}: {record}")


def _read_parquet(uri: str) -> pd.DataFrame:
    """Read a small parquet input through the configured CoreWeave filesystem."""

    with fsspec.open(uri, "rb") as handle:
        return pq.read_table(handle).to_pandas()


def _write_parquet(frame: pd.DataFrame, uri: str) -> None:
    """Write a small parquet input through the configured CoreWeave filesystem."""

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), handle)


def _validate_targets(*, targets_uri: str, sets_uri: str) -> dict:
    """Check the mirrored eval set is the 334-unit universe exp245 published.

    The digests already prove the bytes are the published ones; this proves the
    published ones are still the eval set this code expects -- same unit count,
    same per-set sizes, sequences whose lengths match ``L``, no duplicate units.
    """

    targets = _read_parquet(targets_uri)
    with fsspec.open(sets_uri, "rt") as handle:
        sets = pd.read_csv(handle)
    # The manifest lists every FoldBench monomer, including the ones held out of
    # the scored universe (a document that does not fit the model's context).
    # The targets are the scorable ones, so the sizes are compared on those.
    scorable = sets[sets.scorable == 1]

    units = list(zip(targets.dataset, targets.stem, strict=True))
    validation = {
        "units": len(targets),
        "unique_units": len(set(units)),
        "unique_stems": int(targets.stem.nunique()),
        "set_sizes": scorable.eval_set.value_counts().to_dict(),
        "excluded": sets.loc[sets.scorable == 0, "stem"].tolist(),
    }
    if validation["units"] != EXPECTED_UNITS:
        raise ValueError(f"expected {EXPECTED_UNITS} units, got {validation['units']}")
    if validation["unique_units"] != EXPECTED_UNITS:
        raise ValueError("targets contain duplicate (dataset, stem) units")
    if validation["set_sizes"] != EXPECTED_SET_SIZES:
        raise ValueError(f"eval set sizes changed: {validation['set_sizes']}")
    if set(targets.stem) != set(scorable.stem):
        raise ValueError("targets and the set manifest cover different proteins")
    if (targets.input_seq.str.len() != targets.L).any():
        raise ValueError("target sequence lengths do not match L")
    return validation


def _submit_phase(
    *,
    client,
    run_id: str,
    model_mirror_run_id: str,
    worker_b64: str,
    targets_uri: str,
    output_uri: str,
    smoke: bool,
    seed: int,
    contact_mult: int,
    checkpoints: tuple[Checkpoint, ...],
) -> list[JobHandle]:
    requests = []
    shards = range(1) if smoke else range(NUM_SHARDS)
    num_shards = 1 if smoke else NUM_SHARDS
    phase = "smoke" if smoke else "full"
    for model_slot, checkpoint in enumerate(checkpoints):
        model_manifest_b64 = base64.b64encode(
            json.dumps(expected_manifest(checkpoint), sort_keys=True).encode()
        ).decode()
        for shard_idx in shards:
            name = (
                f"e245-{run_id}-{phase[:2]}-{checkpoint.job_label}-s{shard_idx:02d}"
            )
            command = _child_command(
                worker_b64=worker_b64,
                model_manifest_b64=model_manifest_b64,
                model_uri=checkpoint_model_uri(model_mirror_run_id, checkpoint),
                targets_uri=targets_uri,
                output_uri=output_uri,
                label=checkpoint.label,
                shard_idx=shard_idx,
                num_shards=num_shards,
                vllm_port=20_000 + model_slot * 10_000 + shard_idx * 100,
                seed=seed,
                contact_mult=contact_mult,
                accept_unfinished=(
                    not smoke and checkpoint.accepted_unfinished_rollouts > 0
                ),
                limit=1 if smoke else None,
            )
            requests.append(_job_request(name=name, command=command))
    jobs = [client.submit(request) for request in requests]
    print(
        json.dumps(
            {"event": "submitted", "phase": phase, "jobs": [job.job_id for job in jobs]}
        ),
        flush=True,
    )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model-mirror-run-id",
        help="Reuse verified model mirrors from another run (default: --run-id).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument("--suite", choices=sorted(CHECKPOINT_SUITES), default="exp245")
    args = parser.parse_args()
    if args.contact_mult < 6:
        raise ValueError("--contact-mult must be at least the standard value of 6")
    model_mirror_run_id = args.model_mirror_run_id or args.run_id
    checkpoints = CHECKPOINT_SUITES[args.suite]

    if os.environ.get("MARIN_PREFIX") != MARIN_PREFIX:
        raise RuntimeError(f"MARIN_PREFIX must be exactly {MARIN_PREFIX}")

    root = run_root(args.run_id)
    started_at = datetime.now(UTC).isoformat()
    worker_b64, worker_sha256 = _read_worker()
    print(
        json.dumps(
            {
                "event": "start",
                "run_id": args.run_id,
                "run_root": root,
                "model_mirror_run_id": model_mirror_run_id,
                "marin_prefix": MARIN_PREFIX,
                "checkpoint_sources": [
                    checkpoint.coreweave_uri or "huggingface"
                    for checkpoint in checkpoints
                ],
                "seed": args.seed,
                "suite": args.suite,
            }
        ),
        flush=True,
    )

    for checkpoint in checkpoints:
        if not checkpoint.coreweave_uri:
            raise ValueError(
                f"checkpoint copying is disabled; no CoreWeave path for {checkpoint.label}"
            )
        verify_checkpoint_at_uri(
            checkpoint=checkpoint,
            source_uri=checkpoint.coreweave_uri,
            verification_uri=(
                f"{root}/inputs/checkpoint_verification/{checkpoint.label}.json"
            ),
        )

    targets_uri = f"{root}/inputs/eval_targets_foldbench_monomers.parquet"
    sets_uri = f"{root}/inputs/eval_sets.csv"
    ground_truth_uri = f"{root}/inputs/gt_universe_foldbench_monomers.jsonl"
    mirror_public_input(
        url=TARGETS_URL,
        destination_uri=targets_uri,
        expected_size=TARGETS_SIZE,
        expected_sha256=TARGETS_SHA256,
    )
    mirror_public_input(
        url=SETS_MANIFEST_URL,
        destination_uri=sets_uri,
        expected_size=SETS_MANIFEST_SIZE,
        expected_sha256=SETS_MANIFEST_SHA256,
    )
    mirror_public_input(
        url=GROUND_TRUTH_URL,
        destination_uri=ground_truth_uri,
        expected_size=GROUND_TRUTH_SIZE,
        expected_sha256=GROUND_TRUTH_SHA256,
    )
    validation = _validate_targets(targets_uri=targets_uri, sets_uri=sets_uri)
    print(json.dumps({"event": "targets_validated", **validation}), flush=True)

    client = current_client()
    smoke_root = f"{root}/smoke"
    smoke_jobs = _submit_phase(
        client=client,
        run_id=args.run_id,
        model_mirror_run_id=model_mirror_run_id,
        worker_b64=worker_b64,
        targets_uri=targets_uri,
        output_uri=smoke_root,
        smoke=True,
        seed=args.seed,
        contact_mult=args.contact_mult,
        checkpoints=checkpoints,
    )
    _wait_for_jobs(smoke_jobs, "smoke")
    _validate_smokes(smoke_root, checkpoints)

    rollout_root = f"{root}/rollout"
    full_jobs = _submit_phase(
        client=client,
        run_id=args.run_id,
        model_mirror_run_id=model_mirror_run_id,
        worker_b64=worker_b64,
        targets_uri=targets_uri,
        output_uri=rollout_root,
        smoke=False,
        seed=args.seed,
        contact_mult=args.contact_mult,
        checkpoints=checkpoints,
    )
    _wait_for_jobs(full_jobs, "full")

    result = finalize(
        run_root=root,
        score_root=rollout_root,
        ground_truth_uri=ground_truth_uri,
        sets_manifest_uri=sets_uri,
        worker_sha256=worker_sha256,
        job_ids=[job.job_id for job in smoke_jobs + full_jobs],
        started_at=started_at,
        model_mirror_run_id=model_mirror_run_id,
        sampling_seed=args.seed,
        contact_mult=args.contact_mult,
        checkpoints=checkpoints,
        suite=args.suite,
    )
    print(
        json.dumps({"event": "complete", "result": result}, sort_keys=True), flush=True
    )


if __name__ == "__main__":
    main()
