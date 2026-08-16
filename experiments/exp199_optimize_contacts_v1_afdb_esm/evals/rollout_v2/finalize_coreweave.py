# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate CoreWeave rollout parts and run exp89's metric script unchanged."""

import base64
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from checkpoint_specs import (
    E8_REFERENCE_METRICS,
    E8_REFERENCE_TOLERANCE,
    GROUND_TRUTH_SHA256,
    Checkpoint,
    checkpoint_model_uri,
)

EXPECTED_UNITS = 577
EXPECTED_UNIQUE_STEMS = 575
EXPECTED_ROLLOUTS_PER_UNIT = 100
EXP89_OUTPUT_COLUMNS = [
    "dataset",
    "stem",
    "n_residues",
    "model",
    "mode",
    "predictor",
    "range",
    "cut",
    "precision",
    "n_candidate",
    "n_true",
    "n_top",
    "neff_tier",
    "fold_verdict",
    "seq_leakage",
    "msa_neff",
    "length",
]


def read_json(filesystem, path: str) -> dict:
    """Read one JSON object from fsspec storage."""

    with filesystem.open(path, "rt") as file:
        return json.load(file)


def write_json(filesystem, path: str, data: dict) -> None:
    """Write canonical JSON through fsspec."""

    with filesystem.open(path, "wt") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def read_table(filesystem, path: str) -> pa.Table:
    """Read one parquet table from fsspec storage."""

    with filesystem.open(path, "rb") as file:
        return pq.read_table(file)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(filesystem, source: str, destination: Path) -> None:
    """Download one object-store file to local ephemeral disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem.get_file(source, str(destination))


def completion_records(
    filesystem, label_root: str
) -> tuple[list[dict], dict[tuple[str, str], int]]:
    """Load markers and validate unit and rollout completeness."""

    marker_paths = sorted(filesystem.glob(f"{label_root}/complete/*.json"))
    if not marker_paths:
        raise ValueError(f"no completion markers under {label_root}")
    markers = [read_json(filesystem, path) for path in marker_paths]
    lengths: dict[tuple[str, str], int] = {}
    rollout_total = 0
    for marker in markers:
        if marker["unfinished_rollouts"] != 0:
            raise ValueError(f"unfinished rollouts in completion marker: {marker}")
        rollout_total += marker["total_rollouts"]
        for unit in marker["units"]:
            key = (unit["dataset"], unit["stem"])
            if key in lengths:
                raise ValueError(f"duplicate completed unit: {key}")
            if unit["n_rollouts"] != EXPECTED_ROLLOUTS_PER_UNIT:
                raise ValueError(f"wrong rollout count for {key}: {unit['n_rollouts']}")
            lengths[key] = unit["L"]
    if len(lengths) != EXPECTED_UNITS:
        raise ValueError(
            f"incomplete units under {label_root}: {len(lengths)} != {EXPECTED_UNITS}"
        )
    if len({stem for _, stem in lengths}) != EXPECTED_UNIQUE_STEMS:
        raise ValueError("unique-stem count does not match the exp89 universe")
    expected_rollouts = EXPECTED_UNITS * EXPECTED_ROLLOUTS_PER_UNIT
    if rollout_total != expected_rollouts:
        raise ValueError(f"rollout total {rollout_total} != {expected_rollouts}")
    return markers, lengths


def dense_scores(
    *,
    filesystem,
    label_root: str,
    markers: list[dict],
    lengths: dict[tuple[str, str], int],
    local_directory: Path,
    dense_s3_root: str,
) -> int:
    """Rebuild symmetric exp89-compatible matrices and publish them to S3."""

    local_directory.mkdir(parents=True, exist_ok=True)
    matrices = {
        key: np.zeros((length, length), dtype=np.float32)
        for key, length in lengths.items()
    }
    score_paths = [marker["score_uri"] for marker in markers]
    existing_paths = sorted(filesystem.glob(f"{label_root}/scores/*.parquet"))
    if {filesystem._strip_protocol(path) for path in score_paths} != set(
        existing_paths
    ):
        raise ValueError(
            f"score-part set does not match completion markers under {label_root}"
        )
    seen_pairs: set[tuple[str, str, int, int]] = set()
    for path in score_paths:
        table = read_table(filesystem, path).to_pydict()
        for dataset, stem, length, row, column, votes in zip(
            table["dataset"],
            table["stem"],
            table["L"],
            table["i"],
            table["j"],
            table["votes"],
            strict=True,
        ):
            key = (dataset, stem)
            if key not in matrices or matrices[key].shape != (length, length):
                raise ValueError(
                    f"score row has unknown identity or length: {key}, L={length}"
                )
            pair = (dataset, stem, row, column)
            if pair in seen_pairs:
                raise ValueError(f"duplicate sparse score pair: {pair}")
            seen_pairs.add(pair)
            matrices[key][row, column] = votes
            matrices[key][column, row] = votes

    for (dataset, stem), matrix in matrices.items():
        destination = local_directory / f"{dataset}__{stem}.npz"
        np.savez_compressed(destination, score=matrix.astype(np.float16))
        filesystem.put_file(str(destination), f"{dense_s3_root}/{destination.name}")
    return len(matrices)


def collect_timings(
    *,
    filesystem,
    markers: list[dict],
    expected_units: set[tuple[str, str]],
) -> pd.DataFrame:
    """Combine and validate per-protein timing records."""

    tables = [read_table(filesystem, marker["timing_uri"]) for marker in markers]
    timings = pa.concat_tables(tables).to_pandas()
    timing_units = set(zip(timings.dataset, timings.stem, strict=True))
    if timing_units != expected_units or len(timings) != len(expected_units):
        raise ValueError("timing rows do not cover each completed unit exactly once")
    if not timings.complete.all() or timings.unfinished_rollouts.sum() != 0:
        raise ValueError("timing records report incomplete rollouts")
    if set(timings.n_rollouts) != {EXPECTED_ROLLOUTS_PER_UNIT}:
        raise ValueError("timing records have an incorrect rollout count")
    return timings


def write_exp89_script(path: Path) -> str:
    """Materialize the exact exp89 metric source supplied by the submitter."""

    encoded = os.environ.get("EXP89_COMPUTE_METRICS_B64")
    expected_sha256 = os.environ.get("EXP89_COMPUTE_METRICS_SHA256")
    if not encoded or not expected_sha256:
        raise ValueError("the submitter did not supply the pinned exp89 metric source")
    source = base64.b64decode(encoded)
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"exp89 metric source digest mismatch: {actual_sha256} != {expected_sha256}"
        )
    path.write_bytes(source)
    return actual_sha256


def run_exp89_metrics(
    *,
    scratch: Path,
    ground_truth: Path,
    score_directories: dict[str, Path],
) -> tuple[Path, Path, str]:
    """Run exp89's compute_metrics.py unchanged with empty baseline inputs."""

    script = scratch / "compute_metrics_exp89.py"
    script_sha256 = write_exp89_script(script)
    empty_scores = scratch / "empty_scores"
    empty_scores.mkdir(exist_ok=True)
    empty_precision = scratch / "empty_precision.csv"
    pd.DataFrame(columns=EXP89_OUTPUT_COLUMNS).to_csv(empty_precision, index=False)
    empty_raw = scratch / "empty_raw.parquet"
    pd.DataFrame(
        columns=["role", "mode", "model", "dataset", "stem", "i", "j", "degree"]
    ).to_parquet(empty_raw, index=False)
    output = scratch / "contact_precision_all.csv"
    command = [
        sys.executable,
        str(script),
        "--gt",
        str(ground_truth),
        "--scores",
        str(empty_scores),
        "--exp78-precision",
        str(empty_precision),
        "--exp78-raw",
        str(empty_raw),
        "--exp74-raw",
        str(empty_raw),
        "--out",
        str(output),
    ]
    for label, directory in score_directories.items():
        metric_label = f"marinfold-{label.replace('_', '-')}"
        command.extend(["--extra", f"{metric_label}={directory}"])
    subprocess.run(command, check=True)
    marinfold_output = scratch / "marinfold_precision.csv"
    return output, marinfold_output, script_sha256


def validate_e8_reference(aggregate: pd.DataFrame, suite: str) -> dict | None:
    """Compare the E8 suite to exp82's historical rollout metrics."""

    if suite != "e8-reference":
        return None
    headline = aggregate[
        (aggregate["subset"] == "legacy_554")
        & aggregate["range"].isin(["all", "long"])
        & aggregate["cut"].isin(["R", "AUC"])
    ]
    if len(headline) != len(E8_REFERENCE_METRICS):
        raise ValueError(f"E8 headline metric set is incomplete: {headline}")
    comparisons = []
    for row in headline.itertuples(index=False):
        key = (row.range, row.cut)
        expected = E8_REFERENCE_METRICS[key]
        difference = abs(row.precision - expected)
        comparisons.append(
            {
                "range": row.range,
                "cut": row.cut,
                "expected": expected,
                "actual": row.precision,
                "absolute_difference": difference,
                "within_tolerance": difference <= E8_REFERENCE_TOLERANCE,
            }
        )
    return {
        "reference": (
            "experiments/exp82_evals_contacts_v1_contact_prediction/"
            "data/where_we_stand_summary.csv:marinfold-cv1-exp75-rollout"
        ),
        "tolerance": E8_REFERENCE_TOLERANCE,
        "comparisons": comparisons,
        "passed": all(record["within_tolerance"] for record in comparisons),
    }


def ground_truth_units(path: Path) -> list[tuple[str, str]]:
    """Return ordered evaluation-unit identities from the published JSONL."""

    units = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            units.append((record["dataset"], record["stem"]))
    if len(units) != EXPECTED_UNITS or len(set(units)) != EXPECTED_UNITS:
        raise ValueError("ground truth does not contain 577 distinct evaluation units")
    if len({stem for _, stem in units}) != EXPECTED_UNIQUE_STEMS:
        raise ValueError("ground truth does not contain 575 unique stems")
    return units


def aggregate_subsets(
    precision: pd.DataFrame,
    *,
    ordered_units: list[tuple[str, str]],
    eval2_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate the 577 outputs over the legacy and eval2 reporting cuts."""

    legacy_units = set(ordered_units[:554])
    eval2_units = set(
        zip(eval2_manifest.dataset, eval2_manifest.stem, strict=True)
    )
    natural_units = set(
        zip(
            eval2_manifest.loc[eval2_manifest.designed_any == 0, "dataset"],
            eval2_manifest.loc[eval2_manifest.designed_any == 0, "stem"],
            strict=True,
        )
    )
    strict_manifest = eval2_manifest[eval2_manifest.passes_30 == 1]
    strict_units = set(zip(strict_manifest.dataset, strict_manifest.stem, strict=True))
    strict_natural = strict_manifest[strict_manifest.designed_any == 0]
    strict_natural_units = set(
        zip(strict_natural.dataset, strict_natural.stem, strict=True)
    )
    subsets = {
        "universe_577": set(ordered_units),
        "legacy_554": legacy_units,
        "eval2": eval2_units,
        "eval2_natural": natural_units,
        "eval2_30": strict_units,
        "eval2_natural_30": strict_natural_units,
    }
    expected_counts = {
        "universe_577": 577,
        "legacy_554": 554,
        "eval2": 307,
        "eval2_natural": 78,
        "eval2_30": 275,
        "eval2_natural_30": 61,
    }
    counts = {name: len(units) for name, units in subsets.items()}
    if counts != expected_counts:
        raise ValueError(f"reporting subset counts changed: {counts} != {expected_counts}")
    if not all(units <= subsets["universe_577"] for units in subsets.values()):
        raise ValueError("a reporting subset contains an unknown evaluation unit")

    unit_index = pd.MultiIndex.from_frame(precision[["dataset", "stem"]])
    frames = []
    for name, units in subsets.items():
        mask = unit_index.isin(units)
        frame = (
            precision.loc[mask]
            .groupby(["model", "range", "cut"], as_index=False)
            .agg(
                precision=("precision", "mean"),
                valid_values=("precision", "count"),
                total_values=("precision", "size"),
            )
        )
        frame.insert(0, "subset", name)
        if set(frame.total_values) != {len(units)}:
            raise ValueError(f"metric rows are incomplete for subset {name}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), counts


def finalize(
    *,
    run_root: str,
    score_root: str,
    ground_truth_uri: str,
    eval2_manifest_uri: str,
    worker_sha256: str,
    job_ids: list[str],
    started_at: str,
    model_mirror_run_id: str,
    sampling_seed: int,
    checkpoints: tuple[Checkpoint, ...],
    suite: str,
) -> dict:
    """Validate all outputs, compute metrics, and publish a provenance manifest."""

    filesystem, root = fsspec.core.url_to_fs(run_root)
    scratch = Path("/tmp/exp199_rollout_v2_finalize")
    scratch.mkdir(parents=True, exist_ok=True)
    ground_truth = scratch / "gt_universe.jsonl"
    _, ground_truth_path = fsspec.core.url_to_fs(ground_truth_uri)
    download_file(filesystem, ground_truth_path, ground_truth)
    if sha256_file(ground_truth) != GROUND_TRUTH_SHA256:
        raise ValueError("ground-truth SHA-256 does not match the exp89 universe")
    ordered_units = ground_truth_units(ground_truth)
    _, eval2_manifest_path = fsspec.core.url_to_fs(eval2_manifest_uri)
    eval2_manifest_local = scratch / "eval2_manifest.csv"
    download_file(filesystem, eval2_manifest_path, eval2_manifest_local)
    eval2_manifest = pd.read_csv(eval2_manifest_local)

    score_directories: dict[str, Path] = {}
    timing_frames: list[pd.DataFrame] = []
    validations: dict[str, dict] = {}
    for checkpoint in checkpoints:
        label_root = f"{score_root.rstrip('/')}/{checkpoint.label}"
        _, label_path = fsspec.core.url_to_fs(label_root)
        markers, lengths = completion_records(filesystem, label_path)
        local_scores = scratch / "dense_scores" / checkpoint.label
        dense_root = f"{root}/dense_scores/{checkpoint.label}"
        matrix_count = dense_scores(
            filesystem=filesystem,
            label_root=label_path,
            markers=markers,
            lengths=lengths,
            local_directory=local_scores,
            dense_s3_root=dense_root,
        )
        timings = collect_timings(
            filesystem=filesystem,
            markers=markers,
            expected_units=set(lengths),
        )
        timing_frames.append(timings)
        score_directories[checkpoint.label] = local_scores
        validations[checkpoint.label] = {
            "units": len(lengths),
            "unique_stems": len({stem for _, stem in lengths}),
            "rollouts": int(timings.n_rollouts.sum()),
            "unfinished_rollouts": int(timings.unfinished_rollouts.sum()),
            "dense_matrices": matrix_count,
        }

    timings_all = pd.concat(timing_frames, ignore_index=True)
    timings_path = scratch / "timings.csv"
    timings_all.to_csv(timings_path, index=False)
    filesystem.put_file(str(timings_path), f"{root}/results/timings.csv")

    precision_path, marinfold_path, metric_script_sha256 = run_exp89_metrics(
        scratch=scratch,
        ground_truth=ground_truth,
        score_directories=score_directories,
    )
    precision = pd.read_csv(marinfold_path)
    expected_models = {
        f"marinfold-{checkpoint.label.replace('_', '-')}" for checkpoint in checkpoints
    }
    if set(precision.model) != expected_models:
        raise ValueError(
            f"metric output model identities do not match: {set(precision.model)}"
        )
    counts = precision.groupby("model").size().to_dict()
    if set(counts.values()) != {EXPECTED_UNITS * 20}:
        raise ValueError(f"metric rows are incomplete: {counts}")
    aggregate = (
        precision.groupby(["model", "range", "cut"], as_index=False)
        .agg(
            precision=("precision", "mean"),
            valid_values=("precision", "count"),
            total_values=("precision", "size"),
        )
        .sort_values(["model", "range", "cut"])
    )
    aggregate_path = scratch / "aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    subset_aggregate, subset_counts = aggregate_subsets(
        precision,
        ordered_units=ordered_units,
        eval2_manifest=eval2_manifest,
    )
    subset_aggregate_path = scratch / "subset_aggregate_metrics.csv"
    subset_aggregate.to_csv(subset_aggregate_path, index=False)
    filesystem.put_file(
        str(precision_path), f"{root}/results/contact_precision_all.csv"
    )
    filesystem.put_file(str(marinfold_path), f"{root}/results/marinfold_precision.csv")
    filesystem.put_file(str(aggregate_path), f"{root}/results/aggregate_metrics.csv")
    filesystem.put_file(
        str(subset_aggregate_path),
        f"{root}/results/subset_aggregate_metrics.csv",
    )

    headline = subset_aggregate[
        subset_aggregate["subset"].isin(
            ["legacy_554", "eval2", "eval2_natural", "eval2_30", "eval2_natural_30"]
        )
        & subset_aggregate["range"].isin(["all", "long"])
        & subset_aggregate["cut"].isin(["R", "AUC"])
    ].to_dict(orient="records")
    reference_validation = validate_e8_reference(subset_aggregate, suite)
    if reference_validation is not None:
        write_json(
            filesystem,
            f"{root}/results/e8_reference_validation.json",
            reference_validation,
        )
        if not reference_validation["passed"]:
            raise ValueError(
                "E8 reference validation exceeded the allowed tolerance: "
                f"{reference_validation}"
            )
    reused_existing_coreweave = all(
        checkpoint.coreweave_uri is not None for checkpoint in checkpoints
    )
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "run_root": run_root,
        "checkpoint_source": {
            "identities": [
                {
                    "kind": (
                        "coreweave-s3"
                        if checkpoint.hf_repo_id is None
                        else "huggingface"
                    ),
                    "repo_id": checkpoint.hf_repo_id,
                    "revision": checkpoint.hf_revision,
                    "coreweave_uri": checkpoint.coreweave_uri,
                }
                for checkpoint in checkpoints
            ],
            "transport": (
                "pre-existing CoreWeave S3 checkpoint verified in place against "
                "its pinned file manifest"
                if reused_existing_coreweave
                else "pinned Hugging Face snapshot download on ephemeral CoreWeave "
                "disk, then CoreWeave S3 upload"
            ),
            "checkpoint_copied": not reused_existing_coreweave,
            "gcs_used": False,
            "source_dtypes": sorted({checkpoint.source_dtype for checkpoint in checkpoints}),
            "evaluated_dtype": "bfloat16",
            "mirror_run_id": model_mirror_run_id,
        },
        "checkpoints": [
            {
                "label": checkpoint.label,
                "run_name": checkpoint.run_name,
                "step": checkpoint.step,
                "hf_repo_id": checkpoint.hf_repo_id,
                "hf_revision": checkpoint.hf_revision,
                "coreweave_uri": checkpoint_model_uri(model_mirror_run_id, checkpoint),
                "weight_shard_digests": checkpoint.weight_shard_digests,
                "source_dtype": checkpoint.source_dtype,
                "train_loss": checkpoint.train_loss,
                "eval_loss": checkpoint.eval_loss,
                "wandb_url": f"https://wandb.ai/eric-czech/marin/runs/{checkpoint.run_name}",
                "wandb_metric_keys": {
                    "train_loss": "train/loss",
                    "eval_loss": "eval/loss",
                },
            }
            for checkpoint in checkpoints
        ],
        "sampling": {
            "recipe": "rollout_resample",
            "n_rollouts": 100,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "token_budget": "min(8192-prompt_tokens, 6*L+128)",
            "seed": sampling_seed,
            "tie_break": None,
        },
        "validation": validations,
        "reporting_subset_units": subset_counts,
        "headline_metrics": headline,
        "reference_validation": reference_validation,
        "worker_sha256": worker_sha256,
        "exp89_compute_metrics_sha256": metric_script_sha256,
        "job_ids": job_ids,
        "outputs": {
            "dense_scores": f"{run_root}/dense_scores",
            "timings": f"{run_root}/results/timings.csv",
            "marinfold_precision": f"{run_root}/results/marinfold_precision.csv",
            "contact_precision_all": f"{run_root}/results/contact_precision_all.csv",
            "aggregate_metrics": f"{run_root}/results/aggregate_metrics.csv",
            "subset_aggregate_metrics": (
                f"{run_root}/results/subset_aggregate_metrics.csv"
            ),
        },
    }
    write_json(filesystem, f"{root}/results/run_manifest.json", manifest)
    print(
        json.dumps({"headline_metrics": headline, "validation": validations}, indent=2),
        flush=True,
    )
    return manifest
