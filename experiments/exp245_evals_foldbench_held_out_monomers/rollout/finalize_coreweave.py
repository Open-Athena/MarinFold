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
    EVAL_SETS,
    EXPECTED_SET_SIZES,
    EXPECTED_UNITS,
    GROUND_TRUTH_SHA256,
    Checkpoint,
    checkpoint_model_uri,
)

EXPECTED_ROLLOUTS_PER_UNIT = 100
PR234_WORKER_SHA256 = "f28829f9826a7089f082cb55de45582c7cbc389ea3d700e5c5869ff29bb6bc82"
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
    usable_total = 0
    unfinished_total = 0
    for marker in markers:
        marker_unfinished = marker["unfinished_rollouts"]
        if marker_unfinished != 0 and not marker.get("accepted_unfinished", False):
            raise ValueError(f"unfinished rollouts in completion marker: {marker}")
        rollout_total += marker["total_rollouts"]
        unfinished_total += marker_unfinished
        usable_total += marker.get(
            "usable_rollouts", marker["total_rollouts"] - marker_unfinished
        )
        for unit in marker["units"]:
            key = (unit["dataset"], unit["stem"])
            if key in lengths:
                raise ValueError(f"duplicate completed unit: {key}")
            if unit["n_rollouts"] != EXPECTED_ROLLOUTS_PER_UNIT:
                raise ValueError(f"wrong rollout count for {key}: {unit['n_rollouts']}")
            unit_unfinished = unit.get("unfinished_rollouts", 0)
            unit_usable = unit.get(
                "usable_rollouts", unit["n_rollouts"] - unit_unfinished
            )
            if unit_usable + unit_unfinished != EXPECTED_ROLLOUTS_PER_UNIT:
                raise ValueError(f"usable/requested rollout mismatch for {key}: {unit}")
            lengths[key] = unit["L"]
    if len(lengths) != EXPECTED_UNITS:
        raise ValueError(
            f"incomplete units under {label_root}: {len(lengths)} != {EXPECTED_UNITS}"
        )
    if len({stem for _, stem in lengths}) != EXPECTED_UNITS:
        raise ValueError("unique-stem count does not match the evaluation universe")
    expected_rollouts = EXPECTED_UNITS * EXPECTED_ROLLOUTS_PER_UNIT
    if rollout_total != expected_rollouts:
        raise ValueError(f"rollout total {rollout_total} != {expected_rollouts}")
    if usable_total + unfinished_total != expected_rollouts:
        raise ValueError("usable and unfinished rollout totals do not cover the run")
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
    if set(timings.n_rollouts) != {EXPECTED_ROLLOUTS_PER_UNIT}:
        raise ValueError("timing records have an incorrect rollout count")
    if not (
        timings.stopped_rollouts + timings.unfinished_rollouts == timings.n_rollouts
    ).all():
        raise ValueError("usable and unfinished timing counts do not cover each unit")
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


def ground_truth_units(path: Path) -> list[tuple[str, str]]:
    """Return ordered evaluation-unit identities from the published JSONL."""

    units = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            units.append((record["dataset"], record["stem"]))
    if len(units) != EXPECTED_UNITS or len(set(units)) != EXPECTED_UNITS:
        raise ValueError(
            f"ground truth does not contain {EXPECTED_UNITS} distinct units"
        )
    if len({stem for _, stem in units}) != EXPECTED_UNITS:
        # Unlike the 577-unit universe, every FoldBench monomer appears once
        # under one dataset label, so units and stems are in bijection here.
        raise ValueError("ground truth stems are not unique")
    return units


def aggregate_subsets(
    precision: pd.DataFrame,
    *,
    ordered_units: list[tuple[str, str]],
    sets_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate the 334 outputs over exp245's reporting cuts.

    Nine cuts: the whole universe, each of the three eval sets, and the viral /
    non-viral split of each. The viral cells are small (6 and 13 proteins) and
    are reported because #241 showed the two strata rank models differently, not
    because either alone is a headline number.
    """

    by_unit = {
        (row.dataset, row.stem): row
        for row in sets_manifest.itertuples()
    }
    missing = [unit for unit in ordered_units if unit not in by_unit]
    if missing:
        raise ValueError(f"units absent from the set manifest: {missing[:5]}")

    subsets: dict[str, set[tuple[str, str]]] = {"universe": set(ordered_units)}
    for name in EVAL_SETS:
        members = {u for u in ordered_units if by_unit[u].eval_set == name}
        subsets[name] = members
        subsets[f"{name}-viral"] = {u for u in members if by_unit[u].is_viral == 1}
        subsets[f"{name}-nonviral"] = {u for u in members if by_unit[u].is_viral == 0}

    counts = {name: len(units) for name, units in subsets.items()}
    expected = {"universe": EXPECTED_UNITS, **EXPECTED_SET_SIZES}
    for name, size in expected.items():
        if counts[name] != size:
            raise ValueError(f"reporting subset {name} has {counts[name]}, not {size}")
    for name in EVAL_SETS:
        if counts[f"{name}-viral"] + counts[f"{name}-nonviral"] != counts[name]:
            raise ValueError(f"{name} viral split does not partition the set")
    if not all(units <= subsets["universe"] for units in subsets.values()):
        raise ValueError("a reporting subset contains an unknown evaluation unit")

    unit_index = pd.MultiIndex.from_frame(precision[["dataset", "stem"]])
    frames = []
    for name, units in subsets.items():
        if not units:  # an empty viral cell would otherwise emit no rows at all
            continue
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
    sets_manifest_uri: str,
    worker_sha256: str,
    job_ids: list[str],
    started_at: str,
    model_mirror_run_id: str,
    sampling_seed: int,
    contact_mult: int,
    checkpoints: tuple[Checkpoint, ...],
    suite: str,
) -> dict:
    """Validate all outputs, compute metrics, and publish a provenance manifest."""

    filesystem, root = fsspec.core.url_to_fs(run_root)
    scratch = Path("/tmp/exp245_rollout_finalize")
    scratch.mkdir(parents=True, exist_ok=True)
    ground_truth = scratch / "gt_universe.jsonl"
    _, ground_truth_path = fsspec.core.url_to_fs(ground_truth_uri)
    download_file(filesystem, ground_truth_path, ground_truth)
    if sha256_file(ground_truth) != GROUND_TRUTH_SHA256:
        raise ValueError("ground-truth SHA-256 does not match the published universe")
    ordered_units = ground_truth_units(ground_truth)
    _, sets_manifest_path = fsspec.core.url_to_fs(sets_manifest_uri)
    sets_manifest_local = scratch / "eval_sets.csv"
    download_file(filesystem, sets_manifest_path, sets_manifest_local)
    sets_manifest = pd.read_csv(sets_manifest_local)
    sets_manifest = sets_manifest[sets_manifest.scorable == 1]
    # The manifest is keyed by stem; the metric rows are keyed by (dataset,
    # stem), and every monomer carries the one dataset label this eval uses.
    sets_manifest["dataset"] = ordered_units[0][0]

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
        unfinished_rollouts = int(timings.unfinished_rollouts.sum())
        if unfinished_rollouts != checkpoint.accepted_unfinished_rollouts:
            raise ValueError(
                f"unfinished rollout count changed for {checkpoint.label}: "
                f"{unfinished_rollouts} != {checkpoint.accepted_unfinished_rollouts}"
            )
        validations[checkpoint.label] = {
            "units": len(lengths),
            "unique_stems": len({stem for _, stem in lengths}),
            "requested_rollouts": int(timings.n_rollouts.sum()),
            "usable_rollouts": int(timings.stopped_rollouts.sum()),
            "unfinished_rollouts": unfinished_rollouts,
            "affected_units": int((timings.unfinished_rollouts > 0).sum()),
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
        sets_manifest=sets_manifest,
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
        subset_aggregate["subset"].isin(list(EVAL_SETS))
        & subset_aggregate["range"].isin(["all", "long"])
        & subset_aggregate["cut"].isin(["R", "AUC"])
    ].to_dict(orient="records")
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
                    "kind": "coreweave-s3",
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
            "source_dtypes": sorted(
                {checkpoint.source_dtype for checkpoint in checkpoints}
            ),
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
                "accepted_unfinished_rollouts": (
                    checkpoint.accepted_unfinished_rollouts
                ),
                "wandb_url": (
                    f"https://wandb.ai/open-athena/MarinFold/runs/{checkpoint.run_name}"
                    if checkpoint.run_name.startswith(("prot-exp232", "prot-exp199-cw"))
                    else f"https://wandb.ai/eric-czech/marin/runs/{checkpoint.run_name}"
                ),
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
            "token_budget": f"min(8192-prompt_tokens, {contact_mult}*L+128)",
            "contact_multiplier": contact_mult,
            "unfinished_policy": (
                "Nonterminating samples are excluded from contact voting. Every "
                "requested and usable rollout is counted per evaluation unit."
            ),
            "seed": sampling_seed,
            "tie_break": None,
        },
        "validation": validations,
        "reporting_subset_units": subset_counts,
        "headline_metrics": headline,
        "reference_validation": reference_validation,
        "worker_sha256": worker_sha256,
        "score_worker_provenance": {
            "standard_worker_sha256": PR234_WORKER_SHA256,
            "standard_worker_scope": (
                "All E8 and m2-p06 parts, plus m1-p02 shards 0-8 and 11."
            ),
            "accepted_unfinished_worker_sha256": worker_sha256,
            "accepted_unfinished_worker_scope": "m1-p02 shards 9 and 10.",
            "semantic_difference": (
                "The accepted-unfinished worker omits nonterminating samples from "
                "contact voting and records requested, usable, and unfinished counts. "
                "Its voting behavior is unchanged for terminating samples."
            ),
        },
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
