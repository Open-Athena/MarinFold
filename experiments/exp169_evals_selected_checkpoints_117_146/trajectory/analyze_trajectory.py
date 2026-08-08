# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Finalize completed checkpoints and build the checkpoint-trajectory data."""

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from sklearn.metrics import roc_auc_score

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT, CheckpointSpec

HERE = Path(__file__).parent
EXPERIMENT = HERE.parent
REPOSITORY = HERE.parents[2]
DATA = EXPERIMENT / "data"
SCRATCH = REPOSITORY / "scratch" / "exp169-checkpoint-trajectories"
EXPECTED_TARGETS = 554
EXPECTED_ROLLOUTS = 100
RANGES = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
CUTS = (
    ("L", lambda length, n_true: length),
    ("L/2", lambda length, n_true: max(1, length // 2)),
    ("L/5", lambda length, n_true: max(1, length // 5)),
    ("R", lambda length, n_true: n_true),
)
STRATA = ("neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length")


def raw_prefix(spec: CheckpointSpec) -> str:
    """Return the immutable raw prefix for a checkpoint evaluation."""

    return f"{HF_BUCKET_ROOT}/runs/{spec.run_name}/step-{spec.step}"


def derived_prefix(spec: CheckpointSpec) -> str:
    """Return the public compact-artifact prefix for a checkpoint."""

    return f"{HF_BUCKET_ROOT}/derived/{spec.run_name}/step-{spec.step}"


def sha256(path: Path) -> str:
    """Hash one finalized local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(fs: HfFileSystem, path: str) -> Any:
    """Read one JSON object from the HF bucket."""

    with fs.open(path, "r") as source:
        return json.load(source)


def read_csv(fs: HfFileSystem, path: str, **kwargs: Any) -> pd.DataFrame:
    """Read one CSV artifact from the HF bucket."""

    with fs.open(path, "rb") as source:
        return pd.read_csv(source, **kwargs)


def read_parquet_parts(fs: HfFileSystem, pattern: str) -> pd.DataFrame:
    """Read parquet parts sequentially to bound local memory and connections."""

    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet parts match {pattern}")
    frames = []
    for path in paths:
        with fs.open(path, "rb") as source:
            frames.append(pq.read_table(source).to_pandas())
    return pd.concat(frames, ignore_index=True)


def completed_target_count(fs: HfFileSystem, spec: CheckpointSpec) -> tuple[int, int]:
    """Return the number of timing parts and durable target rows."""

    paths = sorted(fs.glob(f"{raw_prefix(spec)}/parts/timings-*.parquet"))
    rows = 0
    for path in paths:
        with fs.open(path, "rb") as source:
            rows += pq.read_metadata(source).num_rows
    return len(paths), rows


def load_ground_truth(fs: HfFileSystem, prefix: str) -> list[dict[str, Any]]:
    """Load and validate the fixed contacts-v1 ground-truth universe."""

    with fs.open(f"{prefix}/inputs/gt_universe.jsonl", "r") as source:
        records = [json.loads(line) for line in source]
    keys = [(str(record["dataset"]), str(record["stem"])) for record in records]
    if len(records) != EXPECTED_TARGETS or len(set(keys)) != EXPECTED_TARGETS:
        raise ValueError(
            f"expected {EXPECTED_TARGETS} unique ground-truth units, got "
            f"{len(records)} rows / {len(set(keys))} keys"
        )
    return records


def true_matrix(length: int, contacts: Iterable[Sequence[float]]) -> np.ndarray:
    """Build the binary ground-truth matrix used by the contacts-v1 evals."""

    matrix = np.zeros((length, length), dtype=bool)
    for raw_i, raw_j, degree in contacts:
        i, j = int(raw_i), int(raw_j)
        if float(degree) >= 0.001 and j - i >= 6 and i < j < length:
            matrix[i, j] = True
    return matrix


def metric_rows(score: np.ndarray, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute the established exp89 precision and AUC metrics for one protein."""

    length = int(record["L"])
    resolved = np.asarray(record["resolved"], dtype=np.int64)
    a, b = np.triu_indices(len(resolved), k=1)
    pair_i, pair_j = resolved[a], resolved[b]
    separation = pair_j - pair_i
    scores = score[pair_i, pair_j]
    truth = true_matrix(length, record["contacts"])[pair_i, pair_j].astype(int)

    rows = []
    for range_name, (low, high) in RANGES.items():
        selected = separation >= low
        if high is not None:
            selected &= separation <= high
        range_scores, range_truth = scores[selected], truth[selected]
        n_candidate, n_true = int(range_scores.size), int(range_truth.sum())
        order = np.argsort(-range_scores, kind="mergesort") if n_candidate else None
        ranked_truth = range_truth[order] if n_candidate else None
        for cut, cut_size in CUTS:
            target = int(cut_size(length, n_true))
            if n_candidate == 0 or target <= 0:
                precision, n_top = math.nan, 0
            else:
                n_top = min(target, n_candidate)
                precision = float(ranked_truth[:n_top].sum()) / n_top
            rows.append(
                {
                    "range": range_name,
                    "cut": cut,
                    "precision": precision,
                    "n_candidate": n_candidate,
                    "n_true": n_true,
                    "n_top": n_top,
                }
            )
        auc = (
            float(roc_auc_score(range_truth, range_scores))
            if n_candidate and 0 < n_true < n_candidate
            else math.nan
        )
        rows.append(
            {
                "range": range_name,
                "cut": "AUC",
                "precision": auc,
                "n_candidate": n_candidate,
                "n_true": n_true,
                "n_top": n_candidate,
            }
        )
    return rows


def validate_timings(timings: pd.DataFrame) -> set[tuple[str, str]]:
    """Require one successful completion marker for every evaluation target."""

    if len(timings) != EXPECTED_TARGETS:
        raise ValueError(f"expected {EXPECTED_TARGETS} timing rows, got {len(timings)}")
    keys = list(zip(timings["dataset"], timings["stem"], strict=True))
    unique = set(keys)
    if len(unique) != EXPECTED_TARGETS:
        raise ValueError(
            f"expected {EXPECTED_TARGETS} unique timing keys, got {len(unique)}"
        )
    if not timings["complete"].all():
        raise ValueError("timing table contains incomplete targets")
    if not (timings["n_rollouts"] == EXPECTED_ROLLOUTS).all():
        raise ValueError("timing table contains a non-production rollout count")
    return {(str(dataset), str(stem)) for dataset, stem in unique}


def sparse_triplets(
    votes: pd.DataFrame, timing_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    """Validate sparse votes and group them by evaluation target."""

    triplets: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int, int]] = set()
    for row in votes.itertuples(index=False):
        key = (str(row.dataset), str(row.stem))
        pair = (*key, int(row.i), int(row.j))
        if key not in timing_keys:
            raise ValueError(f"vote row has no completion marker: {key}")
        if pair in seen:
            raise ValueError(f"duplicate sparse vote pair: {pair}")
        value = int(row.votes)
        if not 1 <= value <= EXPECTED_ROLLOUTS:
            raise ValueError(f"invalid vote count for {pair}: {value}")
        seen.add(pair)
        triplets[key].append((int(row.i), int(row.j), value))
    return triplets


def score_targets(
    ground_truth: Sequence[dict[str, Any]],
    triplets: dict[tuple[str, str], list[tuple[int, int, int]]],
    timing_keys: set[tuple[str, str]],
    spec: CheckpointSpec,
) -> pd.DataFrame:
    """Reconstruct one score matrix at a time and return tidy metric rows."""

    rows: list[dict[str, Any]] = []
    for record in ground_truth:
        key = (str(record["dataset"]), str(record["stem"]))
        if key not in timing_keys:
            raise ValueError(f"missing completed target: {key}")
        length = int(record["L"])
        matrix = np.zeros((length, length), dtype=np.int16)
        for i, j, value in triplets.get(key, []):
            if not 0 <= i < j < length:
                raise ValueError(f"vote pair outside {key} length {length}: {(i, j)}")
            matrix[i, j] = matrix[j, i] = value
        strata = record.get("strata", {}) or {}
        base = {
            "dataset": key[0],
            "stem": key[1],
            "n_residues": length,
            "model": f"{spec.run_name}-step-{spec.step}",
            "checkpoint": spec.key,
            "run_key": spec.run_key,
            "model_label": spec.model_label,
            "epoch": spec.epoch,
            "step": spec.step,
            "training_tokens": spec.training_tokens,
            "validation_loss": spec.validation_loss,
            "mode": "single_seq",
            "predictor": "lm",
            **{name: strata.get(name) for name in STRATA},
        }
        rows.extend({**base, **metric} for metric in metric_rows(matrix, record))
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics while retaining all per-protein rows separately."""

    group = [
        "model",
        "checkpoint",
        "run_key",
        "model_label",
        "epoch",
        "step",
        "training_tokens",
        "validation_loss",
        "range",
        "cut",
    ]
    summary = (
        rows.groupby(group, sort=False)["precision"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "precision_mean",
                "median": "precision_median",
                "std": "precision_std",
            }
        )
    )
    summary["precision_sem"] = summary["precision_std"] / np.sqrt(summary["n"])
    range_order = {name: index for index, name in enumerate(RANGES)}
    cut_order = {name: index for index, (name, _) in enumerate(CUTS)} | {
        "AUC": len(CUTS)
    }
    summary["_range_order"] = summary["range"].map(range_order)
    summary["_cut_order"] = summary["cut"].map(cut_order)
    return summary.sort_values(["_range_order", "_cut_order"]).drop(
        columns=["_range_order", "_cut_order"]
    )


def upload(fs: HfFileSystem, path: Path, prefix: str) -> None:
    """Upload one artifact with one connection."""

    with (
        path.open("rb") as source,
        fs.open(f"{prefix}/{path.name}", "wb") as destination,
    ):
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def finalized_paths(spec: CheckpointSpec) -> dict[str, Path]:
    """Return scratch paths for one checkpoint's compact artifacts."""

    root = SCRATCH / "finalized" / spec.key
    root.mkdir(parents=True, exist_ok=True)
    return {
        "rows": root / "metric_rows.csv.gz",
        "summary": root / "summary.csv",
        "timings": root / "timings.csv.gz",
        "manifest": root / "manifest.json",
    }


def finalize_checkpoint(
    fs: HfFileSystem, spec: CheckpointSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate one complete raw checkpoint evaluation and publish derivatives."""

    source = raw_prefix(spec)
    manifest = read_json(fs, f"{source}/manifest.json")
    if (
        manifest.get("n_targets") != EXPECTED_TARGETS
        or manifest.get("n_rollouts") != EXPECTED_ROLLOUTS
    ):
        raise ValueError(f"raw run is not a full production evaluation: {source}")
    if manifest.get("checkpoint") != asdict(spec):
        raise ValueError(f"raw checkpoint manifest does not match catalog: {source}")

    timings = read_parquet_parts(fs, f"{source}/parts/timings-*.parquet")
    timing_keys = validate_timings(timings)
    votes = read_parquet_parts(fs, f"{source}/parts/votes-*.parquet")
    triplets = sparse_triplets(votes, timing_keys)
    rows = score_targets(load_ground_truth(fs, source), triplets, timing_keys, spec)
    summary = summarize(rows)

    paths = finalized_paths(spec)
    rows.to_csv(paths["rows"], index=False, compression="gzip")
    summary.to_csv(paths["summary"], index=False)
    timings.to_csv(paths["timings"], index=False, compression="gzip")
    artifact_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_prefix": f"hf://{source}",
        "derived_prefix": f"hf://{derived_prefix(spec)}",
        "checkpoint": asdict(spec),
        "source_manifest": manifest,
        "validation": {
            "n_targets": len(timing_keys),
            "n_rollouts": EXPECTED_ROLLOUTS,
            "n_sparse_vote_rows": len(votes),
            "n_metric_rows": len(rows),
        },
        "metric_recipe": {
            "ranges": RANGES,
            "cuts": [name for name, _ in CUTS] + ["AUC"],
            "minimum_contact_degree": 0.001,
            "minimum_sequence_separation": 6,
            "ranking": "descending integer rollout votes; stable mergesort for ties",
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
    )
    destination = derived_prefix(spec)
    for path in paths.values():
        upload(fs, path, destination)
    print(
        f"[finalize] {spec.key}: {len(votes):,} votes -> hf://{destination}",
        flush=True,
    )
    return rows, summary, timings


def load_finalized(
    fs: HfFileSystem, spec: CheckpointSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load an existing validated derivative if it matches the catalog."""

    prefix = derived_prefix(spec)
    manifest_path = f"{prefix}/manifest.json"
    fs.invalidate_cache(manifest_path)
    if not fs.exists(manifest_path):
        return None
    manifest = read_json(fs, manifest_path)
    if manifest.get("checkpoint") != asdict(spec):
        raise ValueError(
            f"derived checkpoint manifest does not match catalog: {prefix}"
        )
    rows = read_csv(fs, f"{prefix}/metric_rows.csv.gz", compression="gzip")
    summary = read_csv(fs, f"{prefix}/summary.csv")
    timings = read_csv(fs, f"{prefix}/timings.csv.gz", compression="gzip")
    if len(rows) != EXPECTED_TARGETS * len(RANGES) * (len(CUTS) + 1):
        raise ValueError(f"unexpected derived metric row count: {prefix}")
    validate_timings(timings)
    return rows, summary, timings


def trajectory_row(
    spec: CheckpointSpec, summary: pd.DataFrame, timings: pd.DataFrame
) -> dict[str, Any]:
    """Build one plot-ready checkpoint row from its validated derivatives."""

    total_rollouts = len(timings) * EXPECTED_ROLLOUTS
    stopped_rollouts = int(timings["stopped_rollouts"].sum())
    row: dict[str, Any] = {
        "run_key": spec.run_key,
        "run_name": spec.run_name,
        "model_label": spec.model_label,
        "checkpoint": spec.key,
        "epoch": spec.epoch,
        "step": spec.step,
        "training_tokens": spec.training_tokens,
        "training_tokens_billions": spec.training_tokens / 1e9,
        "validation_loss": spec.validation_loss,
        "checkpoint_uri": spec.checkpoint_uri,
        "hf_raw_prefix": f"hf://{raw_prefix(spec)}",
        "hf_derived_prefix": f"hf://{derived_prefix(spec)}",
        "accelerator": ",".join(sorted(set(timings["gpu_name"].astype(str)))),
        "inference_seconds": float(timings["elapsed_seconds"].sum()),
        "checkpoint_prepare_seconds": float(
            timings["checkpoint_prepare_seconds"].sum()
        ),
        "model_load_seconds": float(timings["model_load_seconds"].sum()),
        "generated_tokens": int(timings["generated_tokens"].sum()),
        "total_rollouts": total_rollouts,
        "stopped_rollouts": stopped_rollouts,
        "truncated_rollouts": total_rollouts - stopped_rollouts,
        "stop_rate": stopped_rollouts / total_rollouts,
    }
    r_rows = summary[summary["cut"] == "R"].set_index("range")
    for range_name in RANGES:
        metric = r_rows.loc[range_name]
        row[f"r_precision_{range_name}"] = float(metric["precision_mean"])
        row[f"r_precision_{range_name}_sem"] = float(metric["precision_sem"])
    return row


def paired_changes(rows: pd.DataFrame) -> pd.DataFrame:
    """Build paired per-protein R-precision changes between adjacent checkpoints."""

    r_rows = rows[rows["cut"] == "R"]
    available = set(r_rows["checkpoint"])
    output = []
    for run_key in dict.fromkeys(spec.run_key for spec in CHECKPOINTS.values()):
        specs = sorted(
            (spec for spec in CHECKPOINTS.values() if spec.run_key == run_key),
            key=lambda spec: spec.training_tokens,
        )
        for before, after in pairwise(specs):
            if before.key not in available or after.key not in available:
                continue
            before_rows = r_rows[r_rows["checkpoint"] == before.key][
                ["dataset", "stem", "range", "precision"]
            ]
            after_rows = r_rows[r_rows["checkpoint"] == after.key][
                ["dataset", "stem", "range", "precision"]
            ]
            paired = before_rows.merge(
                after_rows,
                on=["dataset", "stem", "range"],
                how="inner",
                validate="one_to_one",
                suffixes=("_before", "_after"),
            )
            if len(paired) != EXPECTED_TARGETS * len(RANGES):
                raise ValueError(
                    f"unexpected paired row count for {before.key} -> {after.key}: "
                    f"{len(paired)}"
                )
            paired["delta"] = paired["precision_after"] - paired["precision_before"]
            for range_name in RANGES:
                delta = paired.loc[paired["range"] == range_name, "delta"].dropna()
                n = len(delta)
                mean = float(delta.mean())
                std = float(delta.std(ddof=1))
                sem = std / math.sqrt(n)
                output.append(
                    {
                        "run_key": run_key,
                        "model_label": before.model_label,
                        "from_epoch": before.epoch,
                        "to_epoch": after.epoch,
                        "from_step": before.step,
                        "to_step": after.step,
                        "from_training_tokens": before.training_tokens,
                        "to_training_tokens": after.training_tokens,
                        "from_validation_loss": before.validation_loss,
                        "to_validation_loss": after.validation_loss,
                        "delta_validation_loss": (
                            after.validation_loss - before.validation_loss
                        ),
                        "range": range_name,
                        "n": n,
                        "delta_r_precision_mean": mean,
                        "delta_r_precision_std": std,
                        "delta_r_precision_sem": sem,
                        "delta_r_precision_ci95_low": mean - 1.96 * sem,
                        "delta_r_precision_ci95_high": mean + 1.96 * sem,
                        "win_rate": float((delta > 0).mean()),
                        "tie_rate": float((delta == 0).mean()),
                    }
                )
    return pd.DataFrame(output)


def matched_token_changes(rows: pd.DataFrame) -> pd.DataFrame:
    """Build paired R-precision differences at shared nominal token budgets."""

    r_rows = rows[rows["cut"] == "R"]
    available = set(r_rows["checkpoint"])
    specs_by_epoch: dict[int, list[CheckpointSpec]] = defaultdict(list)
    for spec in CHECKPOINTS.values():
        if spec.key in available:
            specs_by_epoch[spec.epoch].append(spec)

    output = []
    for epoch, specs in sorted(specs_by_epoch.items()):
        for left, right in combinations(specs, 2):
            if left.run_key == right.run_key:
                continue
            left_rows = r_rows[r_rows["checkpoint"] == left.key][
                ["dataset", "stem", "range", "precision"]
            ]
            right_rows = r_rows[r_rows["checkpoint"] == right.key][
                ["dataset", "stem", "range", "precision"]
            ]
            paired = left_rows.merge(
                right_rows,
                on=["dataset", "stem", "range"],
                how="inner",
                validate="one_to_one",
                suffixes=("_left", "_right"),
            )
            if len(paired) != EXPECTED_TARGETS * len(RANGES):
                raise ValueError(
                    f"unexpected matched row count for {left.key} vs {right.key}: "
                    f"{len(paired)}"
                )
            paired["delta"] = paired["precision_left"] - paired["precision_right"]
            for range_name in RANGES:
                delta = paired.loc[paired["range"] == range_name, "delta"].dropna()
                n = len(delta)
                mean = float(delta.mean())
                std = float(delta.std(ddof=1))
                sem = std / math.sqrt(n)
                output.append(
                    {
                        "epoch": epoch,
                        "left_run_key": left.run_key,
                        "left_model_label": left.model_label,
                        "left_step": left.step,
                        "left_training_tokens": left.training_tokens,
                        "left_validation_loss": left.validation_loss,
                        "right_run_key": right.run_key,
                        "right_model_label": right.model_label,
                        "right_step": right.step,
                        "right_training_tokens": right.training_tokens,
                        "right_validation_loss": right.validation_loss,
                        "range": range_name,
                        "n": n,
                        "delta_r_precision_mean": mean,
                        "delta_r_precision_std": std,
                        "delta_r_precision_sem": sem,
                        "delta_r_precision_ci95_low": mean - 1.96 * sem,
                        "delta_r_precision_ci95_high": mean + 1.96 * sem,
                        "left_win_rate": float((delta > 0).mean()),
                        "tie_rate": float((delta == 0).mean()),
                    }
                )
    return pd.DataFrame(output)


def write_aggregate(
    fs: HfFileSystem,
    status_rows: list[dict[str, Any]],
    rows: list[pd.DataFrame],
    summaries: list[pd.DataFrame],
    timings: list[pd.DataFrame],
) -> None:
    """Write and publish the incrementally complete trajectory artifacts."""

    DATA.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame(status_rows)
    status_path = DATA / "trajectory_checkpoint_status.csv"
    status.to_csv(status_path, index=False)
    output_paths = [status_path]
    if summaries:
        checkpoint_table = pd.DataFrame(
            [
                trajectory_row(
                    CHECKPOINTS[str(summary.iloc[0]["checkpoint"])], summary, timing
                )
                for summary, timing in zip(summaries, timings, strict=True)
            ]
        ).sort_values(["run_key", "training_tokens"])
        table_path = DATA / "trajectory_checkpoint_metrics.csv"
        rows_path = DATA / "trajectory_metric_rows.csv.gz"
        timings_path = DATA / "trajectory_timings.csv.gz"
        paired_path = DATA / "trajectory_paired_changes.csv"
        matched_path = DATA / "trajectory_matched_token_changes.csv"
        checkpoint_table.to_csv(table_path, index=False)
        metric_rows = pd.concat(rows, ignore_index=True)
        metric_rows.to_csv(rows_path, index=False, compression="gzip")
        pd.concat(timings, ignore_index=True).to_csv(
            timings_path, index=False, compression="gzip"
        )
        paired_changes(metric_rows).to_csv(paired_path, index=False)
        matched_token_changes(metric_rows).to_csv(matched_path, index=False)
        output_paths.extend(
            (table_path, paired_path, matched_path, rows_path, timings_path)
        )

    manifest_path = DATA / "trajectory_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "hf_prefix": f"hf://{HF_BUCKET_ROOT}",
        "selected_checkpoints": [asdict(spec) for spec in CHECKPOINTS.values()],
        "n_complete": len(summaries),
        "n_selected": len(CHECKPOINTS),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    output_paths.append(manifest_path)
    summary_prefix = f"{HF_BUCKET_ROOT}/summary"
    for path in output_paths:
        upload(fs, path, summary_prefix)
    print(
        f"[aggregate] {len(summaries)}/{len(CHECKPOINTS)} checkpoints -> "
        f"hf://{summary_prefix}",
        flush=True,
    )


def run(status_only: bool) -> int:
    """Finalize every newly completed checkpoint and refresh aggregate outputs."""

    token = os.environ.get("HF_TOKEN")
    if not status_only and not token:
        raise ValueError("HF_TOKEN must contain the open-athena write token")
    fs = HfFileSystem(token=token or False)
    status_rows: list[dict[str, Any]] = []
    all_rows: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_timings: list[pd.DataFrame] = []
    for spec in CHECKPOINTS.values():
        parts, targets = completed_target_count(fs, spec)
        state = (
            "complete"
            if targets == EXPECTED_TARGETS
            else "running"
            if targets
            else "pending"
        )
        status_rows.append(
            {
                "run_key": spec.run_key,
                "model_label": spec.model_label,
                "checkpoint": spec.key,
                "region": spec.region,
                "epoch": spec.epoch,
                "step": spec.step,
                "training_tokens": spec.training_tokens,
                "validation_loss": spec.validation_loss,
                "checkpoint_uri": spec.checkpoint_uri,
                "hf_raw_prefix": f"hf://{raw_prefix(spec)}",
                "timing_parts": parts,
                "durable_targets": targets,
                "state": state,
            }
        )
        print(
            f"[status] {spec.key}: {targets}/{EXPECTED_TARGETS} targets ({parts} parts)"
        )
        if status_only or targets != EXPECTED_TARGETS:
            continue
        finalized = load_finalized(fs, spec)
        if finalized is None:
            finalized = finalize_checkpoint(fs, spec)
        rows, summary, timings = finalized
        timings = timings.assign(
            checkpoint=spec.key,
            run_key=spec.run_key,
            model_label=spec.model_label,
            epoch=spec.epoch,
            step=spec.step,
            training_tokens=spec.training_tokens,
            validation_loss=spec.validation_loss,
        )
        all_rows.append(rows)
        all_summaries.append(summary)
        all_timings.append(timings)
    if status_only:
        pd.DataFrame(status_rows).to_csv(SCRATCH / "checkpoint_status.csv", index=False)
        return 0
    write_aggregate(fs, status_rows, all_rows, all_summaries, all_timings)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse status-only or finalizing analysis mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    raise SystemExit(run(args.status_only))
