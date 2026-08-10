# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and finalize one complete exp199 contact evaluation."""

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from sklearn.metrics import roc_auc_score

from checkpoint_specs import (
    CHECKPOINTS,
    GROUND_TRUTH_SHA256,
    HF_BUCKET_ROOT,
    checkpoint_manifest,
    validate_run_tag,
)

HERE = Path(__file__).parent
EXPERIMENT = HERE.parent.parent
DATA = EXPERIMENT / "data"
EXPECTED_TARGETS = 554
EXPECTED_ROLLOUTS = 100
TARGETS_SHA256 = "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
PR190_CONTROL_PREFIX = (
    "buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp166/"
    "scores/exp117-control-step-35679"
)
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


def artifact_root(run_tag: str | None = None) -> str:
    """Return the standard or isolated replicate artifact root."""

    run_tag = validate_run_tag(run_tag)
    if run_tag is None:
        return HF_BUCKET_ROOT
    return f"{HF_BUCKET_ROOT}/replicates/{run_tag}"


def raw_prefix(checkpoint: str, run_tag: str | None = None) -> str:
    """Return the immutable input prefix for a checkpoint evaluation."""

    spec = CHECKPOINTS[checkpoint]
    return f"{artifact_root(run_tag)}/runs/{spec.run_name}/step-{spec.step}"


def derived_prefix(checkpoint: str, run_tag: str | None = None) -> str:
    """Return the public derived-artifact prefix for a checkpoint evaluation."""

    spec = CHECKPOINTS[checkpoint]
    return f"{artifact_root(run_tag)}/derived/{spec.run_name}/step-{spec.step}"


def output_paths(checkpoint: str, run_tag: str | None = None) -> dict[str, Path]:
    """Return the small, commit-ready outputs for one checkpoint."""

    stem = CHECKPOINTS[checkpoint].key.replace("-", "_")
    if run_tag is not None:
        stem = f"{stem}_{run_tag.replace('-', '_')}"
    return {
        "rows": DATA / f"contact_eval_{stem}_rows.csv.gz",
        "summary": DATA / f"contact_eval_{stem}_summary.csv",
        "timings": DATA / f"contact_eval_{stem}_timings.csv.gz",
        "manifest": DATA / f"contact_eval_{stem}_manifest.json",
    }


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
    checkpoint: str,
) -> pd.DataFrame:
    """Reconstruct one score matrix at a time and return tidy metric rows."""

    spec = CHECKPOINTS[checkpoint]
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
            "checkpoint": checkpoint,
            "mode": "single_seq",
            "predictor": "lm",
            **{name: strata.get(name) for name in STRATA},
        }
        rows.extend({**base, **metric} for metric in metric_rows(matrix, record))
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics while retaining all per-protein rows separately."""

    summary = (
        rows.groupby(["model", "checkpoint", "range", "cut"], sort=False)["precision"]
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


def all_range_r_precision(rows: pd.DataFrame) -> float:
    """Return the unrounded mean all-range R-precision over all proteins."""

    values = rows.loc[
        (rows["range"] == "all") & (rows["cut"] == "R"), "precision"
    ].to_numpy(dtype=np.float64)
    if len(values) != EXPECTED_TARGETS or not np.isfinite(values).all():
        raise ValueError(
            f"expected {EXPECTED_TARGETS} finite all-range R values, got "
            f"{len(values)} values / {np.isfinite(values).sum()} finite"
        )
    return float(values.mean())


def upload(fs: HfFileSystem, path: Path, prefix: str) -> None:
    """Upload one derived artifact with one connection."""

    destination_path = f"{prefix}/{path.name}"
    with (
        path.open("rb") as source,
        fs.open(destination_path, "wb") as destination,
    ):
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def verify_pr190_control(token: str | None = None) -> dict[str, float]:
    """Reproduce PR #190's control metrics from its archived lossless votes."""

    fs = HfFileSystem(token=token)
    manifest = read_json(fs, f"{PR190_CONTROL_PREFIX}/manifest.json")
    spec = CHECKPOINTS["exp117-control-step35679"]
    expected_checkpoint = {
        "repo_id": spec.hf_repo_id,
        "subfolder": spec.hf_subfolder,
    }
    observed_checkpoint = {
        key: manifest.get("checkpoint", {}).get(key) for key in expected_checkpoint
    }
    if observed_checkpoint != expected_checkpoint:
        raise ValueError(f"PR #190 checkpoint mismatch: {observed_checkpoint!r}")
    if (
        manifest.get("n_targets") != EXPECTED_TARGETS
        or manifest.get("n_rollouts") != EXPECTED_ROLLOUTS
        or manifest.get("targets_sha256") != TARGETS_SHA256
        or manifest.get("ground_truth_sha256") != GROUND_TRUTH_SHA256
    ):
        raise ValueError("PR #190 source manifest does not match the fixed eval")

    ground_truth = load_ground_truth(fs, PR190_CONTROL_PREFIX)
    timings = read_parquet_parts(fs, f"{PR190_CONTROL_PREFIX}/parts/timings-*.parquet")
    timing_keys = validate_timings(timings)
    votes = read_parquet_parts(fs, f"{PR190_CONTROL_PREFIX}/parts/votes-*.parquet")
    rows = score_targets(
        ground_truth,
        sparse_triplets(votes, timing_keys),
        timing_keys,
        "exp117-control-step35679",
    )
    values = {
        range_name: float(
            rows.loc[
                (rows["range"] == range_name) & (rows["cut"] == "R"),
                "precision",
            ].mean()
        )
        for range_name in RANGES
    }
    if values["all"] != spec.reference_r_all:
        raise ValueError(
            f"PR #190 R-all changed: {values['all']!r} != {spec.reference_r_all!r}"
        )
    for range_name, value in values.items():
        print(f"[pr190] R-{range_name}: {value!r}", flush=True)
    return values


def finalize(
    checkpoint: str, token: str, run_tag: str | None = None
) -> dict[str, Path]:
    """Validate a full run, write local artifacts, and publish them to HF."""

    fs = HfFileSystem(token=token)
    source = raw_prefix(checkpoint, run_tag)
    manifest = read_json(fs, f"{source}/manifest.json")
    if (
        manifest.get("n_targets") != EXPECTED_TARGETS
        or manifest.get("n_rollouts") != EXPECTED_ROLLOUTS
    ):
        raise ValueError(
            f"source manifest is not a complete production evaluation: {manifest}"
        )
    spec = CHECKPOINTS[checkpoint]
    if manifest.get("checkpoint") != checkpoint_manifest(spec):
        raise ValueError(
            "source manifest checkpoint does not match the selected catalog entry"
        )

    ground_truth = load_ground_truth(fs, source)
    timings = read_parquet_parts(fs, f"{source}/parts/timings-*.parquet")
    timing_keys = validate_timings(timings)
    votes = read_parquet_parts(fs, f"{source}/parts/votes-*.parquet")
    triplets = sparse_triplets(votes, timing_keys)
    rows = score_targets(ground_truth, triplets, timing_keys, checkpoint)
    summary = summarize(rows)
    observed_r_all = all_range_r_precision(rows)
    reference_validation = None
    if spec.reference_r_all is not None and spec.reference_tolerance is not None:
        delta = observed_r_all - spec.reference_r_all
        reference_validation = {
            "expected_r_all": spec.reference_r_all,
            "observed_r_all": observed_r_all,
            "delta": delta,
            "absolute_tolerance": spec.reference_tolerance,
            "passed": abs(delta) <= spec.reference_tolerance,
        }
        if not reference_validation["passed"]:
            raise ValueError(
                f"control R-all {observed_r_all!r} differs from reference "
                f"{spec.reference_r_all!r} by {delta!r}, exceeding "
                f"tolerance {spec.reference_tolerance!r}"
            )
    print(f"[finalize] all-range R-precision: {observed_r_all!r}", flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    paths = output_paths(checkpoint, run_tag)
    rows.to_csv(paths["rows"], index=False, compression="gzip")
    summary.to_csv(paths["summary"], index=False)
    timings.to_csv(paths["timings"], index=False, compression="gzip")
    artifact_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_prefix": f"hf://{source}",
        "derived_prefix": f"hf://{derived_prefix(checkpoint, run_tag)}",
        "run_tag": run_tag,
        "checkpoint": checkpoint_manifest(spec),
        "source_manifest": manifest,
        "validation": {
            "n_targets": len(timing_keys),
            "n_rollouts": EXPECTED_ROLLOUTS,
            "n_sparse_vote_rows": len(votes),
            "n_metric_rows": len(rows),
            "r_all_precision": observed_r_all,
            "reference": reference_validation,
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
    destination = derived_prefix(checkpoint, run_tag)
    for path in paths.values():
        upload(fs, path, destination)
        print(f"[finalize] {path} -> hf://{destination}/{path.name}", flush=True)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse the selected checkpoint."""

    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", choices=sorted(CHECKPOINTS))
    selection.add_argument("--verify-pr190-control", action="store_true")
    parser.add_argument("--run-tag")
    return parser.parse_args()


def main() -> int:
    """Finalize one complete checkpoint evaluation."""

    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if args.verify_pr190_control:
        if args.run_tag is not None:
            raise ValueError("--run-tag cannot be used with --verify-pr190-control")
        verify_pr190_control(token)
        return 0
    if not token:
        raise ValueError("HF_TOKEN must contain the open-athena write token")
    paths = finalize(args.checkpoint, token, args.run_tag)
    print(f"[finalize] complete: {', '.join(str(path) for path in paths.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
