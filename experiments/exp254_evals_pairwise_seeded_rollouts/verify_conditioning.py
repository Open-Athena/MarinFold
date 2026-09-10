# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently verify conditioning outputs from their recorded raw text.

This CPU-only check reconstructs document position maps, reparses completions,
and rebuilds every continuation vote matrix. It validates completion manifests,
timing aggregates, engine metadata, and the frozen first-pass source identity.
It never computes accuracy or interprets an operational smoke as a population
result. Probability matrices are checked for completeness and numerical validity;
verifying their numerical predictions would require another model execution.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from common import parse_rollout, realization

ARMS = (
    "iid",
    "iid_repeat",
    "true_small",
    "false_small",
    "pred_small",
    "true_large",
    "false_large",
    "pred_large",
)
PAYLOAD_SUFFIXES = ("npz", "raw.json.gz", "timings.csv")
STRING_METADATA = (
    "model_nickname",
    "model_source",
    "runner_tag",
    "gpu_name",
    "gpu_compute_capability",
    "hostname",
    "platform",
    "torch_version",
    "vllm_version",
    "transformers_version",
    "plan_sha256",
    "worker_sha256",
)


def sha256(payload: bytes) -> str:
    """Return a content digest for a serialized input or output."""
    return hashlib.sha256(payload).hexdigest()


def position_frames(
    target: dict, n_repeats: int, n_rollouts: int
) -> dict[int, list[list[int]]]:
    """Reconstruct the actual document position rings used by the worker."""
    # The document library is optional for other experiment analysis commands.
    from marinfold.document_structures.contacts_v1 import residues_from_sequence

    residues = residues_from_sequence(target["input_seq"])
    frames = {}
    for repeat in range(n_repeats):
        frames[repeat] = [
            realization(target["stem"], residues, f"conditional-v1-rep{repeat}-r{r}")[1]
            for r in range(n_rollouts)
        ]
        if any(
            len(positions) != target["L"] or len(set(positions)) != target["L"]
            for positions in frames[repeat]
        ):
            raise ValueError(f"{target['stem']}: invalid reconstructed position map")
    return frames


def checked_payloads(
    run: Path, stem: str, plan_sha: str
) -> tuple[dict, dict[str, bytes]]:
    """Read and validate all payloads against an exact completion manifest."""
    marker_path = run / "units" / f"{stem}.complete.json"
    marker = json.loads(marker_path.read_text())
    if marker["stem"] != stem or marker["plan_sha256"] != plan_sha:
        raise ValueError(f"{stem}: completion identity differs from frozen plan")
    if set(marker["files"]) != set(PAYLOAD_SUFFIXES):
        raise ValueError(f"{stem}: completion manifest has missing or extra payloads")
    elapsed = float(marker["elapsed_seconds"])
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{stem}: invalid completed-unit elapsed time")
    payloads = {}
    for suffix in PAYLOAD_SUFFIXES:
        payload = (run / "units" / f"{stem}.{suffix}").read_bytes()
        expected = marker["files"][suffix]
        if len(payload) != expected["bytes"] or sha256(payload) != expected["sha256"]:
            raise ValueError(f"{stem}.{suffix}: checksum or length mismatch")
        payloads[suffix] = payload
    return marker, payloads


def check_numeric_matrix(
    matrix: np.ndarray, length: int, label: str, n_rollouts: int | None
) -> None:
    """Validate counts or a symmetric forced-next-contact readout."""
    if matrix.shape != (length, length) or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: invalid matrix shape or nonfinite values")
    if not np.array_equal(matrix, matrix.T) or (matrix < 0).any():
        raise ValueError(f"{label}: asymmetric or negative scores")
    if n_rollouts is not None:
        if (matrix > n_rollouts).any() or not (matrix == np.floor(matrix)).all():
            raise ValueError(f"{label}: invalid vote counts")
        return
    # The helper adds two orientations. Its unscored diagonal doubles a self
    # event, so validate unordered probability bounds only off the diagonal.
    ii, jj = np.triu_indices(length, k=1)
    if (matrix[ii, jj] > 1).any():
        raise ValueError(f"{label}: unordered pair probability exceeds one")


def reparse_group(
    completions: list,
    frames: list[list[int]],
    length: int,
    given: list,
    budget: int,
    label: str,
) -> tuple[np.ndarray, dict]:
    """Rebuild votes and counters from raw text, checking stored parsed contacts."""
    if len(completions) != len(frames):
        raise ValueError(f"{label}: wrong number of raw completions")
    votes = np.zeros((length, length), dtype=np.int64)
    given_set = {tuple(pair) for pair in given}
    token_total, copied, novel, empty = 0, 0, 0, 0
    for index, (completion, positions) in enumerate(
        zip(completions, frames, strict=True)
    ):
        if completion["finish_reason"] != "stop":
            raise ValueError(f"{label}/{index}: rollout did not stop normally")
        tokens = completion["tokens"]
        if not isinstance(tokens, int) or not 0 < tokens <= budget:
            raise ValueError(f"{label}/{index}: invalid completion token count")
        pairs = parse_rollout(
            completion["text"], {position: i for i, position in enumerate(positions)}
        )
        if [list(pair) for pair in pairs] != completion["contacts"]:
            raise ValueError(
                f"{label}/{index}: saved contacts differ from reparsed raw text"
            )
        # Count directly, without sharing the worker's advanced-index update.
        for i, j in pairs:
            votes[i, j] += 1
            votes[j, i] += 1
        copied += len(set(pairs) & given_set)
        novel += len(set(pairs) - given_set)
        token_total += tokens
        empty += not pairs
    return votes, {
        "generated_tokens": token_total,
        "copied_context_pairs": copied,
        "novel_generated_pairs": novel,
        "unfinished_rollouts": 0,
        "empty_rollouts": int(empty),
    }


def timing_number(row: pd.Series, name: str, nonnegative: bool = True) -> float:
    """Read a finite timing/counter value and reject invalid telemetry."""
    value = float(row[name])
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"invalid timing field {name}")
    return value


def verify_unit(plan: dict, target: dict, run: Path, plan_sha: str) -> dict:
    """Verify every raw group, score matrix, and timing row for one target."""
    stem = target["stem"]
    n_rollouts, n_repeats = plan["n_rollouts"], plan["n_repeats"]
    marker, payloads = checked_payloads(run, stem, plan_sha)
    raw = json.loads(gzip.decompress(payloads["raw.json.gz"]))
    expected_groups = {
        f"r{repeat}__{arm}" for repeat in range(n_repeats) for arm in ARMS
    }
    if set(raw) != expected_groups:
        raise ValueError(f"{stem}: raw output has missing or extra groups")
    timings = pd.read_csv(
        io.BytesIO(payloads["timings.csv"]),
        dtype={name: str for name in STRING_METADATA},
    )
    if timings.duplicated(["replicate", "mode"]).any():
        raise ValueError(f"{stem}: duplicate timing rows")
    timing_groups = {
        f"r{row.replicate}__{row.mode}" for row in timings.itertuples(index=False)
    }
    if timing_groups != expected_groups or len(timings) != len(expected_groups):
        raise ValueError(f"{stem}: timing rows have missing or extra groups")
    for name in STRING_METADATA:
        if (
            timings[name].isna().any()
            or not timings[name].str.len().gt(0).all()
            or timings[name].nunique() != 1
        ):
            raise ValueError(f"{stem}: missing or inconsistent worker metadata {name}")
    meta = timings.iloc[0]
    if (
        meta.plan_sha256 != plan_sha
        or meta.model_nickname != f"{plan['model_run']}-step-{plan['step']}"
    ):
        raise ValueError(f"{stem}: timing metadata differs from frozen plan")
    if not re.fullmatch(r"[0-9a-f]{64}", meta.worker_sha256):
        raise ValueError(f"{stem}: invalid worker source digest")
    if meta.runner_tag not in ("local", "iris"):
        raise ValueError(f"{stem}: unknown execution runner")
    frames = position_frames(target, n_repeats, n_rollouts)
    expected_arrays = {f"{group}__votes" for group in expected_groups}
    expected_arrays |= {
        f"{group}__prob"
        for group in expected_groups
        if not group.endswith("__iid_repeat")
    }
    units_empty, tokens_total = 0, 0
    with np.load(io.BytesIO(payloads["npz"]), allow_pickle=False) as archive:
        if set(archive.files) != expected_arrays:
            raise ValueError(f"{stem}: NPZ has missing or extra matrices")
        for repeat in range(n_repeats):
            for arm in ARMS:
                key = f"r{repeat}__{arm}"
                budget = 6 * target["L"] + 128
                given = target["contexts"][repeat][arm]
                reconstructed, counters = reparse_group(
                    raw[key],
                    frames[repeat],
                    target["L"],
                    given,
                    budget,
                    f"{stem}/{key}",
                )
                votes = archive[f"{key}__votes"]
                check_numeric_matrix(votes, target["L"], key, n_rollouts)
                if not np.array_equal(reconstructed, votes):
                    raise ValueError(
                        f"{stem}/{key}: saved votes differ from raw-text reconstruction"
                    )
                if arm != "iid_repeat":
                    check_numeric_matrix(
                        archive[f"{key}__prob"], target["L"], key + " probability", None
                    )
                row = timings[
                    (timings.replicate == repeat) & (timings["mode"] == arm)
                ].iloc[0]
                required_counts = dict(
                    stem=stem,
                    n_residues=target["L"],
                    n_pairs=target["L"] * (target["L"] - 1) // 2,
                    n_rollouts=n_rollouts,
                    n_given=len(given),
                    max_new_tokens=budget,
                    **{
                        name: value
                        for name, value in counters.items()
                        if name != "empty_rollouts"
                    },
                )
                for name, value in required_counts.items():
                    if row[name] != value:
                        raise ValueError(
                            f"{stem}/{key}: timing field {name} disagrees with raw output or plan"
                        )
                elapsed = timing_number(row, "elapsed_seconds")
                probe = timing_number(row, "probability_probe_seconds")
                total = timing_number(row, "total_seconds")
                timing_number(row, "model_load_seconds")
                if timing_number(row, "gpu_total_memory_gb") <= 0:
                    raise ValueError(f"{stem}: invalid GPU memory metadata")
                if total + 1e-6 < elapsed + probe or (
                    arm == "iid_repeat" and probe != 0
                ):
                    raise ValueError(f"{stem}/{key}: inconsistent timing durations")
                prompt_tokens = timing_number(row, "prompt_tokens")
                if (
                    prompt_tokens != int(prompt_tokens)
                    or prompt_tokens <= 0
                    or prompt_tokens + budget > plan["max_model_len"]
                ):
                    raise ValueError(
                        f"{stem}/{key}: invalid prompt/completion allowance"
                    )
                timestamp = datetime.fromisoformat(row.timestamp_utc)
                if timestamp.utcoffset() != timedelta(0):
                    raise ValueError(f"{stem}/{key}: timestamp is not UTC")
                units_empty += counters["empty_rollouts"]
                tokens_total += counters["generated_tokens"]
    if marker["elapsed_seconds"] + 1e-6 < timings.total_seconds.sum():
        raise ValueError(
            f"{stem}: completed-unit time is shorter than its recorded arm times"
        )
    return {
        "stem": stem,
        "n_groups": len(expected_groups),
        "n_completions": n_rollouts * len(expected_groups),
        "empty_rollouts": units_empty,
        "generated_tokens": tokens_total,
        "worker_sha256": meta.worker_sha256,
        "vllm_version": meta.vllm_version,
        "transformers_version": meta.transformers_version,
        "torch_version": meta.torch_version,
        "gpu_name": meta.gpu_name,
        "model_source": meta.model_source,
    }


def verify(plan_path: Path, run: Path, stem: str | None = None) -> dict:
    """Validate one operational smoke or the entire frozen target population."""
    plan_bytes = plan_path.read_bytes()
    plan, plan_sha = json.loads(plan_bytes), sha256(plan_bytes)
    if tuple(plan["arms"]) != ARMS or min(plan["n_rollouts"], plan["n_repeats"]) <= 0:
        raise ValueError("unexpected protocol arms or sample counts")
    targets = {target["stem"]: target for target in plan["targets"]}
    if not targets or len(targets) != len(plan["targets"]):
        raise ValueError("plan has empty or duplicate targets")
    source_path = plan_path.parent / plan["source_votes_file"]
    source_bytes = source_path.read_bytes()
    if sha256(source_bytes) != plan["source_votes_sha256"]:
        raise ValueError("archived source vote checksum differs from frozen plan")
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as source:
        if set(source.files) != set(targets):
            raise ValueError("archived source has missing or extra targets")
        for name, target in targets.items():
            check_numeric_matrix(
                source[name], target["L"], f"source/{name}", plan["source_n_rollouts"]
            )
    if stem is not None:
        if stem not in targets:
            raise ValueError(f"smoke target {stem!r} is not in frozen plan")
        selected = [targets[stem]]
    else:
        selected = list(targets.values())
        expected_files = {
            f"{name}.{suffix}"
            for name in targets
            for suffix in (*PAYLOAD_SUFFIXES, "complete.json")
        }
        actual_files = {path.name for path in (run / "units").iterdir()}
        if actual_files != expected_files:
            raise ValueError(
                f"full run has missing/extra unit files: missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
    units = [verify_unit(plan, target, run, plan_sha) for target in selected]
    for name in (
        "worker_sha256",
        "vllm_version",
        "transformers_version",
        "torch_version",
        "model_source",
    ):
        if len({unit[name] for unit in units}) != 1:
            raise ValueError(f"full run mixes incompatible worker metadata: {name}")
    return {
        "scope": "operational_smoke" if stem else "full_frozen_plan",
        "plan_sha256": plan_sha,
        "n_verified_targets": len(units),
        "n_planned_targets": len(targets),
        "n_verified_completions": sum(unit["n_completions"] for unit in units),
        "source_sha256": sha256(source_bytes),
        "units": units,
        "limitations": "Raw text, stored contacts, counts, completion statuses, and telemetry "
        "are cross-checked. Stored token counts are reconciled with telemetry, not "
        "retokenized; raw token IDs were not recorded. Probability values are "
        "numerically validated but not independently recomputed. No accuracy is scored.",
    }


def main() -> int:
    """Print an integrity-only report, optionally writing it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stem")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = verify(args.plan, args.run, args.stem)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
