# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize exp177 sparse target shape statistics from S3 parquet files."""

import argparse
import os
from collections.abc import Sequence

import fsspec
import numpy as np
import pyarrow.parquet as pq

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_soft_target_loss_h2h/stats/sparse_target_shapes_v1/"
    "2026.08.19.1/*.parquet"
)

STAT_COLUMNS = (
    "num_tokens",
    "padding_tokens_at_seq_len",
    "residue_count",
    "contact_count",
    "unique_endpoint_count",
    "neighbor_nnz",
    "max_degree",
    "degree_p90",
    "degree_p95",
    "degree_p99",
    "filled_from_real_slot",
)


def _storage_options() -> dict:
    if fsspec_s3 := os.environ.get("FSSPEC_S3"):
        import json

        return json.loads(fsspec_s3)
    return {}


def _read_columnar(fs: fsspec.AbstractFileSystem, paths: Sequence[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {column: [] for column in STAT_COLUMNS}
    for path in paths:
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, columns=list(STAT_COLUMNS))
        for column in STAT_COLUMNS:
            arrays[column].append(table[column].to_numpy(zero_copy_only=False))
    return {column: np.concatenate(parts) if parts else np.asarray([]) for column, parts in arrays.items()}


def _print_quantiles(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{name}: empty")
        return
    quantiles = np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0], method="higher")
    labels = ("min", "p50", "p90", "p95", "p99", "p999", "max")
    rendered = " ".join(f"{label}={int(value)}" for label, value in zip(labels, quantiles, strict=True))
    print(f"{name}: {rendered} mean={float(np.mean(values)):.2f}")


def _coverage(arrays: dict[str, np.ndarray], *, residue_cap: int, degree_cap: int) -> int:
    return int(np.sum((arrays["residue_count"] <= residue_cap) & (arrays["max_degree"] <= degree_cap)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Candidate bucket as RESIDUES,DEGREE. Can be repeated.",
    )
    args = parser.parse_args()

    fs, _, paths = fsspec.get_fs_token_paths(args.input, storage_options=_storage_options())
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(args.input)
    arrays = _read_columnar(fs, paths)
    n = int(arrays["contact_count"].size)
    print(f"files={len(paths)} rows={n}")
    print(f"quota_fill_rows={int(np.sum(arrays['filled_from_real_slot']))}")
    for column in STAT_COLUMNS:
        if column == "filled_from_real_slot":
            continue
        _print_quantiles(column, arrays[column])

    buckets = args.bucket or ["256,32", "512,32", "1024,32", "2000,32"]
    covered = np.zeros(n, dtype=np.bool_)
    print("candidate_bucket_coverage:")
    for bucket in buckets:
        residue_cap_s, degree_cap_s = bucket.split(",", 1)
        residue_cap = int(residue_cap_s)
        degree_cap = int(degree_cap_s)
        mask = (arrays["residue_count"] <= residue_cap) & (arrays["max_degree"] <= degree_cap)
        incremental = mask & ~covered
        covered |= mask
        print(
            f"  r{residue_cap}-d{degree_cap}: total={int(np.sum(mask))} "
            f"({np.mean(mask) * 100:.2f}%) incremental={int(np.sum(incremental))} "
            f"cumulative={int(np.sum(covered))} ({np.mean(covered) * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
