# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare contact R-precision from saved per-protein score matrices."""

import argparse
import json
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd

RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
MIN_DEGREE = 0.001
MIN_SEP = 6


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with fsspec.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def true_matrix(length: int, contacts: list[list[float]]) -> np.ndarray:
    mat = np.zeros((length, length), dtype=bool)
    for i_raw, j_raw, degree_raw in contacts:
        i = int(i_raw)
        j = int(j_raw)
        degree = float(degree_raw)
        if i < j < length and degree >= MIN_DEGREE and (j - i) >= MIN_SEP:
            mat[i, j] = True
    return mat


def resolved_pairs(resolved: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_arr = np.asarray(resolved, dtype=np.int64)
    a, b = np.triu_indices(len(resolved_arr), k=1)
    i = resolved_arr[a]
    j = resolved_arr[b]
    return i, j, j - i


def score_path(score_dir: str, dataset: str, stem: str) -> str:
    return f"{score_dir.rstrip('/')}/{dataset}__{stem}.npz"


def load_score(path: str, score_key: str) -> np.ndarray | None:
    if not fsspec.core.url_to_fs(path)[0].exists(fsspec.core.url_to_fs(path)[1]):
        return None
    with fsspec.open(path, "rb") as fh:
        loaded = np.load(fh)
        if score_key not in loaded:
            available = ", ".join(loaded.files)
            raise KeyError(f"{path} has keys [{available}], not {score_key!r}")
        return loaded[score_key].astype(np.float64)


def metric_rows(records: list[dict[str, Any]], model: str, score_dir: str, score_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        dataset = str(rec["dataset"])
        stem = str(rec["stem"])
        length = int(rec["L"])
        path = score_path(score_dir, dataset, stem)
        score = load_score(path, score_key)
        if score is None:
            continue
        if score.shape != (length, length):
            print(f"[metrics] {model} {dataset}/{stem}: score shape {score.shape} != {(length, length)}; skipping")
            continue
        truth = true_matrix(length, rec["contacts"])
        pi, pj, sep = resolved_pairs(rec["resolved"])
        pair_scores = score[pi, pj]
        pair_truth = truth[pi, pj].astype(np.int32)
        for range_name, (lo, hi) in RANGES.items():
            in_range = sep >= lo
            if hi is not None:
                in_range = in_range & (sep <= hi)
            s = pair_scores[in_range]
            t = pair_truth[in_range]
            n_candidate = int(s.size)
            n_true = int(t.sum())
            if n_candidate == 0 or n_true <= 0:
                precision = float("nan")
                n_top = 0
            else:
                order = np.argsort(-s, kind="mergesort")
                n_top = min(n_true, n_candidate)
                precision = float(t[order[:n_top]].sum()) / n_top
            rows.append({
                "dataset": dataset,
                "stem": stem,
                "n_residues": length,
                "model": model,
                "range": range_name,
                "cut": "R",
                "precision": precision,
                "n_candidate": n_candidate,
                "n_true": n_true,
                "n_top": n_top,
            })
    print(f"[metrics] {model}: {len({(r['dataset'], r['stem']) for r in rows})}/{len(records)} proteins")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl")
    parser.add_argument("--scores", action="append", required=True, help="label=score_dir")
    parser.add_argument("--score-key", default="score", help="NPZ array key to rank by (default: score)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.gt)
    rows: list[dict[str, Any]] = []
    for spec in args.scores:
        label, sep, directory = spec.partition("=")
        if not sep:
            raise ValueError(f"--scores must be label=dir, got {spec!r}")
        rows.extend(metric_rows(records, label, directory, args.score_key))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["model", "range"], dropna=False)["precision"].agg(["count", "mean", "median"])
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
