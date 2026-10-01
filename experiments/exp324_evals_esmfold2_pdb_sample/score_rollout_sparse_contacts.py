"""Score sparse rollout vote parts against the exp324 10k contact ground truth."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MIN_SEQ_SEP = 6
RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}
CUTS = ("L", "L/2", "L/5", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-prefix", required=True, help="S3/local prefix containing <label>/scores/*.parquet")
    parser.add_argument("--label", required=True)
    parser.add_argument("--gt-parquet", type=Path, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--expected", type=int, default=10_000)
    return parser.parse_args()


def target_for_cut(cut: str, seq_len: int, n_true: int) -> int:
    if cut == "L":
        return seq_len
    if cut == "L/2":
        return max(1, seq_len // 2)
    if cut == "L/5":
        return max(1, seq_len // 5)
    if cut == "R":
        return n_true
    raise ValueError(cut)


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n = int(labels.size)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    sum_pos_ranks = float(ranks[labels.astype(bool)].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metric_rows(stem: str, model: str, seq_len: int, gt_i: list[int], gt_j: list[int], score: np.ndarray) -> list[dict[str, Any]]:
    true = np.zeros((seq_len, seq_len), dtype=bool)
    for i, j in zip(gt_i, gt_j, strict=True):
        if 0 <= i < j < seq_len and (j - i) >= MIN_SEQ_SEP:
            true[i, j] = True
    pair_i, pair_j = np.triu_indices(seq_len, k=1)
    pair_sep = pair_j - pair_i
    cand_score_all = score[pair_i, pair_j]
    cand_true_all = true[pair_i, pair_j]
    rows: list[dict[str, Any]] = []
    for range_name, (lo, hi) in RANGES.items():
        in_range = pair_sep >= lo
        if hi is not None:
            in_range &= pair_sep <= hi
        scores = cand_score_all[in_range]
        labels = cand_true_all[in_range].astype(np.int64)
        n_candidate = int(scores.size)
        n_true = int(labels.sum())
        auc = auc_from_scores(scores, labels) if n_candidate else math.nan
        order = np.argsort(-scores, kind="mergesort") if n_candidate else np.array([], dtype=np.int64)
        labels_sorted = labels[order] if n_candidate else labels
        for cut in CUTS:
            target = target_for_cut(cut, seq_len, n_true)
            if n_candidate == 0 or target <= 0:
                precision = math.nan
                n_top = 0
            else:
                n_top = min(target, n_candidate)
                precision = float(labels_sorted[:n_top].sum()) / n_top
            rows.append(
                {
                    "stem": stem,
                    "model": model,
                    "range": range_name,
                    "cut": cut,
                    "precision": precision,
                    "auc": auc,
                    "n_candidate": n_candidate,
                    "n_true": n_true,
                    "n_top": n_top,
                }
            )
    return rows


def read_score_parts(scores_root: str) -> pd.DataFrame:
    fs, root = fsspec.core.url_to_fs(scores_root.rstrip("/"))
    paths = sorted(fs.glob(f"{root}/scores/*.parquet"))
    if not paths:
        # Also support a raw prefix whose files are directly below it.
        paths = sorted(fs.glob(f"{root}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no score parquet parts under {scores_root}")
    frames = []
    for path in paths:
        with fs.open(path, "rb") as handle:
            frames.append(pq.read_table(handle).to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_csv(uri: str, frame: pd.DataFrame) -> None:
    with fsspec.open(uri, "wt") as handle:
        frame.to_csv(handle, index=False)


def main() -> None:
    args = parse_args()
    scores_root = f"{args.scores_prefix.rstrip('/')}/{args.label}"
    sparse = read_score_parts(scores_root)
    gt = pq.read_table(args.gt_parquet).to_pandas()
    sparse_by_stem = {stem: group for stem, group in sparse.groupby("stem", sort=False)}
    rows: list[dict[str, Any]] = []
    scored: set[str] = set()
    no_prediction_stems: list[str] = []
    for gt_row in gt.itertuples(index=False):
        stem = gt_row.stem
        seq_len = int(gt_row.seq_len)
        score = np.zeros((seq_len, seq_len), dtype=np.float32)
        group = sparse_by_stem.get(stem)
        if group is None:
            no_prediction_stems.append(stem)
        else:
            ii = group["i"].to_numpy(dtype=np.int64)
            jj = group["j"].to_numpy(dtype=np.int64)
            vv = group["votes"].to_numpy(dtype=np.float32)
            score[ii, jj] = vv
            score[jj, ii] = vv
        rows.extend(metric_rows(stem, args.label, seq_len, list(gt_row.gt_contact_i), list(gt_row.gt_contact_j), score))
        scored.add(stem)
    unexpected = sorted(set(sparse_by_stem) - set(gt["stem"]))
    metric_df = pd.DataFrame(rows)
    summary_df = (
        metric_df.groupby(["model", "range", "cut"], dropna=False)
        .agg(mean_precision=("precision", "mean"), n_proteins=("precision", "count"))
        .reset_index()
    )
    out = args.out_prefix.rstrip("/")
    write_csv(f"{out}/contact_precision.csv", metric_df)
    write_csv(f"{out}/contact_precision_summary.csv", summary_df)
    with fsspec.open(f"{out}/score_manifest.json", "wt") as handle:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": args.label,
                "scores_root": scores_root,
                "n_scored": len(scored),
                "expected": args.expected,
                "n_no_predictions": len(no_prediction_stems),
                "no_prediction_stems": no_prediction_stems[:100],
                "n_unexpected_prediction_stems": len(unexpected),
                "unexpected_prediction_stems": unexpected[:100],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(summary_df[(summary_df["range"].isin(["all", "long"])) & (summary_df["cut"].isin(["R"]))].to_string(index=False))
    if len(scored) != args.expected:
        raise SystemExit(f"incomplete: scored {len(scored)} != {args.expected}")
    if unexpected:
        raise SystemExit(f"found {len(unexpected)} prediction stems absent from ground truth")


if __name__ == "__main__":
    main()
