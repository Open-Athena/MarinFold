"""Shared metrics for the AF2Rank Rosetta-decoy benchmark."""

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

Row = Mapping[str, str]
ScoreFunction = Callable[[Row], float]


def af2rank_composite(row: Row) -> float:
    """Return the composite confidence score used in the AF2Rank paper."""
    return float(row["plddt"]) * float(row["ptm"]) * float(row["tm_diff"])


BASELINE_SCORES: dict[str, ScoreFunction] = {
    "AF2Rank composite": af2rank_composite,
    "AF2 pTM": lambda row: float(row["ptm"]),
    "AF2 mean pLDDT": lambda row: float(row["plddt"]),
    "DeepAccNet": lambda row: float(row["danscore"]),
    # Lower Rosetta energy is better; all other methods use higher-is-better.
    "Rosetta energy": lambda row: -float(row["rosettascore"]),
}


def load_af2rank_rows(path: Path) -> list[dict[str, str]]:
    """Load the authors' corrected AF2Rank CSV and validate its schema.

    Args:
        path: Path to ``rosetta_gapseq.csv``.

    Returns:
        Rows as string-valued dictionaries.
    """
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "target",
            "decoy_id",
            "gdt_ts",
            "tmscore",
            "rosettascore",
            "danscore",
            "tm_diff",
            "plddt",
            "ptm",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def rankdata(values: Iterable[float]) -> list[float]:
    """Assign one-based average ranks, matching Spearman tie handling."""
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2 + 1
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute Spearman correlation with average ranks for ties."""
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if len(left_rank) != len(right_rank):
        raise ValueError("Spearman inputs have different lengths")
    if len(left_rank) < 2:
        raise ValueError("Spearman correlation needs at least two observations")

    left_mean = statistics.mean(left_rank)
    right_mean = statistics.mean(right_rank)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_rank, right_rank, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left_rank)
    right_ss = sum((value - right_mean) ** 2 for value in right_rank)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return numerator / denominator


def summarize_baselines(rows: Iterable[Row]) -> list[dict[str, int | float | str]]:
    """Reproduce AF2Rank's target-macro correlation and top-1 endpoints.

    Native and no-template control rows are excluded, matching Figure 2 of the
    paper. Each target contributes equally regardless of its decoy count.
    """
    by_target: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row["decoy_id"] not in {"native", "none"}:
            by_target[row["target"]].append(row)
    if not by_target:
        raise ValueError("no decoy rows found")

    summaries: list[dict[str, int | float | str]] = []
    for method, score_function in BASELINE_SCORES.items():
        correlations: list[float] = []
        top1_tm_scores: list[float] = []
        top1_gdt_ts_scores: list[float] = []
        for target in sorted(by_target):
            target_rows = by_target[target]
            scores = [score_function(row) for row in target_rows]
            tm_scores = [float(row["tmscore"]) for row in target_rows]
            correlations.append(spearman_correlation(scores, tm_scores))

            # Python's max keeps the first exact tie. Preserving source-row
            # order reproduces the authors' published aggregation, including
            # tied DeepAccNet scores.
            selected = max(target_rows, key=score_function)
            top1_tm_scores.append(float(selected["tmscore"]))
            top1_gdt_ts_scores.append(float(selected["gdt_ts"]))

        summaries.append(
            {
                "method": method,
                "n_targets": len(by_target),
                "mean_target_spearman_tmscore": statistics.mean(correlations),
                "mean_top1_tmscore": statistics.mean(top1_tm_scores),
                "mean_top1_gdt_ts": statistics.mean(top1_gdt_ts_scores),
            }
        )
    return summaries
