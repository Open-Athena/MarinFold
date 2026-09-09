"""Paired per-protein bootstrap comparisons of saved exp281 evaluations."""

import argparse
import csv
from collections import defaultdict

import fsspec
import numpy as np

from common import write_json


def read_scores(path: str, metric: str) -> dict[tuple[str, str, str], float]:
    """Require unique target/mode/budget rows and finite values for comparison."""
    with fsspec.open(path, "r") as handle:
        result = {}
        for row in csv.DictReader(handle):
            key = (row["target_id"], row["forced"], row["budget"])
            value = float(row[metric])
            if key in result or not np.isfinite(value):
                raise ValueError("duplicate or nonfinite evaluation row")
            result[key] = value
    return result


def paired_difference(baseline: dict, candidate: dict, seed: int = 281) -> dict:
    """Resample proteins, preserving dependence between their budget/mode rows."""
    if baseline.keys() != candidate.keys() or not baseline:
        raise ValueError("comparison requires identical nonempty target/mode/budget sets")
    grouped = defaultdict(list)
    for key, value in baseline.items():
        grouped[key[0]].append(candidate[key] - value)
    differences = np.array([np.mean(values) for _, values in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(10000)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"proteins": len(differences), "delta": float(differences.mean()),
            "ci95": [float(low), float(high)], "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()), "bootstrap_seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", default="final_f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = paired_difference(read_scores(args.baseline, args.metric), read_scores(args.candidate, args.metric))
    write_json(args.output, {"metric": args.metric, **result})


if __name__ == "__main__":
    main()
