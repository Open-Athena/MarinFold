"""Extend discovery to 1,000 and score ground-truth-selected best individual F1."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import bootstrap_mean
from analyze import candidate_sets, load, r_precision

HERE = Path(__file__).resolve().parent
ENDPOINTS = [
    "r_precision",
    "true_coverage",
    "true_discovered",
    "false_discovered",
    "discovery_precision",
    "oracle_f1",
    "running_mean_f1",
    "mean_f1",
]


def contact_f1(predicted: set[int], truth: set[int]) -> float:
    """Score one contact set; an empty ground-truth universe is undefined."""
    if not truth:
        return float("nan")
    return 2 * len(predicted & truth) / (len(predicted) + len(truth))


def f1_trajectory(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix maxima and prefix means without reordering samples."""
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    return np.maximum.accumulate(scores), np.cumsum(scores) / np.arange(
        1, len(scores) + 1
    )


def summarize_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap proteins jointly for every curve point with identical resamples."""
    assert np.isfinite(values).all()
    n = len(values)
    indices = np.random.default_rng(373).integers(n, size=(10000, n))
    weights = np.zeros((10000, n))
    np.add.at(weights, (np.arange(10000)[:, None], indices), 1)
    replicated = (weights / n) @ values
    lo, hi = np.quantile(replicated, [0.025, 0.975], axis=0)
    return values.mean(axis=0), lo, hi


def reduce_condition(task: tuple[str, str, dict, list[int]]) -> tuple[list, list]:
    """Combine immutable original records with added samples and score a condition."""
    model, stem, truth, grid = task
    name = f"{stem}-t1.0-fixed.json.gz"
    original = load(HERE / "scratch/raw" / model / name)
    extension = load(HERE / "scratch/raw_1000" / model / name)
    record = {**original, "samples": original["samples"] + extension["samples"]}
    assert [s["rollout"] for s in record["samples"]] == list(range(1000))
    sets, true, ids, labels = candidate_sets(record, truth)
    length = truth["L"]
    curves, singles = [], []
    base = {
        "stem": stem,
        "model": model,
        "eval_set": original["target"]["eval_set"],
        "temperature": 1.0,
        "mode": "fixed",
    }
    for region, minimum in [("all", 6), ("long", 24)]:
        keep = ids % length - ids // length >= minimum
        ranged_ids, ranged_labels = ids[keep], labels[keep]
        true_range = {x for x in true if x % length - x // length >= minimum}
        if not true_range:
            raise ValueError(
                f"Empty truth: {stem}, {region}; report before aggregation"
            )
        ranged = [
            {x for x in sample if x % length - x // length >= minimum}
            for sample in sets
        ]
        scores = np.array([contact_f1(sample, true_range) for sample in ranged])
        oracle, running = f1_trajectory(scores)
        assert np.all(np.diff(oracle) >= 0) and np.all(oracle >= running - 1e-12)
        singles.extend(
            {**base, "range": region, "rollout": k, "f1": float(score)}
            for k, score in enumerate(scores)
        )
        votes = np.zeros(length * length, np.int32)
        discovered: set[int] = set()
        for k, sample in enumerate(ranged, 1):
            if sample:
                votes[list(sample)] += 1
            discovered |= sample
            if k not in grid:
                continue
            tp = len(discovered & true_range)
            curves.append(
                {
                    **base,
                    "range": region,
                    "k": k,
                    "r_precision": r_precision(votes, ranged_ids, ranged_labels),
                    "true_coverage": tp / len(true_range),
                    "true_discovered": tp,
                    "false_discovered": len(discovered - true_range),
                    "discovery_precision": tp / len(discovered) if discovered else 0.0,
                    "n_true": len(true_range),
                    "n_candidate": len(ranged_ids),
                    "oracle_f1": oracle[k - 1],
                    "running_mean_f1": running[k - 1],
                    "mean_f1": float(scores.mean()),
                }
            )
    return curves, singles


def main() -> None:
    """Save per-protein tables, uncertainty, paired contrasts and prefix regression checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads((HERE / "data/extension_plan.json").read_text())
    validation = json.loads((HERE / "data/validation_1000.json").read_text())
    assert validation["new_samples"] == 162400
    truths = json.loads((HERE / "inputs/ground_truth.json").read_text())
    tasks = [
        (model, truth["stem"], truth, plan["discovery_k"])
        for truth in truths
        for model in ["default", "mpnn"]
    ]
    curve_rows, single_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (curves, singles) in enumerate(pool.map(reduce_condition, tasks), 1):
            curve_rows.extend(curves)
            single_rows.extend(singles)
            if index % 20 == 0:
                print("REDUCED", index, "of", len(tasks), flush=True)
    curves = pd.DataFrame(curve_rows)
    singles = pd.DataFrame(single_rows)
    assert len(curves) == 232 * 2 * len(plan["discovery_k"])
    assert len(singles) == 232 * 2 * 1000
    # Check scientific equivalence against the already-published original prefixes.
    keys = ["stem", "model", "temperature", "mode", "range", "k"]
    old = pd.read_csv(HERE / "data/ensemble_curves.csv")
    old = (
        old[old.temperature.eq(1) & old["mode"].eq("fixed")]
        .set_index(keys)
        .sort_index()
    )
    new = curves[curves.k.le(300)].set_index(keys).sort_index()
    assert old.index.equals(new.index)
    for name in ENDPOINTS[:5] + ["n_true", "n_candidate"]:
        np.testing.assert_allclose(new[name], old[name], rtol=0, atol=1e-12)
    curves.to_csv(HERE / "data/ensemble_curves_1000.csv", index=False)
    # Compact derived scores retain enough information to rebuild every oracle curve.
    singles[["stem", "model", "eval_set", "range", "rollout", "f1"]].to_csv(
        HERE / "data/rollout_f1_1000.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    summaries, oracle_summaries, contrasts, gains, blocks = [], [], [], [], []
    for (cohort, region), part in curves.groupby(["eval_set", "range"]):
        for model, selected in part.groupby("model"):
            base = {
                "eval_set": cohort,
                "range": region,
                "model": model,
                "temperature": 1.0,
                "mode": "fixed",
            }
            for k, rows in selected.groupby("k"):
                for endpoint in ENDPOINTS:
                    if endpoint == "mean_f1" and k != 1000:
                        continue
                    mean, lo, hi, n = bootstrap_mean(
                        rows.sort_values("stem")[endpoint].to_numpy()
                    )
                    summaries.append(
                        {
                            **base,
                            "k": k,
                            "endpoint": endpoint,
                            "mean": mean,
                            "ci_low": lo,
                            "ci_high": hi,
                            "n": n,
                        }
                    )
            values = (
                singles[
                    singles.eval_set.eq(cohort)
                    & singles["range"].eq(region)
                    & singles.model.eq(model)
                ]
                .pivot(index="stem", columns="rollout", values="f1")
                .sort_index()
            )
            assert values.shape == (97 if cohort == "eval-val" else 19, 1000)
            for endpoint, matrix in [
                ("oracle_f1", np.maximum.accumulate(values.to_numpy(), axis=1)),
                (
                    "running_mean_f1",
                    np.cumsum(values.to_numpy(), axis=1) / np.arange(1, 1001),
                ),
            ]:
                mean, lo, hi = summarize_matrix(matrix)
                oracle_summaries.extend(
                    {
                        **base,
                        "k": k + 1,
                        "endpoint": endpoint,
                        "mean": mean[k],
                        "ci_low": lo[k],
                        "ci_high": hi[k],
                        "n": len(values),
                    }
                    for k in range(1000)
                )
            mean, lo, hi, n = bootstrap_mean(values.mean(axis=1).to_numpy())
            oracle_summaries.append(
                {
                    **base,
                    "k": 1000,
                    "endpoint": "mean_f1",
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )
            for block in range(10):
                mean, lo, hi, n = bootstrap_mean(
                    values.iloc[:, block * 100 : (block + 1) * 100]
                    .mean(axis=1)
                    .to_numpy()
                )
                blocks.append(
                    {
                        **base,
                        "block": block,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                    }
                )
            for endpoint in ENDPOINTS:
                if endpoint == "mean_f1":
                    continue
                pair = selected.pivot(index="stem", columns="k", values=endpoint)
                mean, lo, hi, n = bootstrap_mean((pair[1000] - pair[300]).to_numpy())
                gains.append(
                    {
                        **base,
                        "endpoint": endpoint,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                    }
                )
        for k, rows in part.groupby("k"):
            for endpoint in ENDPOINTS:
                if endpoint == "mean_f1" and k != 1000:
                    continue
                pair = rows.pivot(index="stem", columns="model", values=endpoint)
                mean, lo, hi, n = bootstrap_mean((pair.mpnn - pair.default).to_numpy())
                contrasts.append(
                    {
                        "eval_set": cohort,
                        "range": region,
                        "k": k,
                        "endpoint": endpoint,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                    }
                )
    for name, rows in [
        ("curve_summary_1000", summaries),
        ("oracle_summary_1000", oracle_summaries),
        ("curve_contrasts_1000", contrasts),
        ("extension_gains_1000", gains),
        ("f1_seed_blocks_1000", blocks),
    ]:
        pd.DataFrame(rows).to_csv(HERE / "data" / f"{name}.csv", index=False)
    checks = {
        "original_curve_rows_reproduced": len(old),
        "protein_curve_rows": len(curves),
        "individual_f1_scores": len(singles),
        "oracle_summary_rows": len(oracle_summaries),
        "zero_truth_conditions": 0,
        "all_original_prefix_endpoints_equal_atol": 1e-12,
        "oracle_monotonic_and_dominates_running_mean": True,
    }
    (HERE / "data/analysis_validation_1000.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
