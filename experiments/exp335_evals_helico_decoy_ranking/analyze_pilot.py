"""Analyze the target-balanced Helico decoy-ranking pilot."""

import csv
import json
import random
import shutil
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from benchmark import spearman_correlation

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "scratch" / "results" / "pilot"
FULL_BENCHMARK_CANDIDATES = 180_079 + 133
H100_USD_PER_HOUR = 3.95
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 335


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a non-empty CSV file."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV with stable columns."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated empirical percentile."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(values: list[float]) -> tuple[float, float]:
    """Return a deterministic 95% nonparametric bootstrap interval."""
    rng = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def native_rank(
    rows: list[dict], score: Callable[[dict], float]
) -> tuple[float, float]:
    """Return the native's average rank and one-positive AUROC."""
    native = next(row for row in rows if row["candidate_kind"] == "native")
    decoys = [row for row in rows if row["candidate_kind"] == "decoy"]
    native_score = score(native)
    better = sum(score(row) > native_score for row in decoys)
    tied = sum(score(row) == native_score for row in decoys)
    rank = 1 + better + 0.5 * tied
    auroc = (sum(score(row) < native_score for row in decoys) + 0.5 * tied) / len(
        decoys
    )
    return rank, auroc


def main() -> None:
    """Join truth and predictions, then write preregistered pilot endpoints."""
    truth_rows = load_csv(DATA / "pilot_candidates.csv")
    prediction_rows = load_csv(RESULTS / "candidate_metrics_full.csv")
    timings = load_csv(RESULTS / "timings_full.csv")
    truth = {(row["target"], row["decoy_id"]): row for row in truth_rows}
    if set(truth) != {(row["target"], row["decoy_id"]) for row in prediction_rows}:
        raise ValueError("pilot truth and prediction candidate keys differ")
    for row in prediction_rows:
        row.update(truth[(row["target"], row["decoy_id"])])

    methods: dict[str, Callable[[dict], float]] = {
        "Helico pTM": lambda row: float(row["max_ptm"]),
        "Helico mean-sample pTM": lambda row: float(row["mean_sample_ptm"]),
        "Helico mean CA pLDDT": lambda row: float(row["mean_ca_plddt"]),
        "Helico composite": lambda row: float(row["helico_composite"]),
        "AF2Rank composite": lambda row: float(row["af2rank_composite"]),
        "AF2 pTM": lambda row: float(row["af2_ptm"]),
        "DeepAccNet": lambda row: float(row["deepaccnet"]),
        "Rosetta energy": lambda row: float(row["negative_rosetta_energy"]),
    }
    native_methods = {
        "Helico pTM",
        "Helico mean-sample pTM",
        "Helico mean CA pLDDT",
        "Helico composite",
        "AF2Rank composite",
        "AF2 pTM",
    }

    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        by_target[row["target"]].append(row)

    per_target = []
    native_rows = []
    for method, score in methods.items():
        for target in sorted(by_target):
            rows = by_target[target]
            decoys = [row for row in rows if row["candidate_kind"] == "decoy"]
            scores = [score(row) for row in decoys]
            tm_scores = [float(row["tmscore"]) for row in decoys]
            selected = max(decoys, key=score)
            oracle_tm = max(tm_scores)
            per_target.append(
                {
                    "method": method,
                    "target": target,
                    "n_decoys": len(decoys),
                    "spearman_tmscore": spearman_correlation(scores, tm_scores),
                    "top1_tmscore": float(selected["tmscore"]),
                    "top1_gdt_ts": float(selected["gdt_ts"]),
                    "oracle_tmscore": oracle_tm,
                    "top1_tmscore_regret": oracle_tm - float(selected["tmscore"]),
                    "selected_decoy_id": selected["decoy_id"],
                }
            )
            if method in native_methods:
                rank, auroc = native_rank(rows, score)
                native_rows.append(
                    {
                        "method": method,
                        "target": target,
                        "n_candidates": len(rows),
                        "native_rank": rank,
                        "native_rank_percentile": (len(rows) - rank) / (len(rows) - 1),
                        "native_reciprocal_rank": 1 / rank,
                        "native_top1": int(rank == 1),
                        "native_vs_decoy_auroc": auroc,
                    }
                )

    summary_rows = []
    for method in methods:
        rows = [row for row in per_target if row["method"] == method]
        summary: dict[str, str | int | float] = {
            "method": method,
            "n_targets": len(rows),
            "decoys_per_target": int(rows[0]["n_decoys"]),
        }
        for metric in (
            "spearman_tmscore",
            "top1_tmscore",
            "top1_gdt_ts",
            "top1_tmscore_regret",
        ):
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_mean_interval(values)
            summary[f"mean_{metric}"] = statistics.mean(values)
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summary_rows.append(summary)

    native_summary = []
    for method in sorted(native_methods):
        rows = [row for row in native_rows if row["method"] == method]
        summary = {"method": method, "n_targets": len(rows)}
        for metric in (
            "native_rank",
            "native_rank_percentile",
            "native_reciprocal_rank",
            "native_top1",
            "native_vs_decoy_auroc",
        ):
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_mean_interval(values)
            summary[f"mean_{metric}"] = statistics.mean(values)
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        native_summary.append(summary)

    elapsed = [float(row["elapsed_seconds"]) for row in timings]
    measured_gpu_hours = sum(elapsed) / 3600
    projected_gpu_hours = statistics.mean(elapsed) * FULL_BENCHMARK_CANDIDATES / 3600
    target_mean_times = []
    for target in sorted(by_target):
        values = [
            float(row["elapsed_seconds"])
            for row in timings
            if row["stem"].startswith(f"{target}/")
        ]
        target_mean_times.append(statistics.mean(values))
    low_seconds, high_seconds = bootstrap_mean_interval(target_mean_times)
    projection = {
        "pilot_candidates": len(prediction_rows),
        "measured_inference_gpu_hours": measured_gpu_hours,
        "measured_inference_compute_usd": measured_gpu_hours * H100_USD_PER_HOUR,
        "mean_seconds_per_candidate": statistics.mean(elapsed),
        "median_seconds_per_candidate": statistics.median(elapsed),
        "full_candidates": FULL_BENCHMARK_CANDIDATES,
        "naive_full_projection_gpu_hours": projected_gpu_hours,
        "naive_full_projection_compute_usd": projected_gpu_hours * H100_USD_PER_HOUR,
        "naive_full_projection_wall_hours_at_8_h100": projected_gpu_hours / 8,
        "target_bootstrap_projection_gpu_hours_low": (
            low_seconds * FULL_BENCHMARK_CANDIDATES / 3600
        ),
        "target_bootstrap_projection_gpu_hours_high": (
            high_seconds * FULL_BENCHMARK_CANDIDATES / 3600
        ),
        "caveat": (
            "Inference-only projection from nine stratified targets (61-150 residues); "
            "excludes model startup and targets up to 223 residues. Benchmark batching before full launch."
        ),
    }

    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "pilot_per_target_metrics.csv", per_target)
    write_csv(DATA / "pilot_metric_summary.csv", summary_rows)
    write_csv(DATA / "pilot_native_ranking.csv", native_rows)
    write_csv(DATA / "pilot_native_summary.csv", native_summary)
    (DATA / "pilot_timing_projection.json").write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n"
    )
    for source, destination in (
        (RESULTS / "candidate_metrics_full.csv", DATA / "pilot_candidate_metrics.csv"),
        (RESULTS / "sample_metrics_full.csv", DATA / "pilot_sample_metrics.csv"),
        (RESULTS / "timings_full.csv", DATA / "timings.csv"),
    ):
        write_csv(destination, load_csv(source))
    shutil.copyfile(
        RESULTS / "run_manifest_full.json", DATA / "pilot_run_manifest.json"
    )

    for row in summary_rows:
        print(
            f"{row['method']}: rho={row['mean_spearman_tmscore']:.4f}, "
            f"top-1 TM={row['mean_top1_tmscore']:.4f}"
        )
    helico_native = next(row for row in native_summary if row["method"] == "Helico pTM")
    print(
        f"Helico pTM native top-1: {helico_native['mean_native_top1']:.3f}; "
        f"mean native rank: {helico_native['mean_native_rank']:.3f}"
    )
    print(
        f"Measured {measured_gpu_hours:.3f} inference H100-hours; naive full projection "
        f"{projected_gpu_hours:.1f} H100-hours / "
        f"${projection['naive_full_projection_compute_usd']:.0f}"
    )


if __name__ == "__main__":
    main()
