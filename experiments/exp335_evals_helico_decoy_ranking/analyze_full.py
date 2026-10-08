"""Analyze the complete 133-target Helico decoy-ranking run."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from analyze_pilot import bootstrap_mean_interval, native_rank
from benchmark import af2rank_composite, load_af2rank_rows, spearman_correlation
from full_worker_cw import CONFIG, RUN_FINGERPRINT

H100_PLANNING_USD_PER_HOUR = 3.95
COMMITTED_TIMING_COLUMNS = (
    "stem",
    "n_residues",
    "n_pairs",
    "mode",
    "elapsed_seconds",
    "model_load_seconds",
    "total_seconds",
    "model_nickname",
    "runner_tag",
    "gpu_name",
    "gpu_total_memory_gb",
    "gpu_compute_capability",
    "hostname",
    "platform",
    "torch_version",
    "timestamp_utc",
    "cuequivariance_torch_version",
    "model_load_share_seconds",
    "target_setup_share_seconds",
    "runner_setup_share_seconds",
    "output_share_seconds",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a non-empty CSV."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries with stable Unix line endings."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("scratch/full_results")
    )
    parser.add_argument("--af2rank-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def paired_comparisons(
    rows: list[dict],
    comparisons: list[tuple[str, str]],
    metrics: tuple[str, ...],
    endpoint: str,
) -> list[dict]:
    """Compute paired target-bootstrap intervals for method differences."""
    by_method_target = {(str(row["method"]), str(row["target"])): row for row in rows}
    targets = sorted({str(row["target"]) for row in rows})
    output = []
    for left_method, right_method in comparisons:
        for metric in metrics:
            differences = [
                float(by_method_target[(left_method, target)][metric])
                - float(by_method_target[(right_method, target)][metric])
                for target in targets
            ]
            low, high = bootstrap_mean_interval(differences)
            output.append(
                {
                    "endpoint": endpoint,
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "n_targets": len(differences),
                    "mean_left_minus_right": statistics.mean(differences),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                }
            )
    return output


def main() -> None:
    """Join predictions to truth and compute target-macro endpoints."""
    args = parse_args()
    predictions = load_csv(args.results_dir / "candidate_metrics.csv")
    timings = load_csv(args.results_dir / "timings.csv")
    truth_rows = [
        row for row in load_af2rank_rows(args.af2rank_csv) if row["decoy_id"] != "none"
    ]
    truth = {(row["target"], row["decoy_id"]): row for row in truth_rows}
    prediction_keys = {(row["target"], row["decoy_id"]) for row in predictions}
    if set(truth) != prediction_keys:
        raise ValueError(
            f"truth/prediction mismatch: truth-only={len(set(truth) - prediction_keys)}, "
            f"prediction-only={len(prediction_keys - set(truth))}"
        )
    for row in predictions:
        row.update(truth[(row["target"], row["decoy_id"])])

    methods: dict[str, Callable[[dict], float]] = {
        "Helico pTM": lambda row: float(row["max_ptm"]),
        "Helico mean-sample pTM": lambda row: float(row["mean_sample_ptm"]),
        "Helico mean CA pLDDT": lambda row: float(row["mean_ca_plddt"]),
        "Helico composite": lambda row: float(row["helico_composite"]),
        "AF2Rank composite": af2rank_composite,
        "AF2 pTM": lambda row: float(row["ptm"]),
        "DeepAccNet": lambda row: float(row["danscore"]),
        "Rosetta energy": lambda row: -float(row["rosettascore"]),
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
    for row in predictions:
        by_target[row["target"]].append(row)
    if len(by_target) != 133:
        raise ValueError(f"expected 133 targets, found {len(by_target)}")

    per_target = []
    native_rows = []
    for method, score in methods.items():
        for target in sorted(by_target):
            rows = by_target[target]
            decoys = [row for row in rows if row["decoy_id"] != "native"]
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
            "min_decoys_per_target": min(int(row["n_decoys"]) for row in rows),
            "median_decoys_per_target": statistics.median(
                int(row["n_decoys"]) for row in rows
            ),
            "max_decoys_per_target": max(int(row["n_decoys"]) for row in rows),
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

    comparison_rows = paired_comparisons(
        per_target,
        [
            ("Helico pTM", "Rosetta energy"),
            ("Helico pTM", "AF2Rank composite"),
            ("Helico composite", "AF2Rank composite"),
        ],
        ("spearman_tmscore", "top1_tmscore"),
        "decoy_ranking",
    )
    comparison_rows.extend(
        paired_comparisons(
            native_rows,
            [
                ("Helico pTM", "AF2Rank composite"),
                ("Helico composite", "AF2Rank composite"),
            ],
            ("native_top1", "native_rank", "native_vs_decoy_auroc"),
            "native_selection",
        )
    )

    inference_hours = sum(float(row["elapsed_seconds"]) for row in timings) / 3600
    accounted_hours = sum(float(row["total_seconds"]) for row in timings) / 3600
    timing_summary = {
        "n_candidates": len(timings),
        "inference_h100_hours": inference_hours,
        "accounted_h100_hours": accounted_hours,
        "planning_cost_at_3_95_usd_per_h100_hour": (
            accounted_hours * H100_PLANNING_USD_PER_HOUR
        ),
        "mean_inference_seconds": statistics.mean(
            float(row["elapsed_seconds"]) for row in timings
        ),
        "median_inference_seconds": statistics.median(
            float(row["elapsed_seconds"]) for row in timings
        ),
        "mean_total_seconds": statistics.mean(
            float(row["total_seconds"]) for row in timings
        ),
    }
    run_manifest = {
        "run_fingerprint": RUN_FINGERPRINT,
        "config": CONFIG,
        "coverage": {
            "targets": len(by_target),
            "candidates": len(predictions),
            "decoys": sum(row["decoy_id"] != "native" for row in predictions),
            "natives": sum(row["decoy_id"] == "native" for row in predictions),
        },
        "timing": timing_summary,
    }

    args.output_dir.mkdir(exist_ok=True)
    write_csv(args.output_dir / "full_per_target_metrics.csv", per_target)
    write_csv(args.output_dir / "full_metric_summary.csv", summary_rows)
    write_csv(args.output_dir / "full_native_ranking.csv", native_rows)
    write_csv(args.output_dir / "full_native_summary.csv", native_summary)
    write_csv(args.output_dir / "full_paired_comparisons.csv", comparison_rows)
    committed_timings = [
        {column: row[column] for column in COMMITTED_TIMING_COLUMNS} for row in timings
    ]
    write_csv(args.output_dir / "timings.csv", committed_timings)
    (args.output_dir / "full_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    for row in summary_rows:
        print(
            f"{row['method']}: rho={row['mean_spearman_tmscore']:.4f}, "
            f"top-1 TM={row['mean_top1_tmscore']:.4f}"
        )
    for row in native_summary:
        print(
            f"{row['method']}: native top-1={row['mean_native_top1']:.4f}, "
            f"mean rank={row['mean_native_rank']:.2f}"
        )
    print(json.dumps(timing_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
