"""Audit completed sweep coverage, baseline reproducibility, and sensitivity."""

import json
from pathlib import Path

import pandas as pd

from plot_length import interval

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Save checks derived from the full scorer's CSVs, without new inference."""
    proteins = pd.read_csv(HERE / "data/spiking_per_protein.csv")
    summary = pd.read_csv(HERE / "data/spiking_summary.csv")
    timings = pd.read_csv(HERE / "data/timings.csv")
    archived = pd.read_csv(HERE / "data/length_per_protein.csv")
    if len(timings) != 6664 or timings.n_rollouts.sum() != 666400:
        raise ValueError("The audit requires all 6,664 conditions and 666,400 rollouts")
    if timings.duplicated(["stem", "mode"]).any() or timings.stem.nunique() != 97:
        raise ValueError("Duplicate conditions or unexpected protein coverage")
    baseline = proteins.loc[(proteins.policy == "random") & (proteins.axis == "count") & (proteins.dose == 0)]
    comparison = baseline.merge(archived, on=["stem", "range"], validate="one_to_one")
    if len(comparison) != 194:
        raise ValueError("The archived baseline must cover both ranges on all 97 proteins")
    comparison["new_minus_archived"] = comparison.r_precision - comparison.precision
    comparison.to_csv(HERE / "data/baseline_reproducibility.csv", index=False)
    baselines = {}
    for contact_range, group in comparison.groupby("range"):
        lower, upper = interval(group.new_minus_archived.to_numpy())
        baselines[contact_range] = {
            "n_proteins": len(group), "new_mean": float(group.r_precision.mean()),
            "archived_mean": float(group.precision.mean()),
            "mean_difference": float(group.new_minus_archived.mean()),
            "difference_95ci": [lower, upper],
            "mean_absolute_per_protein_difference": float(group.new_minus_archived.abs().mean()),
        }
    sensitivity = {}
    for contact_range, group in summary.groupby("range"):
        sensitivity[contact_range] = {
            "maximum_absolute_change_in_mean_r_precision": float((group.r_precision - group.validity_gated_r_precision).abs().max()),
            "maximum_absolute_change_in_mean_gain": float((group.delta - group.validity_gated_delta).abs().max()),
        }
    policy_comparisons = []
    for metric in ("r_precision", "delta", "validity_gated_r_precision", "validity_gated_delta"):
        paired = proteins.pivot(index=["stem", "axis", "dose", "range"], columns="policy", values=metric).reset_index()
        paired["difference"] = paired.sse_balanced - paired.random
        for key, group in paired.groupby(["axis", "dose", "range"]):
            values = group.difference.dropna().to_numpy()
            lower, upper = interval(values)
            policy_comparisons.append(dict(zip(["axis", "dose", "range"], key)) | {
                "metric": metric, "n_proteins": len(values), "sse_minus_random": float(values.mean()),
                "lower": lower, "upper": upper,
            })
    pd.DataFrame(policy_comparisons).to_csv(HERE / "data/paired_policy_comparisons.csv", index=False)
    # Model-load time is repeated on every condition and is only additive once
    # per worker. Total condition time deliberately excludes that shared cost.
    workers = timings.drop_duplicates("worker_id")
    result = {
        "conditions": len(timings), "proteins": int(timings.stem.nunique()),
        "rollouts": int(timings.n_rollouts.sum()),
        "invalid_rollouts": int(timings.n_invalid_rollouts.sum()),
        "unfinished_rollouts": int(timings.n_unfinished.sum()),
        "malformed_statements": int(timings.n_malformed.sum()),
        "out_of_range_pairs": int(timings.n_out_of_range.sum()),
        "pure_inference_worker_hours": float(timings.elapsed_seconds.sum() / 3600),
        "condition_total_worker_hours": float(timings.total_seconds.sum() / 3600),
        "worker_model_load_seconds": float(workers.model_load_seconds.sum()),
        "workers_with_retained_conditions": len(workers),
        "first_condition_timestamp_utc": timings.timestamp_utc.min(),
        "last_condition_timestamp_utc": timings.timestamp_utc.max(),
        "timing_scope": "Retained completed conditions only; excludes canceled incomplete work and v1 attempts.",
        "baseline_reproducibility": baselines,
        "zero_invalid_rollout_sensitivity": sensitivity,
    }
    (HERE / "data/final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
