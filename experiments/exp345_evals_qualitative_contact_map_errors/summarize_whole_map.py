"""Summarize secondary comparisons and diagnose coverage of the cluster probe."""

import json

import pandas as pd

from analyze_whole_map import HERE, bootstrap


def main() -> None:
    """Write paired comparison and cluster-eligibility audit artifacts."""
    methods = pd.read_csv(HERE / "data/whole_map_methods.csv")
    all_f1 = methods[methods["range"].eq("all")].groupby(["stem", "method"]).f1.mean().unstack()
    gap = all_f1.sample_oracle_200-all_f1.pooled_200
    comparisons = {"oracle_beats_pooled": int((gap > 0).sum()), "pooled_beats_oracle": int((gap < 0).sum()),
                   "oracle_200_minus_100": bootstrap((all_f1.sample_oracle_200-all_f1.sample_oracle_100).to_numpy()),
                   "largest_oracle_gains": all_f1.assign(gap=gap).sort_values("gap", ascending=False)[["pooled_200", "sample_oracle_200", "gap"]].head(10).reset_index().to_dict("records")}
    clusters = pd.read_csv(HERE / "data/whole_map_clusters.csv")
    coverage = {}
    for k, part in clusters[clusters["range"].eq("all")].groupby("k"):
        groups = part.groupby(["stem", "training_pool"]).agg(eligible=("eligible", "sum"), max_train=("n_train", "max"), max_test=("n_test", "max"))
        groups.reset_index().assign(k=k).to_csv(HERE / f"data/whole_map_cluster_coverage_k{k}.csv", index=False)
        coverage[int(k)] = {"n_folds": len(groups), "at_least_two_eligible": int((groups.eligible >= 2).sum()),
                            "median_largest_training_cluster": float(groups.max_train.median()),
                            "median_largest_test_cluster": float(groups.max_test.median())}
    comparisons["cluster_coverage"] = coverage
    (HERE / "data/whole_map_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    timings = pd.read_csv(HERE / "data/whole_map_production_timings.csv")
    manifest = json.loads((HERE / "data/whole_map_raw_manifest.json").read_text())
    jobs = json.loads((HERE / "data/whole_map_jobs.json").read_text())
    assert manifest["n_maps"] == 19400 and len(timings) == 97 and timings.unfinished_rollouts.sum() == 0
    assert {r["worker_sha256"] for r in manifest["markers"]} == {jobs[-1]["worker_sha256"]}
    execution = {"job_id": jobs[-1]["job_id"], "n_proteins": 97, "n_maps": 19400, "unfinished_rollouts": 0,
                 "gpu": timings.gpu_name.unique().tolist(), "workers": timings.hostname.unique().tolist(),
                 "pure_inference_seconds": float(timings.elapsed_seconds.sum()),
                 "one_time_model_load_seconds": float(timings.model_load_seconds.iloc[0]),
                 "one_time_model_stage_seconds": float(timings.model_stage_seconds.iloc[0]),
                 "generated_tokens": int(timings.generated_tokens.sum()),
                 "raw_archive_bytes": json.loads((HERE / "data/whole_map_archive.json").read_text())["bytes"]}
    (HERE / "data/whole_map_execution_summary.json").write_text(json.dumps(execution, indent=2) + "\n")


if __name__ == "__main__":
    main()
