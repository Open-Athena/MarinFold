"""Create intention-to-treat and cluster-availability effect summaries."""

from pathlib import Path

import numpy as np
import pandas as pd
from analyze_natural import bootstrap_mean_ci

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METRICS = (
    "validity_gated_oracle_r_precision",
    "consensus_r_precision",
    "true_union_recall",
    "mean_pairwise_jaccard",
)


def main() -> None:
    """Bootstrap frozen-policy deltas overall and by truth-free availability."""
    rows = []
    seed = 326_500
    for split in ("dev", "heldout"):
        scores = pd.read_csv(DATA / f"{split}_natural.csv")
        scores = scores[
            (scores.map_variant == "continuation")
            & (scores["range"] == "all")
            & scores.arm.isin(["iid100", "random_k5", "cluster_k5"])
        ]
        diagnostic_name = "natural-dev" if split == "dev" else "natural-heldout"
        diagnostics = pd.read_csv(DATA / f"{diagnostic_name}_plan_diagnostics.csv")
        available = (
            diagnostics[diagnostics.bundle_size == 5]
            .set_index("stem")
            .n_stable_clusters.gt(0)
        )
        pivot = scores.pivot(index="stem", columns="arm")
        availability = available.loc[pivot.index]
        groups = {
            "all": np.ones(len(pivot), dtype=bool),
            "stable_cluster_available": availability.to_numpy(dtype=bool),
            "fallback": ~availability.to_numpy(dtype=bool),
        }
        for group, keep in groups.items():
            for comparator in ("iid100", "random_k5"):
                for metric in METRICS:
                    values = (
                        pivot[metric]["cluster_k5"] - pivot[metric][comparator]
                    ).to_numpy(dtype=float)[keep]
                    low, high = bootstrap_mean_ci(values, seed)
                    rows.append(
                        {
                            "split": split,
                            "availability_group": group,
                            "comparator": comparator,
                            "metric": metric,
                            "n": len(values),
                            "control_mean": float(
                                pivot[metric][comparator].to_numpy()[keep].mean()
                            ),
                            "cluster_policy_mean": float(
                                pivot[metric]["cluster_k5"].to_numpy()[keep].mean()
                            ),
                            "mean_delta": float(values.mean()),
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
                    seed += 1
    output = pd.DataFrame(rows)
    output.to_csv(DATA / "coverage_stratified_deltas.csv", index=False)
    print(
        output[
            output.metric.isin(
                ["validity_gated_oracle_r_precision", "consensus_r_precision"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
