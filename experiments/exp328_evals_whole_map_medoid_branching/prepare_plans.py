"""Build frozen truth-free branch plans from 50 existing iid warm-up maps."""

import argparse
import json
from pathlib import Path

import pandas as pd

from map_policy import make_seed_plans

HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = (
    HERE.parent / "exp326_evals_contact_cluster_branching" / "data" / "targets.csv"
)


def selected_targets(targets: pd.DataFrame, selection: str) -> pd.DataFrame:
    """Select one preregistered target cohort."""
    if selection == "natural-dev":
        return targets[(targets.cohort == "eval-val") & (targets.split == "dev")]
    if selection == "natural-heldout":
        return targets[(targets.cohort == "eval-val") & (targets.split == "test")]
    if selection == "foldswitch-dev":
        return targets[
            (targets.cohort == "foldswitch")
            & (targets.split == "dev")
            & targets.primary.astype(bool)
        ]
    raise ValueError(f"unknown selection: {selection}")


def load_warmup(path: Path) -> list[list[list[int]]]:
    """Read the first 50 ordered rollout maps from one parquet."""
    frame = pd.read_parquet(path).sort_values("rollout")
    if len(frame) < 50 or not set(range(50)).issubset(set(frame.rollout.astype(int))):
        raise ValueError(f"{path}: incomplete first 50 rollouts")
    return frame.set_index("rollout").loc[range(50), "contacts"].tolist()


def main() -> None:
    """Write paired medoid and geometrically spread random plans."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=["natural-dev", "natural-heldout", "foldswitch-dev"],
        required=True,
    )
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args()
    targets = selected_targets(pd.read_csv(args.targets), args.selection)
    expected = {"natural-dev": 16, "natural-heldout": 81, "foldswitch-dev": 15}
    if len(targets) != expected[args.selection]:
        raise ValueError(
            f"{args.selection}: expected {expected[args.selection]}, got {len(targets)}"
        )
    rows, diagnostics = [], []
    for target in targets.sort_values(["L", "stem"]).itertuples():
        maps = load_warmup(args.warmup_root / target.cohort / f"{target.stem}.parquet")
        for bundle_size in (8, 16):
            medoid, random_control, summary = make_seed_plans(
                maps,
                bundle_size,
                int(target.L),
                seed=328_000 + bundle_size * 1_000 + int(target.L),
            )
            diagnostics.append(
                {
                    "selection": args.selection,
                    "cohort": target.cohort,
                    "stem": target.stem,
                    "L": int(target.L),
                    "bundle_size": bundle_size,
                    **summary,
                }
            )
            for arm, plans in (
                (f"medoid_k{bundle_size}", medoid),
                (f"random_k{bundle_size}", random_control),
            ):
                for rollout, plan in enumerate(plans):
                    rows.append(
                        {
                            "selection": args.selection,
                            "cohort": target.cohort,
                            "stem": target.stem,
                            "L": int(target.L),
                            "arm": arm,
                            "rollout": rollout,
                            "bundle_json": json.dumps(plan.bundle),
                            "source_rollout": plan.source_rollout,
                            "cluster_id": plan.cluster_id,
                            "cluster_visits": plan.cluster_visits,
                            "medoid_rollout": plan.medoid_rollout,
                            "n_clusters": plan.n_clusters,
                            "silhouette": plan.silhouette,
                            "used_fallback": plan.used_fallback,
                        }
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / f"{args.selection}_plans.csv", index=False)
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(
        args.out / f"{args.selection}_plan_diagnostics.csv", index=False
    )
    print(
        diagnostic_frame.groupby("bundle_size")
        .agg(
            proteins=("stem", "nunique"),
            clusters=("n_clusters", "mean"),
            silhouette=("silhouette", "mean"),
            fallback_fraction=("used_fallback", "mean"),
            min_cluster_size=("min_cluster_size", "mean"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
