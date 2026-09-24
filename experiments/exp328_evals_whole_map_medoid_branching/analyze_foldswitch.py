"""Measure whether medoid branching visits both fold-switch contact modes."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
TARGETS = (
    EXPERIMENTS / "exp326_evals_contact_cluster_branching" / "data" / "targets.csv"
)
TRUTH = (
    EXPERIMENTS
    / "exp326_evals_contact_cluster_branching"
    / "data"
    / "foldswitch_truth.csv"
)
ARMS = ("medoid_k8", "random_k8", "medoid_k16", "random_k16")


def canonical(contacts) -> frozenset[tuple[int, int]]:
    """Canonicalize a contact map."""
    return frozenset(
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in contacts
        if int(left) != int(right)
    )


def fold_scores(contacts, target: dict) -> dict[str, float | bool]:
    """Apply the established fold-specific contact-region screen."""
    common = {int(position) for position in target["common_positions"]}
    predicted = {
        pair for pair in canonical(contacts) if pair[0] in common and pair[1] in common
    }
    fold1 = canonical(target["contacts_fold1"])
    fold2 = canonical(target["contacts_fold2"])
    unique1, unique2 = fold1 - fold2, fold2 - fold1
    low, high = int(target["fs_lo"]), int(target["fs_hi"])

    def touches(pair: tuple[int, int]) -> bool:
        return low <= pair[0] < high or low <= pair[1] < high

    unique1_region = {pair for pair in unique1 if touches(pair)}
    unique2_region = {pair for pair in unique2 if touches(pair)}
    predicted_region = {pair for pair in predicted if touches(pair)}
    recall1 = len(predicted_region & unique1_region) / len(unique1_region)
    recall2 = len(predicted_region & unique2_region) / len(unique2_region)
    enrichment = recall1 - recall2
    return {
        "recall_fold1_region": recall1,
        "recall_fold2_region": recall2,
        "enrichment": enrichment,
        "fold1_hit": bool(recall1 >= 0.25 and enrichment >= 0.10),
        "fold2_hit": bool(recall2 >= 0.25 and enrichment <= -0.10),
    }


def mean_pairwise_jaccard(maps: list[frozenset[tuple[int, int]]]) -> float:
    """Mean set Jaccard across one rollout pool."""
    values = []
    for left, right in combinations(maps, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))


def branch_maps(
    frame: pd.DataFrame, include_seed: bool
) -> list[frozenset[tuple[int, int]]]:
    """Read branch continuations with prompt contacts excluded or restored."""
    maps = []
    for row in frame.sort_values("rollout").itertuples():
        seed = canonical(row.seed_contacts)
        continuation = canonical(row.contacts) - seed
        maps.append((continuation | seed) if include_seed else continuation)
    return maps


def score_pool(
    maps: list[frozenset[tuple[int, int]]], valid: np.ndarray, target: dict
) -> dict[str, float | bool]:
    """Score oracle mode coverage and structural diversity for 100 maps."""
    if len(maps) != 100 or len(valid) != 100:
        raise ValueError("fold-switch pools must contain exactly 100 maps")
    scored = [fold_scores(contacts, target) for contacts in maps]
    fold1_hits = [
        bool(item["fold1_hit"]) and bool(ok)
        for item, ok in zip(scored, valid, strict=True)
    ]
    fold2_hits = [
        bool(item["fold2_hit"]) and bool(ok)
        for item, ok in zip(scored, valid, strict=True)
    ]
    oracle_fold1, oracle_fold2 = any(fold1_hits), any(fold2_hits)
    return {
        "oracle_fold1": oracle_fold1,
        "oracle_fold2": oracle_fold2,
        "oracle_dual": oracle_fold1 and oracle_fold2,
        "fold1_hit_rollouts": float(sum(fold1_hits)),
        "fold2_hit_rollouts": float(sum(fold2_hits)),
        "max_fold1_recall": float(max(item["recall_fold1_region"] for item in scored)),
        "max_fold2_recall": float(max(item["recall_fold2_region"] for item in scored)),
        "max_fold1_enrichment": float(max(item["enrichment"] for item in scored)),
        "max_fold2_enrichment": float(-min(item["enrichment"] for item in scored)),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(maps),
        "unique_maps": float(len(set(maps))),
        "mean_contacts": float(np.mean([len(contacts) for contacts in maps])),
        "invalid_rollouts": float((~valid).sum()),
    }


def main() -> None:
    """Score iid and branch pools on the 15 frozen fold-switch dev pairs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, default=HERE / "_cache")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--prefix", default="dev")
    args = parser.parse_args()
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    if not set(arms) <= set(ARMS):
        raise ValueError(f"unknown arms: {set(arms) - set(ARMS)}")
    targets = pd.read_csv(TARGETS)
    targets = targets[
        (targets.cohort == "foldswitch")
        & (targets.split == "dev")
        & targets.primary.astype(bool)
    ]
    truth_frame = pd.read_csv(TRUTH)
    for column in ("contacts_fold1", "contacts_fold2", "common_positions"):
        truth_frame[column] = truth_frame[column].map(json.loads)
    truth = {record["pair_id"]: record for record in truth_frame.to_dict("records")}
    rows = []
    for target in targets.sort_values(["L", "stem"]).itertuples():
        iid = (
            pd.read_parquet(args.warmup_root / "foldswitch" / f"{target.stem}.parquet")
            .sort_values("rollout")
            .iloc[:100]
        )
        if list(iid.rollout.astype(int)) != list(range(100)):
            raise ValueError(f"{target.stem}: iid baseline is not rollouts 0..99")
        iid_maps = [canonical(contacts) for contacts in iid.contacts]
        iid_valid = iid.finished.to_numpy(dtype=bool) & (
            iid.malformed_contacts.to_numpy(dtype=int) == 0
        )
        rows.append(
            {
                "arm": "iid100",
                "map_variant": "continuation",
                "pair_id": target.stem,
                **score_pool(iid_maps, iid_valid, truth[target.stem]),
            }
        )
        for arm in arms:
            branch = pd.read_parquet(
                args.branch_root / arm / "foldswitch" / f"{target.stem}.parquet"
            ).sort_values("rollout")
            branch_valid = branch.finished.to_numpy(dtype=bool) & (
                branch.malformed_contacts.to_numpy(dtype=int) == 0
            )
            valid = np.concatenate([iid_valid[:50], branch_valid])
            for variant, include_seed in (("continuation", False), ("complete", True)):
                rows.append(
                    {
                        "arm": arm,
                        "map_variant": variant,
                        "pair_id": target.stem,
                        **score_pool(
                            iid_maps[:50] + branch_maps(branch, include_seed),
                            valid,
                            truth[target.stem],
                        ),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(HERE / "data" / f"{args.prefix}_foldswitch.csv", index=False)
    summary = results.groupby(["arm", "map_variant"], as_index=False).agg(
        n=("pair_id", "nunique"),
        oracle_dual=("oracle_dual", "sum"),
        oracle_fold1=("oracle_fold1", "sum"),
        oracle_fold2=("oracle_fold2", "sum"),
        fold1_hit_rollouts=("fold1_hit_rollouts", "mean"),
        fold2_hit_rollouts=("fold2_hit_rollouts", "mean"),
        max_fold1_recall=("max_fold1_recall", "mean"),
        max_fold2_recall=("max_fold2_recall", "mean"),
        mean_pairwise_jaccard=("mean_pairwise_jaccard", "mean"),
        mean_contacts=("mean_contacts", "mean"),
        invalid_rollouts=("invalid_rollouts", "sum"),
    )
    summary.to_csv(HERE / "data" / f"{args.prefix}_foldswitch_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
