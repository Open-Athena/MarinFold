"""Score fixed-budget iid warm-up plus medoid-branch rollout pools."""

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
METRIC_MODULE_PATH = (
    EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set" / "compute_metrics.py"
)
METRIC_SPEC = importlib.util.spec_from_file_location(
    "exp89_compute_metrics", METRIC_MODULE_PATH
)
if METRIC_SPEC is None or METRIC_SPEC.loader is None:
    raise ImportError(
        f"cannot load established metric implementation: {METRIC_MODULE_PATH}"
    )
METRIC_MODULE = importlib.util.module_from_spec(METRIC_SPEC)
METRIC_SPEC.loader.exec_module(METRIC_MODULE)
metric_rows = METRIC_MODULE.metric_rows
resolved_pairs = METRIC_MODULE.resolved_pairs
true_matrix = METRIC_MODULE.true_matrix

TARGETS = (
    EXPERIMENTS / "exp326_evals_contact_cluster_branching" / "data" / "targets.csv"
)
TRUTH = (
    EXPERIMENTS
    / "exp245_evals_foldbench_held_out_monomers"
    / "data"
    / "gt_universe_scored.jsonl"
)
BRANCH_ARMS = ("medoid_k8", "random_k8", "medoid_k16", "random_k16")
METRICS = (
    "validity_gated_oracle_r_precision",
    "oracle_r_precision",
    "consensus_r_precision",
    "true_union_recall",
    "mean_rollout_precision",
    "mean_pairwise_jaccard",
    "mean_contacts",
)


def ordered_true_pairs(record: dict, region: str) -> set[tuple[int, int]]:
    """Return truth contacts in the resolved evaluation universe."""
    resolved = {int(value) for value in record["resolved"]}
    minimum = 24 if region == "long" else 6
    return {
        (int(left), int(right))
        for left, right, degree in record["contacts"]
        if float(degree) >= 0.001
        and int(right) - int(left) >= minimum
        and int(left) in resolved
        and int(right) in resolved
    }


def canonical_contacts(contacts, record: dict, region: str) -> list[tuple[int, int]]:
    """Canonicalize and filter one emitted map, preserving first occurrence."""
    resolved = {int(value) for value in record["resolved"]}
    minimum = 24 if region == "long" else 6
    result = []
    seen: set[tuple[int, int]] = set()
    for raw_left, raw_right in contacts:
        pair = (min(int(raw_left), int(raw_right)), max(int(raw_left), int(raw_right)))
        if (
            pair in seen
            or pair[0] not in resolved
            or pair[1] not in resolved
            or pair[1] - pair[0] < minimum
        ):
            continue
        seen.add(pair)
        result.append(pair)
    return result


def vote_matrix(maps: list[list[tuple[int, int]]], length: int) -> np.ndarray:
    """Accumulate symmetric map occurrence counts."""
    votes = np.zeros((length, length), dtype=np.float32)
    for contacts in maps:
        for left, right in set(contacts):
            votes[left, right] += 1
            votes[right, left] += 1
    return votes


def consensus_r_precision(
    maps: list[list[tuple[int, int]]], record: dict, region: str
) -> float:
    """Score the vote matrix using exp89's exact R-cutoff evaluator."""
    length = int(record["L"])
    resolved = np.asarray(record["resolved"], dtype=np.int64)
    rows = pd.DataFrame(
        metric_rows(
            vote_matrix(maps, length),
            true_matrix(length, record["contacts"]),
            *resolved_pairs(resolved),
            length,
            with_precision=True,
        )
    )
    return float(rows[(rows["range"] == region) & (rows.cut == "R")].precision.iloc[0])


def rollout_r_precision(
    contacts: list[tuple[int, int]], true: set[tuple[int, int]]
) -> float:
    """Fixed-R precision with missing predictions charged as incorrect."""
    if not true:
        return float("nan")
    ranked = list(dict.fromkeys(contacts))[: len(true)]
    return len(set(ranked) & true) / len(true)


def mean_pairwise_jaccard(maps: list[list[tuple[int, int]]]) -> float:
    """Mean set Jaccard across all rollout pairs."""
    values = []
    for left, right in combinations((set(item) for item in maps), 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))


def score_pool(
    maps: list[list[tuple[int, int]]],
    valid: np.ndarray,
    record: dict,
    region: str,
) -> dict[str, float]:
    """Calculate oracle, consensus, coverage, and diversity for one pool."""
    if len(maps) != 100 or len(valid) != 100:
        raise ValueError("rollout pools must contain exactly 100 maps")
    true = ordered_true_pairs(record, region)
    precisions = np.asarray(
        [rollout_r_precision(contacts, true) for contacts in maps], dtype=float
    )
    gated = np.where(valid, precisions, 0.0)
    union = set().union(*(set(contacts) for contacts in maps))
    return {
        "consensus_r_precision": consensus_r_precision(maps, record, region),
        "oracle_r_precision": float(np.max(precisions)),
        "validity_gated_oracle_r_precision": float(np.max(gated)),
        "branch_validity_gated_oracle_r_precision": float(np.max(gated[50:])),
        "mean_rollout_precision": float(np.mean(precisions)),
        "true_union_recall": len(union & true) / len(true) if true else float("nan"),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(maps),
        "unique_maps": float(len({frozenset(contacts) for contacts in maps})),
        "mean_contacts": float(np.mean([len(contacts) for contacts in maps])),
        "invalid_rollouts": float((~valid).sum()),
    }


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, n_bootstrap: int = 50_000
) -> tuple[float, float]:
    """Return a deterministic protein-bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(
        axis=1
    )
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def load_truth() -> dict[str, dict]:
    """Load natural FoldBench monomer truth records."""
    records = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer":
                records[record["stem"]] = record
    return records


def branch_maps(
    frame: pd.DataFrame, record: dict, region: str, include_seed: bool
) -> tuple[list[list[tuple[int, int]]], float]:
    """Read branch continuations and optionally restore their supplied seeds."""
    maps, repeated = [], []
    for row in frame.sort_values("rollout").itertuples():
        seeds = canonical_contacts(row.seed_contacts, record, region)
        continuation = canonical_contacts(row.contacts, record, region)
        seed_set = set(seeds)
        repeated.append(len(seed_set & set(continuation)) / max(len(seed_set), 1))
        if include_seed:
            maps.append(list(dict.fromkeys(seeds + continuation)))
        else:
            maps.append(
                [contact for contact in continuation if contact not in seed_set]
            )
    return maps, float(np.mean(repeated))


def analyze(
    split: str, warmup_root: Path, branch_root: Path, branch_arms: tuple[str, ...]
) -> pd.DataFrame:
    """Score iid100 and 50-iid plus 50-branch pools."""
    targets = pd.read_csv(TARGETS)
    targets = targets[(targets.cohort == "eval-val") & (targets.split == split)]
    truth = load_truth()
    rows = []
    for target in targets.sort_values(["L", "stem"]).itertuples():
        iid = pd.read_parquet(warmup_root / "eval-val" / f"{target.stem}.parquet")
        iid = iid.sort_values("rollout").iloc[:100]
        if list(iid.rollout.astype(int)) != list(range(100)):
            raise ValueError(f"{target.stem}: iid baseline is not rollouts 0..99")
        iid_valid = iid.finished.to_numpy(dtype=bool) & (
            iid.malformed_contacts.to_numpy(dtype=int) == 0
        )
        for region in ("all", "long"):
            iid_maps = [
                canonical_contacts(value, truth[target.stem], region)
                for value in iid.contacts
            ]
            rows.append(
                {
                    "arm": "iid100",
                    "map_variant": "continuation",
                    "split": split,
                    "stem": target.stem,
                    "L": int(target.L),
                    "range": region,
                    "seed_true_fraction": float("nan"),
                    "continuation_seed_repeat_fraction": float("nan"),
                    **score_pool(iid_maps, iid_valid, truth[target.stem], region),
                }
            )
            for arm in branch_arms:
                branch = pd.read_parquet(
                    branch_root / arm / "eval-val" / f"{target.stem}.parquet"
                ).sort_values("rollout")
                if list(branch.rollout.astype(int)) != list(range(50)):
                    raise ValueError(
                        f"{arm}/{target.stem}: branch is not rollouts 0..49"
                    )
                branch_valid = branch.finished.to_numpy(dtype=bool) & (
                    branch.malformed_contacts.to_numpy(dtype=int) == 0
                )
                true = ordered_true_pairs(truth[target.stem], region)
                seeds = [
                    canonical_contacts(value, truth[target.stem], region)
                    for value in branch.seed_contacts
                ]
                flattened = [contact for seed in seeds for contact in seed]
                seed_true_fraction = (
                    sum(contact in true for contact in flattened) / len(flattened)
                    if flattened
                    else float("nan")
                )
                for variant, include_seed in (
                    ("continuation", False),
                    ("complete", True),
                ):
                    continuations, repeat_fraction = branch_maps(
                        branch, truth[target.stem], region, include_seed
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "map_variant": variant,
                            "split": split,
                            "stem": target.stem,
                            "L": int(target.L),
                            "range": region,
                            "seed_true_fraction": seed_true_fraction,
                            "continuation_seed_repeat_fraction": repeat_fraction,
                            **score_pool(
                                iid_maps[:50] + continuations,
                                np.concatenate([iid_valid[:50], branch_valid]),
                                truth[target.stem],
                                region,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def paired_deltas(
    scores: pd.DataFrame, map_variant: str = "continuation"
) -> pd.DataFrame:
    """Compute protein-paired arm deltas against iid and matched random."""
    primary = scores[(scores.map_variant == map_variant) | (scores.arm == "iid100")]
    rows, seed = [], 328_000
    arms = [arm for arm in BRANCH_ARMS if arm in set(primary.arm)]
    for region in ("all", "long"):
        for arm in arms:
            comparators = ["iid100"]
            random_control = arm.replace("medoid", "random")
            if arm.startswith("medoid") and random_control in set(primary.arm):
                comparators.append(random_control)
            candidate = primary[
                (primary.arm == arm) & (primary["range"] == region)
            ].set_index("stem")
            for comparator in comparators:
                control = primary[
                    (primary.arm == comparator) & (primary["range"] == region)
                ].set_index("stem")
                candidate = candidate.loc[control.index]
                for metric in METRICS:
                    values = (candidate[metric] - control[metric]).to_numpy(dtype=float)
                    low, high = bootstrap_mean_ci(values, seed)
                    rows.append(
                        {
                            "arm": arm,
                            "comparator": comparator,
                            "range": region,
                            "metric": metric,
                            "n": len(values),
                            "control_mean": float(control[metric].mean()),
                            "arm_mean": float(candidate[metric].mean()),
                            "mean_delta": float(values.mean()),
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
                    seed += 1
    return pd.DataFrame(rows)


def main() -> None:
    """Run analysis and write tidy committed result tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--warmup-root", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, default=HERE / "_cache")
    parser.add_argument("--arms", default=",".join(BRANCH_ARMS))
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    if not set(arms) <= set(BRANCH_ARMS):
        raise ValueError(f"unknown arms: {set(arms) - set(BRANCH_ARMS)}")
    scores = analyze(args.split, args.warmup_root, args.branch_root, arms)
    deltas = paired_deltas(scores, "continuation")
    complete_deltas = paired_deltas(scores, "complete")
    scores.to_csv(HERE / "data" / f"{args.prefix}_natural.csv", index=False)
    deltas.to_csv(HERE / "data" / f"{args.prefix}_paired_deltas.csv", index=False)
    complete_deltas.to_csv(
        HERE / "data" / f"{args.prefix}_complete_paired_deltas.csv", index=False
    )
    summary = scores.groupby(["arm", "map_variant", "range"], as_index=False).agg(
        n=("stem", "nunique"),
        consensus_r_precision=("consensus_r_precision", "mean"),
        validity_gated_oracle_r_precision=("validity_gated_oracle_r_precision", "mean"),
        oracle_r_precision=("oracle_r_precision", "mean"),
        true_union_recall=("true_union_recall", "mean"),
        mean_rollout_precision=("mean_rollout_precision", "mean"),
        mean_pairwise_jaccard=("mean_pairwise_jaccard", "mean"),
        mean_contacts=("mean_contacts", "mean"),
        seed_true_fraction=("seed_true_fraction", "mean"),
        continuation_seed_repeat_fraction=("continuation_seed_repeat_fraction", "mean"),
        invalid_rollouts=("invalid_rollouts", "sum"),
    )
    summary.to_csv(HERE / "data" / f"{args.prefix}_natural_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPrimary oracle and consensus deltas")
    print(
        deltas[
            deltas.metric.isin(
                ("validity_gated_oracle_r_precision", "consensus_r_precision")
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
