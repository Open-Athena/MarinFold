"""Score rollout votes as inter-chain R-precision on the frozen benchmark."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = HERE / "data/foldbench_complex_contact_eval_targets.parquet"
REQUIRED_VOTE_COLUMNS = {"dataset", "stem", "L", "i", "j", "votes"}


def score_paths(inputs: list[Path]) -> list[Path]:
    """Expand score files and directories into a stable parquet list."""
    paths = []
    for value in inputs:
        if value.is_dir():
            paths.extend(value.rglob("*.parquet"))
        else:
            paths.append(value)
    return sorted(set(paths))


def load_votes(inputs: list[Path]) -> dict[str, dict[tuple[int, int], int]]:
    """Load sparse positive vote counts, ignoring non-score parquet files."""
    votes: dict[str, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    matched = 0
    for path in score_paths(inputs):
        parquet = pq.ParquetFile(path)
        if not REQUIRED_VOTE_COLUMNS.issubset(parquet.schema_arrow.names):
            continue
        matched += 1
        for row in parquet.read(columns=sorted(REQUIRED_VOTE_COLUMNS)).to_pylist():
            pair = tuple(sorted((int(row["i"]), int(row["j"]))))
            votes[row["stem"]][pair] += int(row["votes"])
    if matched == 0:
        raise ValueError("No parquet file had the rollout vote schema")
    return votes


def target_r_precision(target: dict, votes: dict[tuple[int, int], int]) -> dict:
    """Score one target over resolved cross-chain residue pairs only."""
    left, right = target["resolved_positions_by_chain"]
    candidates = [(int(i), int(j)) for i in left for j in right]
    truth = {tuple(map(int, pair)) for pair in target["gt_contacts"]}
    if len(candidates) != target["n_resolved_pairs"]:
        raise ValueError(f"{target['stem']}: candidate-pair count changed")
    candidate_set = set(candidates)
    if not truth.issubset(candidate_set):
        raise ValueError(f"{target['stem']}: ground truth falls outside universe")
    # This is exp89's stable top-R convention applied to the cross-chain
    # row-major universe: descending score, preserving candidate order on ties.
    scores = np.asarray([votes.get(pair, 0) for pair in candidates])
    order = np.argsort(-scores, kind="mergesort")
    top = [candidates[index] for index in order[: len(truth)]]
    correct = sum(pair in truth for pair in top)
    invalid_positive = sum(
        count > 0 and pair not in candidate_set for pair, count in votes.items()
    )
    return {
        "target_id": target["target_id"],
        "dataset": target["dataset"],
        "stem": target["stem"],
        "split": target["split"],
        "group_id": target["group_id"],
        "complex_type": target["complex_type"],
        "n_residues": target["L"],
        "n_candidate": len(candidates),
        "n_true": len(truth),
        "n_top": len(top),
        "n_correct": correct,
        "r_precision": correct / len(truth),
        "random_r_precision": len(truth) / len(candidates),
        "n_positive_predictions": sum(score > 0 for score in scores),
        "n_invalid_positive_predictions": invalid_positive,
    }


def bootstrap_groups(rows: list[dict], draws: int, seed: int) -> tuple[float, float]:
    """Bootstrap mean per-complex R-precision over independent groups."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row["r_precision"])
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        values = [value for group in sampled for value in grouped[str(group)]]
        samples[index] = np.mean(values)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def aggregate(rows: list[dict], draws: int, seed: int) -> list[dict]:
    """Aggregate development, test, and complete cuts."""
    output = []
    for split in ("dev", "test", "all"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        low, high = bootstrap_groups(selected, draws, seed)
        output.append(
            {
                "split": split,
                "n_targets": len(selected),
                "n_groups": len({row["group_id"] for row in selected}),
                "mean_r_precision": float(
                    np.mean([row["r_precision"] for row in selected])
                ),
                "group_bootstrap_95_low": low,
                "group_bootstrap_95_high": high,
                "mean_random_r_precision": float(
                    np.mean([row["random_r_precision"] for row in selected])
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write deterministic result rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Score complete rollout-vote outputs against the frozen ground truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--per-target", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=350)
    args = parser.parse_args()
    targets = pq.read_table(args.targets).to_pylist()
    votes = load_votes(args.scores)
    expected = {target["stem"] for target in targets}
    missing = expected - set(votes)
    if missing:
        raise ValueError(f"Missing rollout scores for {sorted(missing)}")
    extra = set(votes) - expected
    if extra:
        raise ValueError(f"Unexpected rollout score units: {sorted(extra)}")
    rows = [target_r_precision(target, votes[target["stem"]]) for target in targets]
    write_csv(args.per_target, rows)
    aggregate_rows = aggregate(rows, args.bootstrap_draws, args.seed)
    write_csv(args.aggregate, aggregate_rows)
    for row in aggregate_rows:
        print(
            f"{row['split']}: n={row['n_targets']} groups={row['n_groups']} "
            f"R={row['mean_r_precision']:.4f} "
            f"[{row['group_bootstrap_95_low']:.4f}, "
            f"{row['group_bootstrap_95_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
