"""Score guidance arms for contact accuracy, useful diversity, and fold modes."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from selection_policy import canonical, diverse_indices

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix  # noqa: E402

TRUTH = (
    EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data"
    / "gt_universe_scored.jsonl"
)


def ordered_true_pairs(record: dict, region: str) -> set[tuple[int, int]]:
    """Ground-truth contacts in the resolved candidate universe."""
    resolved = set(int(value) for value in record["resolved"])
    minimum = 24 if region == "long" else 6
    return {
        (int(i), int(j))
        for i, j, degree in record["contacts"]
        if float(degree) >= 0.001
        and int(j) - int(i) >= minimum
        and int(i) in resolved
        and int(j) in resolved
    }


def predicted_maps(frame: pd.DataFrame, record: dict, region: str) -> list[list[tuple[int, int]]]:
    """Order-preserving per-rollout maps restricted to resolved positions."""
    resolved = set(int(value) for value in record["resolved"])
    minimum = 24 if region == "long" else 6
    maps = []
    for row in frame.sort_values("rollout").itertuples():
        contacts = []
        for left, right in row.contacts:
            pair = (min(int(left), int(right)), max(int(left), int(right)))
            if pair[0] in resolved and pair[1] in resolved and pair[1] - pair[0] >= minimum:
                contacts.append(pair)
        maps.append(contacts)
    return maps


def r_precision(votes: np.ndarray, record: dict, region: str) -> float:
    """Score one vote matrix with exp89's unchanged candidate universe."""
    length = int(record["L"])
    resolved = np.asarray(record["resolved"], dtype=np.int64)
    rows = pd.DataFrame(
        metric_rows(
            votes,
            true_matrix(length, record["contacts"]),
            *resolved_pairs(resolved),
            length,
            with_precision=True,
        )
    )
    return float(rows[(rows["range"] == region) & (rows.cut == "R")].precision.iloc[0])


def vote_matrix(maps: list[list[tuple[int, int]]], length: int) -> np.ndarray:
    """Accumulate symmetric occurrence counts from complete rollout maps."""
    votes = np.zeros((length, length), dtype=np.float32)
    for contacts in maps:
        for left, right in set(contacts):
            votes[left, right] += 1
            votes[right, left] += 1
    return votes


def rollout_r_precision(
    contacts: list[tuple[int, int]], true: set[tuple[int, int]]
) -> float:
    """Score one ordered rollout at exp89's R cutoff.

    Contact statements are ranked by emission order. Repeated statements only
    occupy their first rank, and a rollout with fewer than R distinct contacts
    receives zero credit for the unfilled ranks, matching the fixed-size top-R
    denominator in exp89.
    """
    if not true:
        return float("nan")
    ranked = list(dict.fromkeys(contacts))[:len(true)]
    return len(set(ranked) & true) / len(true)


def mean_pairwise_jaccard(maps: list[list[tuple[int, int]]]) -> float:
    """Mean set Jaccard, with two empty maps defined as identical."""
    sets = [set(contacts) for contacts in maps]
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else float("nan")


def log_n_auc(values: list[float], budgets: list[int]) -> float:
    """Normalized area under a metric curve parameterized by log rollout count."""
    if len(values) == 1:
        return values[0]
    x = np.log(np.asarray(budgets, dtype=float))
    return float(np.trapezoid(values, x=x) / (x[-1] - x[0]))


def score_natural_mode(mode: str, budgets: list[int], split: str) -> pd.DataFrame:
    """Score one mode on natural eval-val proteins."""
    targets = pd.read_csv(HERE / "data" / "targets.csv")
    targets = targets[(targets.cohort == "eval-val") & (targets.split == split)]
    truth = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer":
                truth[record["stem"]] = record
    rows = []
    for target in targets.itertuples():
        path = HERE / "_cache" / mode / "eval-val" / f"{target.stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path).sort_values("rollout")
        if len(frame) < max(budgets):
            raise ValueError(f"{mode}/{target.stem}: only {len(frame)} rollouts")
        record = truth[target.stem]
        for region in ("all", "long"):
            maps = predicted_maps(frame.iloc[:max(budgets)], record, region)
            true = ordered_true_pairs(record, region)
            union_recalls, consensus = [], []
            for budget in budgets:
                prefix = maps[:budget]
                prefix_frame = frame.iloc[:budget]
                union = set().union(*(set(contacts) for contacts in prefix))
                union_recalls.append(len(union & true) / len(true) if true else float("nan"))
                consensus.append(r_precision(vote_matrix(prefix, int(target.L)), record, region))
                rollout_precision = [rollout_r_precision(contacts, true) for contacts in prefix]
                valid = (
                    prefix_frame.finished.to_numpy(dtype=bool)
                    & (prefix_frame.malformed_contacts.to_numpy(dtype=int) == 0)
                )
                gated_precision = [
                    score if is_valid else 0.0
                    for score, is_valid in zip(rollout_precision, valid, strict=True)
                ]
                rows.append({
                    "mode": mode,
                    "split": split,
                    "stem": target.stem,
                    "L": int(target.L),
                    "range": region,
                    "N": budget,
                    "consensus_r_precision": consensus[-1],
                    "true_union_recall": union_recalls[-1],
                    "mean_rollout_precision": float(np.mean(rollout_precision)),
                    "mean_validity_gated_rollout_r_precision": float(np.mean(gated_precision)),
                    "oracle_r_precision": float(np.max(rollout_precision)),
                    "validity_gated_oracle_r_precision": float(np.max(gated_precision)),
                    "mean_pairwise_jaccard": mean_pairwise_jaccard(prefix),
                    "unique_maps": len({frozenset(contacts) for contacts in prefix}),
                    "mean_contacts": float(np.mean([len(contacts) for contacts in prefix])),
                    "finished": int(frame.iloc[:budget].finished.sum()),
                    "malformed": int(frame.iloc[:budget].malformed_contacts.sum()),
                    "invalid_rollouts": int((~valid).sum()),
                })
            auc = log_n_auc(union_recalls, budgets)
            for row in rows[-len(budgets):]:
                row["union_recall_log_n_auc"] = auc
    return pd.DataFrame(rows)


def contact_log_ratio_diagnostic(mode: str, split: str) -> pd.DataFrame:
    """Test whether emitted true contacts carry a larger native/null ratio."""
    targets = pd.read_csv(HERE / "data" / "targets.csv")
    targets = targets[(targets.cohort == "eval-val") & (targets.split == split)]
    truth = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer":
                truth[record["stem"]] = ordered_true_pairs(record, "all")
    rows = []
    for target in targets.itertuples():
        frame = pd.read_parquet(HERE / "_cache" / mode / "eval-val" / f"{target.stem}.parquet")
        labels, scores = [], []
        for rollout in frame.itertuples():
            for pair, score in zip(rollout.contacts, rollout.contact_log_ratios):
                ordered = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
                labels.append(ordered in truth[target.stem])
                scores.append(float(score))
        labels_array = np.asarray(labels, dtype=bool)
        scores_array = np.asarray(scores, dtype=float)
        auc = (
            float(roc_auc_score(labels_array, scores_array))
            if labels_array.any() and (~labels_array).any() else float("nan")
        )
        rows.append({
            "mode": mode,
            "split": split,
            "stem": target.stem,
            "n_contacts": len(labels),
            "n_true": int(labels_array.sum()),
            "mean_true_log_ratio": float(scores_array[labels_array].mean())
            if labels_array.any() else float("nan"),
            "mean_false_log_ratio": float(scores_array[~labels_array].mean())
            if (~labels_array).any() else float("nan"),
            "true_false_log_ratio_delta": (
                float(scores_array[labels_array].mean() - scores_array[~labels_array].mean())
                if labels_array.any() and (~labels_array).any() else float("nan")
            ),
            "log_ratio_auc": auc,
        })
    return pd.DataFrame(rows)


def fold_scores(contacts, target: dict) -> dict:
    """Apply exp304's fold-specific contact screen to one map."""
    common = {int(position) for position in target["common_positions"]}
    predicted = {pair for pair in canonical(contacts) if pair[0] in common and pair[1] in common}
    fold1, fold2 = canonical(target["contacts_fold1"]), canonical(target["contacts_fold2"])
    unique1, unique2 = fold1 - fold2, fold2 - fold1
    low, high = int(target["fs_lo"]), int(target["fs_hi"])

    def touches(pair: tuple[int, int]) -> bool:
        return low <= pair[0] < high or low <= pair[1] < high

    unique1_region = {pair for pair in unique1 if touches(pair)}
    unique2_region = {pair for pair in unique2 if touches(pair)}
    pred_region = {pair for pair in predicted if touches(pair)}
    recall1 = len(pred_region & unique1_region) / len(unique1_region) if unique1_region else np.nan
    recall2 = len(pred_region & unique2_region) / len(unique2_region) if unique2_region else np.nan
    enrichment = recall1 - recall2
    return {
        "recall_fold1_region": recall1,
        "recall_fold2_region": recall2,
        "enrichment": enrichment,
        "fold1_hit": bool(recall1 >= 0.25 and enrichment >= 0.10),
        "fold2_hit": bool(recall2 >= 0.25 and enrichment <= -0.10),
    }


def score_foldswitch_mode(mode: str, split: str, n_rollouts: int) -> pd.DataFrame:
    """Score oracle-pool and blind-shortlist fold-mode coverage."""
    targets = pd.read_csv(HERE / "data" / "targets.csv")
    targets = targets[
        (targets.cohort == "foldswitch") & (targets.split == split) & targets.primary
    ]
    truth = {
        record["pair_id"]: record
        for record in pd.read_parquet(HERE / "data" / "foldswitch_truth.parquet").to_dict("records")
    }
    rows = []
    for target in targets.itertuples():
        frame = pd.read_parquet(
            HERE / "_cache" / mode / "foldswitch" / f"{target.stem}.parquet"
        ).sort_values("rollout").iloc[:n_rollouts]
        if len(frame) != n_rollouts:
            raise ValueError(f"{mode}/{target.stem}: incomplete fold-switch pool")
        maps = [canonical(contacts) for contacts in frame.contacts]
        scored = [fold_scores(contacts, truth[target.stem]) for contacts in maps]
        valid = (
            frame.finished.to_numpy(dtype=bool)
            & (frame.malformed_contacts.to_numpy(dtype=int) == 0)
        )
        for score, is_valid in zip(scored, valid, strict=True):
            if not is_valid:
                score["fold1_hit"] = False
                score["fold2_hit"] = False
        selected = diverse_indices(maps, min(16, len(maps)))
        oracle_fold1 = any(item["fold1_hit"] for item in scored)
        oracle_fold2 = any(item["fold2_hit"] for item in scored)
        blind_fold1 = any(scored[index]["fold1_hit"] for index in selected)
        blind_fold2 = any(scored[index]["fold2_hit"] for index in selected)
        rows.append({
            "mode": mode,
            "split": split,
            "pair_id": target.stem,
            "N": n_rollouts,
            "oracle_fold1": oracle_fold1,
            "oracle_fold2": oracle_fold2,
            "oracle_dual": oracle_fold1 and oracle_fold2,
            "blind_fold1": blind_fold1,
            "blind_fold2": blind_fold2,
            "blind_dual": blind_fold1 and blind_fold2,
            "mean_pairwise_jaccard": mean_pairwise_jaccard([list(value) for value in maps]),
            "unique_maps": len(set(maps)),
            "mean_contacts": float(np.mean([len(value) for value in maps])),
            "finished": int(frame.finished.sum()),
            "malformed": int(frame.malformed_contacts.sum()),
            "invalid_rollouts": int((~valid).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Score one or more fetched modes and write tidy development/full tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", required=True, help="comma-separated mode names")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--budgets", default="1,2,5,10,20")
    parser.add_argument("--foldswitch", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    modes = args.modes.split(",")
    budgets = [int(value) for value in args.budgets.split(",")]
    natural = pd.concat(
        [score_natural_mode(mode, budgets, args.split) for mode in modes], ignore_index=True
    )
    diagnostics = pd.concat(
        [contact_log_ratio_diagnostic(mode, args.split) for mode in modes], ignore_index=True
    )
    destination = HERE / "data"
    destination.mkdir(exist_ok=True)
    natural.to_csv(destination / f"{args.output_prefix}_natural.csv", index=False)
    diagnostics.to_csv(destination / f"{args.output_prefix}_log_ratio.csv", index=False)
    final = natural[natural.N == max(budgets)]
    summary = final.groupby(["mode", "range"], as_index=False).agg(
        n=("stem", "nunique"),
        consensus_r_precision=("consensus_r_precision", "mean"),
        true_union_recall=("true_union_recall", "mean"),
        union_recall_log_n_auc=("union_recall_log_n_auc", "mean"),
        mean_rollout_precision=("mean_rollout_precision", "mean"),
        mean_validity_gated_rollout_r_precision=(
            "mean_validity_gated_rollout_r_precision", "mean"
        ),
        oracle_r_precision=("oracle_r_precision", "mean"),
        validity_gated_oracle_r_precision=("validity_gated_oracle_r_precision", "mean"),
        mean_pairwise_jaccard=("mean_pairwise_jaccard", "mean"),
        mean_contacts=("mean_contacts", "mean"),
        malformed=("malformed", "sum"),
        finished=("finished", "sum"),
        invalid_rollouts=("invalid_rollouts", "sum"),
    )
    summary.to_csv(destination / f"{args.output_prefix}_natural_summary.csv", index=False)
    print(summary.to_string(index=False))
    diagnostic_summary = diagnostics.groupby("mode").agg(
        mean_log_ratio_auc=("log_ratio_auc", "mean"),
        mean_true_false_delta=("true_false_log_ratio_delta", "mean"),
    )
    print("\nlog-ratio diagnostic\n", diagnostic_summary.to_string())
    if args.foldswitch:
        fold = pd.concat(
            [score_foldswitch_mode(mode, args.split, max(budgets)) for mode in modes],
            ignore_index=True,
        )
        fold.to_csv(destination / f"{args.output_prefix}_foldswitch.csv", index=False)
        fold_summary = fold.groupby("mode", as_index=False).agg(
            n=("pair_id", "nunique"),
            oracle_dual=("oracle_dual", "sum"),
            blind_dual=("blind_dual", "sum"),
            mean_pairwise_jaccard=("mean_pairwise_jaccard", "mean"),
            mean_contacts=("mean_contacts", "mean"),
            finished=("finished", "sum"),
            malformed=("malformed", "sum"),
            invalid_rollouts=("invalid_rollouts", "sum"),
        )
        fold_summary.to_csv(
            destination / f"{args.output_prefix}_foldswitch_summary.csv", index=False
        )
        print("\nfold-switching\n", fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
