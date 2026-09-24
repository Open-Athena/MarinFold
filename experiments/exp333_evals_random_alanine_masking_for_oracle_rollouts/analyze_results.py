"""Score alanine-masked rollout pools and freeze the confirmation policy."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
TARGETS = (
    EXPERIMENTS
    / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"
    / "data"
    / "targets.csv"
)
TRUTH = (
    EXPERIMENTS
    / "exp245_evals_foldbench_held_out_monomers"
    / "data"
    / "gt_universe_scored.jsonl"
)
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix

BUDGETS = [1, 2, 5, 10, 25, 50, 100]
BASELINE = "iid100"


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, n_bootstrap: int = 50_000
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a paired mean."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def truth_records() -> dict[str, dict[str, Any]]:
    """Load the fixed FoldBench monomer measurement records."""
    records = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer":
                records[record["stem"]] = record
    return records


def ordered_true_pairs(record: dict[str, Any], region: str) -> set[tuple[int, int]]:
    """Ground-truth contacts in exp89's resolved candidate universe."""
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


def predicted_maps(
    frame: pd.DataFrame, record: dict[str, Any], region: str
) -> list[list[tuple[int, int]]]:
    """Return emission-ordered maps restricted to resolved positions."""
    resolved = {int(value) for value in record["resolved"]}
    minimum = 24 if region == "long" else 6
    maps = []
    for row in frame.itertuples():
        contacts = []
        for raw_left, raw_right in row.contacts:
            pair = (min(int(raw_left), int(raw_right)), max(int(raw_left), int(raw_right)))
            if pair[0] in resolved and pair[1] in resolved and pair[1] - pair[0] >= minimum:
                contacts.append(pair)
        maps.append(contacts)
    return maps


def rollout_r_precision(
    contacts: list[tuple[int, int]], true: set[tuple[int, int]]
) -> float:
    """Score one emission-ordered map at exp89's fixed R cutoff."""
    if not true:
        return float("nan")
    ranked = list(dict.fromkeys(contacts))[: len(true)]
    return len(set(ranked) & true) / len(true)


def vote_matrix(maps: list[list[tuple[int, int]]], length: int) -> np.ndarray:
    """Accumulate one vote per distinct contact per rollout."""
    votes = np.zeros((length, length), dtype=np.float32)
    for contacts in maps:
        for left, right in set(contacts):
            votes[left, right] += 1
            votes[right, left] += 1
    return votes


def consensus_r_precision(
    maps: list[list[tuple[int, int]]], record: dict[str, Any], region: str
) -> float:
    """Score rollout votes with exp89's unchanged evaluator."""
    length = int(record["L"])
    emitted = set().union(*(set(contacts) for contacts in maps))
    true = ordered_true_pairs(record, region)
    if len(emitted) < len(true):
        return len(emitted & true) / len(true) if true else float("nan")
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


def mean_pairwise_jaccard(maps: list[list[tuple[int, int]]]) -> float:
    """Mean map-set Jaccard, defining two empty maps as identical."""
    sets = [set(contacts) for contacts in maps]
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else float("nan")


def read_mode(mode: str, stem: str, n: int = 100) -> pd.DataFrame:
    """Read an ordered complete rollout pool from the local mirror."""
    path = HERE / "_cache" / mode / "eval-val" / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).sort_values("rollout").iloc[:n].copy()
    if len(frame) != n:
        raise ValueError(f"{mode}/{stem}: expected {n} rollouts, found {len(frame)}")
    return frame


def alternating_pool(native: pd.DataFrame, mutant: pd.DataFrame) -> pd.DataFrame:
    """Interleave 50 native and 50 mutant maps at a fixed 100-map budget."""
    rows = []
    for index in range(50):
        rows.append(native.iloc[index])
        rows.append(mutant.iloc[index])
    return pd.DataFrame(rows).reset_index(drop=True)


def top_two_pool(
    native: pd.DataFrame, first: pd.DataFrame, second: pd.DataFrame
) -> pd.DataFrame:
    """Interleave 50 native maps with 25 maps from each selected rate."""
    rows = []
    for index in range(50):
        rows.append(native.iloc[index])
        source = first if index % 2 == 0 else second
        rows.append(source.iloc[index // 2])
    return pd.DataFrame(rows).reset_index(drop=True)


def fixed_pools(stem: str, modes: list[str]) -> dict[str, pd.DataFrame]:
    """Build fixed native, all-mutant, and 50:50 pools."""
    native = read_mode("full_iid_single", stem)
    pools = {BASELINE: native}
    for mode in modes:
        mutant = read_mode(mode, stem)
        pools[f"mutant100_{mode}"] = mutant
        if mode != "mask_p0000":
            pools[f"mix50_{mode}"] = alternating_pool(native, mutant)
    return pools


def score_pool(
    stem: str,
    mode: str,
    frame: pd.DataFrame,
    record: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Score best-of-N and diversity curves for one target/pool."""
    rows = []
    for region in ("all", "long"):
        maps = predicted_maps(frame, record, region)
        true = ordered_true_pairs(record, region)
        invalid_statements = frame.malformed_contacts.to_numpy(dtype=int)
        if "out_of_range_contacts" in frame:
            invalid_statements = (
                invalid_statements
                + frame.out_of_range_contacts.fillna(0).to_numpy(dtype=int)
            )
        valid = frame.finished.to_numpy(dtype=bool) & (invalid_statements == 0)
        for budget in BUDGETS:
            prefix = maps[:budget]
            scores = [rollout_r_precision(contacts, true) for contacts in prefix]
            gated = [score if is_valid else 0.0 for score, is_valid in zip(scores, valid[:budget], strict=True)]
            union = set().union(*(set(contacts) for contacts in prefix))
            rows.append(
                {
                    "split": split,
                    "stem": stem,
                    "L": int(record["L"]),
                    "mode": mode,
                    "range": region,
                    "N": budget,
                    "validity_gated_oracle_r_precision": float(np.max(gated)),
                    "oracle_r_precision": float(np.max(scores)),
                    "mean_rollout_r_precision": float(np.mean(scores)),
                    "consensus_r_precision": consensus_r_precision(prefix, record, region),
                    "true_union_recall": len(union & true) / len(true) if true else float("nan"),
                    "mean_pairwise_jaccard": mean_pairwise_jaccard(prefix),
                    "unique_maps": len({frozenset(contacts) for contacts in prefix}),
                    "mean_contacts": float(np.mean([len(contacts) for contacts in prefix])),
                    "invalid_rollouts": int((~valid[:budget]).sum()),
                    "finished": int(frame.iloc[:budget].finished.sum()),
                    "malformed": int(frame.iloc[:budget].malformed_contacts.sum()),
                    "out_of_range": int(
                        frame.iloc[:budget].out_of_range_contacts.sum()
                        if "out_of_range_contacts" in frame
                        else 0
                    ),
                }
            )
    return rows


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate primary and secondary metrics at best@100."""
    final = frame[frame.N == 100]
    return (
        final.groupby(["mode", "range"], as_index=False)
        .agg(
            n=("stem", "nunique"),
            validity_gated_oracle_r_precision=(
                "validity_gated_oracle_r_precision",
                "mean",
            ),
            consensus_r_precision=("consensus_r_precision", "mean"),
            mean_rollout_r_precision=("mean_rollout_r_precision", "mean"),
            true_union_recall=("true_union_recall", "mean"),
            mean_pairwise_jaccard=("mean_pairwise_jaccard", "mean"),
            unique_maps=("unique_maps", "mean"),
            invalid_rollouts=("invalid_rollouts", "sum"),
            finished=("finished", "sum"),
            malformed=("malformed", "sum"),
            out_of_range=("out_of_range", "sum"),
        )
        .sort_values(["range", "validity_gated_oracle_r_precision"], ascending=[True, False])
    )


def paired_deltas(frame: pd.DataFrame, baseline_mode: str = BASELINE) -> pd.DataFrame:
    """Compute paired best@100 deltas against one named control."""
    final = frame[frame.N == 100]
    rows = []
    seed = 333_000
    for region in ("all", "long"):
        baseline = final[
            (final["mode"] == baseline_mode) & (final["range"] == region)
        ].set_index("stem")
        for mode in sorted(set(final["mode"]) - {baseline_mode}):
            arm = final[(final["mode"] == mode) & (final["range"] == region)].set_index("stem")
            if set(arm.index) != set(baseline.index):
                raise ValueError(f"unpaired target sets for {mode}/{region}")
            values = (
                arm.loc[baseline.index, "validity_gated_oracle_r_precision"]
                - baseline["validity_gated_oracle_r_precision"]
            ).to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, seed)
            rows.append(
                {
                    "mode": mode,
                    "baseline_mode": baseline_mode,
                    "range": region,
                    "N": 100,
                    "n": len(values),
                    "baseline_mean": float(
                        baseline.validity_gated_oracle_r_precision.mean()
                    ),
                    "arm_mean": float(
                        arm.validity_gated_oracle_r_precision.mean()
                    ),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
            seed += 1
    return pd.DataFrame(rows)


def select_top_two(summary: pd.DataFrame, modes: list[str]) -> list[str]:
    """Select the two strongest masked-only rates on development all-range."""
    candidates = summary[
        (summary["range"] == "all")
        & summary["mode"].isin([f"mutant100_{mode}" for mode in modes if mode != "mask_p0000"])
    ].sort_values(["validity_gated_oracle_r_precision", "mode"], ascending=[False, True])
    if len(candidates) < 2:
        raise ValueError("need at least two nonzero mutation rates for heterogeneous pool")
    return [value.removeprefix("mutant100_") for value in candidates["mode"].iloc[:2]]


def add_top_two_pool(
    scored: pd.DataFrame,
    split: str,
    selected_modes: list[str],
    records: dict[str, dict[str, Any]],
    stems: list[str],
) -> pd.DataFrame:
    """Score a frozen 50-native/25+25 heterogeneous mutation pool."""
    rows = []
    for stem in stems:
        pool = top_two_pool(
            read_mode("full_iid_single", stem),
            read_mode(selected_modes[0], stem),
            read_mode(selected_modes[1], stem),
        )
        rows.extend(score_pool(stem, "mix50_top2", pool, records[stem], split))
    return pd.concat([scored, pd.DataFrame(rows)], ignore_index=True)


def freeze_choice(summary: pd.DataFrame, deltas: pd.DataFrame, top_two: list[str]) -> dict[str, Any]:
    """Apply the preregistered validity and development gates."""
    zero = deltas[deltas["mode"] == "mutant100_mask_p0000"].set_index("range")
    validity_passed = bool(
        len(zero) == 2
        and abs(float(zero.loc["all", "mean_delta"])) <= 0.005
        and abs(float(zero.loc["long", "mean_delta"])) <= 0.005
    )
    candidates = deltas[
        ~deltas["mode"].isin(["mutant100_mask_p0000"])
    ].pivot(index="mode", columns="range", values="mean_delta")
    eligible = candidates[(candidates["all"] >= 0.005) & (candidates["long"] >= -0.005)]
    policy_gate_passed = not eligible.empty
    if not validity_passed or not policy_gate_passed:
        reasons = []
        if not validity_passed:
            reasons.append("zero-mask control differed from the published iid pool by more than 0.005")
        if not policy_gate_passed:
            reasons.append("no mutation policy cleared both development thresholds")
        return {
            "status": "no_advance",
            "validity_passed": validity_passed,
            "policy_gate_passed": policy_gate_passed,
            "selected_mode": None,
            "selected_raw_modes": [],
            "heterogeneous_top_two": top_two,
            "reason": "; ".join(reasons),
        }
    selected = eligible.sort_values(["all", "long"], ascending=False).index[0]
    if selected == "mix50_top2":
        raw_modes = top_two
    else:
        raw_modes = [selected.split("_", maxsplit=1)[1]]
    return {
        "status": "frozen_before_heldout",
        "validity_passed": True,
        "selected_mode": selected,
        "selected_raw_modes": raw_modes,
        "heterogeneous_top_two": top_two,
        "development_delta_all": float(eligible.loc[selected, "all"]),
        "development_delta_long": float(eligible.loc[selected, "long"]),
        "selection_rule": "largest all-range delta among policies with all >= 0.005 and long >= -0.005",
        "heldout_accessed_at_freeze": False,
    }


def main() -> None:
    """Score development or frozen confirmation pools."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--modes", help="comma-separated raw mask modes; required on dev")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--choice", type=Path, default=HERE / "data" / "frozen_choice.json")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    targets = pd.read_csv(TARGETS)
    targets = targets[(targets.cohort == "eval-val") & (targets.split == args.split)]
    stems = list(targets.sort_values(["L", "stem"]).stem)
    records = truth_records()

    if args.split == "dev":
        if not args.modes:
            raise ValueError("--modes is required for development analysis")
        modes = [value.strip() for value in args.modes.split(",") if value.strip()]
        rows = []
        for stem in stems:
            for mode, pool in fixed_pools(stem, modes).items():
                rows.extend(score_pool(stem, mode, pool, records[stem], args.split))
        scored = pd.DataFrame(rows)
        preliminary = aggregate(scored)
        top_two = select_top_two(preliminary, modes)
        scored = add_top_two_pool(scored, args.split, top_two, records, stems)
    else:
        choice = json.loads(args.choice.read_text())
        if choice["status"] != "frozen_before_heldout":
            raise ValueError("frozen choice does not authorize confirmation scoring")
        modes = choice["selected_raw_modes"]
        selected = choice["selected_mode"]
        rows = []
        for stem in stems:
            native = read_mode("full_iid_single", stem)
            rows.extend(score_pool(stem, BASELINE, native, records[stem], args.split))
            if selected == "mix50_top2":
                pool = top_two_pool(native, read_mode(modes[0], stem), read_mode(modes[1], stem))
            else:
                mutant = read_mode(modes[0], stem)
                pool = mutant if selected.startswith("mutant100_") else alternating_pool(native, mutant)
            rows.extend(score_pool(stem, selected, pool, records[stem], args.split))
        scored = pd.DataFrame(rows)
        top_two = choice["heterogeneous_top_two"]

    summary = aggregate(scored)
    deltas = paired_deltas(scored)
    destination = HERE / "data"
    destination.mkdir(exist_ok=True)
    scored.to_csv(destination / f"{args.output_prefix}_natural.csv", index=False)
    summary.to_csv(destination / f"{args.output_prefix}_summary.csv", index=False)
    deltas.to_csv(destination / f"{args.output_prefix}_paired_deltas.csv", index=False)
    if args.split == "dev":
        zero_deltas = paired_deltas(scored, "mutant100_mask_p0000")
        zero_deltas.to_csv(
            destination / f"{args.output_prefix}_paired_deltas_vs_zero.csv",
            index=False,
        )
    print(summary.to_string(index=False))
    print("\npaired oracle deltas vs iid100\n", deltas.to_string(index=False))
    if args.freeze:
        if args.split != "dev":
            raise ValueError("--freeze is only valid for development")
        choice = freeze_choice(summary, deltas, top_two)
        args.choice.write_text(json.dumps(choice, indent=2) + "\n")
        print(f"\nfrozen choice\n{json.dumps(choice, indent=2)}")


if __name__ == "__main__":
    main()
