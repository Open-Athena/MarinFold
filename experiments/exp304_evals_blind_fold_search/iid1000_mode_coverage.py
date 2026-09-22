#!/usr/bin/env python
"""Classify cumulative iid contact-fold coverage on the 29 primary test pairs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import MIN_CONTACTS_FS, MIN_ENRICHMENT, MIN_RECALL
from evaluate import score_candidate

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
OLD = HERE / "_cache" / "iid500"
TAIL = HERE / "_cache" / "iid1000_tail_primary"
N_DRAWS = 1000


def raw_by_pair(root: Path, expected: set[str]) -> dict[str, Path]:
    """Index raw one-protein parquet files and require all primary test pairs."""
    indexed = {}
    for path in sorted(root.glob("shard-*.parquet")):
        pair_ids = pd.read_parquet(path, columns=["pair_id"]).pair_id.unique()
        if len(pair_ids) != 1:
            raise ValueError(f"{path}: expected one protein")
        pair_id = str(pair_ids[0])
        if pair_id in expected:
            if pair_id in indexed:
                raise ValueError(f"duplicate {pair_id} in {root}")
            indexed[pair_id] = path
    if set(indexed) != expected:
        raise ValueError(f"{root}: missing {sorted(expected - set(indexed))}")
    return indexed


def absence_curve(successes: int, population: int) -> np.ndarray:
    """Probability of zero successes in a random size-n subset, for every n."""
    probabilities = np.ones(population + 1)
    for n in range(1, population + 1):
        probabilities[n] = (probabilities[n - 1]
                            * max(0, population - successes - n + 1)
                            / (population - n + 1))
    return probabilities


def first_draw(hit: np.ndarray) -> int | None:
    """Return the one-based first qualifying draw, if any."""
    indices = np.flatnonzero(hit)
    return int(indices[0]) + 1 if len(indices) else None


def main() -> None:
    """Recompute the original 500 checks and extend the same stream to 1000."""
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    expected = set(cohort[(cohort.split == "test") & cohort.primary].index)
    frozen_ids = set((DATA / "iid1000_primary_test_ids.txt").read_text().splitlines())
    if len(expected) != 29 or expected != frozen_ids:
        raise ValueError("primary test target list changed")
    old = raw_by_pair(OLD, expected)
    tail = raw_by_pair(TAIL, expected)
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    with (SOURCE / "foldswitch_universe.jsonl").open() as source:
        mismatch = {row["pair_id"]: int(row["n_seq_mismatch"])
                    for row in (json.loads(line) for line in source)}
    counts = {name: np.zeros(N_DRAWS + 1, dtype=int)
              for name in ("neither", "one", "both", "both_strict", "fold1_seen", "fold2_seen")}
    expected_counts = {name: np.zeros(N_DRAWS + 1)
                       for name in ("neither", "one", "both")}
    per_protein = []
    previous = pd.read_csv(DATA / "iid_mode_coverage_per_protein.csv")
    previous = previous[previous.budget == 500].set_index("pair_id")
    for pair_id in sorted(expected):
        first = pd.read_parquet(old[pair_id]).sort_values("rollout")
        second = pd.read_parquet(tail[pair_id]).sort_values("rollout")
        if (first.rollout.tolist() != list(range(500))
                or second.rollout.tolist() != list(range(500, N_DRAWS))
                or set(first.arm) != {"root"} or set(second.arm) != {"root"}):
            raise ValueError(f"{pair_id}: expected disjoint root draws 0..999")
        frame = pd.concat([first, second], ignore_index=True)
        if frame.candidate_id.tolist() != [f"root:{index}" for index in range(N_DRAWS)]:
            raise ValueError(f"{pair_id}: candidate IDs are not a continuous iid stream")
        target = truth.loc[pair_id].to_dict()
        target["pair_id"] = pair_id
        scored = pd.DataFrame([
            score_candidate(record, target, True, "test", "iid", 0)
            for record in frame.to_dict("records")
        ])
        valid = scored.finished & (scored.n_pred_fs >= MIN_CONTACTS_FS)
        hit1 = (valid & (scored.recall_a_fs >= MIN_RECALL)
                & (scored.phi_fs >= MIN_ENRICHMENT)).to_numpy(dtype=bool)
        hit2 = (valid & (scored.recall_b_fs >= MIN_RECALL)
                & (-scored.phi_fs >= MIN_ENRICHMENT)).to_numpy(dtype=bool)
        if np.any(hit1 & hit2):
            raise ValueError(f"{pair_id}: one rollout matched opposing exclusive modes")
        strict1 = hit1 & (scored.recall_a_fs.to_numpy() >= 0.50)
        strict2 = hit2 & (scored.recall_b_fs.to_numpy() >= 0.50)
        seen1 = np.r_[False, np.maximum.accumulate(hit1)]
        seen2 = np.r_[False, np.maximum.accumulate(hit2)]
        seen_strict1 = np.r_[False, np.maximum.accumulate(strict1)]
        seen_strict2 = np.r_[False, np.maximum.accumulate(strict2)]
        counts["neither"] += (~seen1 & ~seen2).astype(int)
        counts["one"] += (seen1 ^ seen2).astype(int)
        counts["both"] += (seen1 & seen2).astype(int)
        counts["both_strict"] += (seen_strict1 & seen_strict2).astype(int)
        counts["fold1_seen"] += seen1.astype(int)
        counts["fold2_seen"] += seen2.astype(int)
        n1, n2 = int(hit1.sum()), int(hit2.sum())
        no1 = absence_curve(n1, N_DRAWS)
        no2 = absence_curve(n2, N_DRAWS)
        no_either = absence_curve(n1 + n2, N_DRAWS)
        expected_counts["neither"] += no_either
        expected_counts["both"] += 1 - no1 - no2 + no_either
        expected_counts["one"] += no1 + no2 - 2 * no_either
        first1, first2 = first_draw(hit1), first_draw(hit2)
        if (bool(previous.loc[pair_id, "fold1_hit"]) != bool(seen1[500])
                or bool(previous.loc[pair_id, "fold2_hit"]) != bool(seen2[500])):
            raise ValueError(f"{pair_id}: 500-draw prefix disagrees with published exp304 score")
        per_protein.append({
            "pair_id": pair_id, "L": int(cohort.loc[pair_id, "L"]),
            "strict_exact": mismatch[pair_id] == 0,
            "n_finished": int(frame.finished.sum()),
            "n_fold1_hits": n1, "n_fold2_hits": n2,
            "first_fold1_draw": first1, "first_fold2_draw": first2,
            "first_dual_draw": max(first1, first2) if first1 and first2 else None,
            "state_500": int(seen1[500]) + int(seen2[500]),
            "state_1000": int(seen1[-1]) + int(seen2[-1]),
            "both_strict_1000": bool(seen_strict1[-1] and seen_strict2[-1]),
        })
    curve = pd.DataFrame({"budget": np.arange(N_DRAWS + 1), **counts,
                          **{f"expected_{name}": values
                             for name, values in expected_counts.items()}})
    if not (curve.neither + curve.one + curve.both).eq(29).all():
        raise ValueError("coverage categories do not partition 29 proteins")
    for budget in (10, 25, 50, 100, 200, 500):
        prior = pd.read_csv(DATA / "iid_mode_coverage_summary.csv")
        reference = prior[(prior.cohort == "primary_test") & (prior.budget == budget)].iloc[0]
        row = curve.iloc[budget]
        if (row.fold1_seen != reference.fold1 or row.fold2_seen != reference.fold2
                or row.both != reference.both or row.neither != reference.neither):
            raise ValueError(f"{budget}-draw summary disagrees with published exp304 score")
    curve.to_csv(DATA / "iid1000_primary_test_curve.csv", index=False)
    pd.DataFrame(per_protein).to_csv(DATA / "iid1000_primary_test_per_protein.csv", index=False)
    timing_files = sorted(TAIL.glob("timing-*.parquet"))
    if len(timing_files) != 29:
        raise ValueError(f"expected 29 tail timing files, found {len(timing_files)}")
    timings = pd.concat((pd.read_parquet(path) for path in timing_files), ignore_index=True)
    if (len(timings) != 29 or set(timings.pair_id) != expected
            or not timings.n_rollouts.eq(500).all()
            or not timings.rollout_start.eq(500).all()):
        raise ValueError("tail timing metadata does not match the planned run")
    timings.sort_values("pair_id").to_csv(DATA / "timings_iid1000_tail.csv", index=False)
    print(curve[curve.budget.isin([0, 100, 200, 500, 750, 1000])]
          [["budget", "neither", "one", "both", "both_strict"]].to_string(index=False))


if __name__ == "__main__":
    main()
