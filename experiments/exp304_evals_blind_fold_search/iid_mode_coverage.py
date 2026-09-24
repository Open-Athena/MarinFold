#!/usr/bin/env python
"""Measure reference-aware Fold1/Fold2 contact-mode coverage in iid rollouts."""

import json
from pathlib import Path

import pandas as pd

from analyze import MIN_CONTACTS_FS, MIN_ENRICHMENT, MIN_RECALL
from budget_curve import raw_files
from evaluate import score_candidate

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
BUDGETS = (10, 25, 50, 100, 200, 500)


def first_hit(mask: pd.Series, rollouts: pd.Series) -> int | None:
    """Return the one-based draw count at the first passing rollout."""
    matches = rollouts[mask]
    return int(matches.iloc[0]) + 1 if not matches.empty else None


def main() -> None:
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    mismatch: dict[str, int] = {}
    with (SOURCE / "foldswitch_universe.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            mismatch[row["pair_id"]] = int(row["n_seq_mismatch"])

    rows = []
    for file in raw_files():
        frame = pd.read_parquet(file).sort_values("rollout").reset_index(drop=True)
        pid = str(frame.pair_id.iloc[0])
        if frame.rollout.tolist() != list(range(500)):
            raise ValueError(f"{pid}: missing or repeated rollout index")
        target = truth.loc[pid].to_dict()
        target["pair_id"] = pid
        scored = pd.DataFrame([
            score_candidate(record, target, bool(cohort.loc[pid, "primary"]),
                            str(cohort.loc[pid, "split"]), "iid", 0)
            for record in frame.to_dict("records")
        ])
        valid = scored.finished & (scored.n_pred_fs >= MIN_CONTACTS_FS)
        hit1 = valid & (scored.recall_a_fs >= MIN_RECALL) & (scored.phi_fs >= MIN_ENRICHMENT)
        hit2 = valid & (scored.recall_b_fs >= MIN_RECALL) & (-scored.phi_fs >= MIN_ENRICHMENT)
        strict1 = hit1 & (scored.recall_a_fs >= 0.50)
        strict2 = hit2 & (scored.recall_b_fs >= 0.50)
        if (hit1 & hit2).any():
            raise ValueError(f"{pid}: mutually exclusive fold hits overlap")
        for budget in BUDGETS:
            h1, h2 = hit1.iloc[:budget], hit2.iloc[:budget]
            first1 = first_hit(h1, frame.rollout.iloc[:budget])
            first2 = first_hit(h2, frame.rollout.iloc[:budget])
            rows.append({
                "pair_id": pid, "budget": budget,
                "primary": bool(cohort.loc[pid, "primary"]),
                "split": str(cohort.loc[pid, "split"]),
                "seq_class": str(cohort.loc[pid, "seq_class"]),
                "strict_exact": mismatch[pid] == 0,
                "n_valid": int(valid.iloc[:budget].sum()),
                "n_fold1_hits": int(h1.sum()), "n_fold2_hits": int(h2.sum()),
                "fold1_hit": bool(h1.any()), "fold2_hit": bool(h2.any()),
                "dual_hit": bool(h1.any() and h2.any()),
                "n_fold1_strict": int(strict1.iloc[:budget].sum()),
                "n_fold2_strict": int(strict2.iloc[:budget].sum()),
                "dual_strict": bool(strict1.iloc[:budget].any()
                                    and strict2.iloc[:budget].any()),
                "first_fold1_draw": first1, "first_fold2_draw": first2,
                "first_dual_draw": max(first1, first2) if first1 and first2 else None,
                "best_fold1_recall_fs": float(scored.recall_a_fs.iloc[:budget][valid.iloc[:budget]].max()),
                "best_fold2_recall_fs": float(scored.recall_b_fs.iloc[:budget][valid.iloc[:budget]].max()),
            })

    per = pd.DataFrame(rows).sort_values(["pair_id", "budget"])
    previous = pd.read_csv(DATA / "budget_per_protein.csv")
    merged = per.merge(previous[["pair_id", "budget", "oracle_dual_hit"]],
                       on=["pair_id", "budget"])
    if not (merged.dual_hit == merged.oracle_dual_hit).all():
        raise ValueError("iid mode coverage disagrees with prior oracle dual-hit scores")
    per.to_csv(DATA / "iid_mode_coverage_per_protein.csv", index=False)

    cohorts = {
        "primary_test": per.primary & (per.split == "test"),
        "primary_test_exact": per.primary & (per.split == "test") & per.strict_exact,
        "primary_all": per.primary,
        "near_identical_all": per.seq_class == "identical",
        "all_non_capped": pd.Series(True, index=per.index),
    }
    summary = []
    for name, selected in cohorts.items():
        for budget, group in per[selected].groupby("budget"):
            fold1 = group.fold1_hit
            fold2 = group.fold2_hit
            summary.append({
                "cohort": name, "budget": budget, "n_proteins": len(group),
                "fold1": int(fold1.sum()), "fold2": int(fold2.sum()),
                "both": int((fold1 & fold2).sum()),
                "both_strict": int(group.dual_strict.sum()),
                "fold1_only": int((fold1 & ~fold2).sum()),
                "fold2_only": int((~fold1 & fold2).sum()),
                "neither": int((~fold1 & ~fold2).sum()),
            })
    report = pd.DataFrame(summary)
    report.to_csv(DATA / "iid_mode_coverage_summary.csv", index=False)
    print(report[report.budget.isin([100, 200, 500])].to_string(index=False))


if __name__ == "__main__":
    main()
