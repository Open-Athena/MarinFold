#!/usr/bin/env python
"""Summarize sealed fold discovery, oracle headroom, and compute cost."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
ARMS = ("iid", "temp", "random", "branch5", "branch10", "branch20")
PRIMARY_METHOD = "branch10"
MIN_CONTACTS_FS = 10
MIN_RECALL = 0.25
MIN_ENRICHMENT = 0.10


def paired_interval(values: np.ndarray, seed: int = 304) -> tuple[float, float]:
    """Protein-level percentile interval for a paired mean difference."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_group(group: pd.DataFrame, dominant: str) -> dict:
    """Best alternate evidence in a shortlist or oracle candidate pool."""
    valid = group[(group.n_pred_fs >= MIN_CONTACTS_FS) & group.finished].copy()
    if valid.empty:
        return {"minority_enrichment": 0.0, "minority_recall": 0.0,
                "dual_contact_hit": False, "n_valid": 0}
    if dominant == "fold1":
        valid["minority_enrichment"] = valid.recall_b_fs - valid.recall_a_fs
        valid["minority_recall"] = valid.recall_b_fs
    else:
        valid["minority_enrichment"] = valid.recall_a_fs - valid.recall_b_fs
        valid["minority_recall"] = valid.recall_a_fs
    a_hit = ((valid.recall_a_fs >= MIN_RECALL)
             & (valid.phi_fs >= MIN_ENRICHMENT)).any()
    b_hit = ((valid.recall_b_fs >= MIN_RECALL)
             & (-valid.phi_fs >= MIN_ENRICHMENT)).any()
    best = valid.sort_values(["minority_enrichment", "minority_recall"],
                             ascending=False).iloc[0]
    return {
        "minority_enrichment": float(best.minority_enrichment),
        "minority_recall": float(best.minority_recall),
        "dual_contact_hit": bool(a_hit and b_hit), "n_valid": len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=HERE / "_cache" / "raw")
    args = parser.parse_args()
    sealed = pd.read_csv(DATA / "scored_shortlists.csv")
    all_candidates = pd.read_parquet(HERE / "_cache" / "scored_all_candidates.parquet")
    baseline = pd.read_csv(SOURCE / "fold_preference.csv").set_index("pair_id")
    timings = pd.concat([pd.read_parquet(path) for path in args.raw.glob("timing-*.parquet")],
                        ignore_index=True)
    timings.to_csv(DATA / "timings.csv", index=False)
    timing_lookup = timings.set_index(["pair_id", "mode"])
    rows = []
    for (pid, method), group in sealed.groupby(["pair_id", "method"]):
        dominant = "fold1" if float(baseline.loc[pid, "phi"]) >= 0 else "fold2"
        blind = score_group(group, dominant)
        pool = all_candidates[(all_candidates.pair_id == pid)
                              & (all_candidates.source_arm.isin(["root", method]))]
        oracle = score_group(pool, dominant)
        root_time = float(timing_lookup.loc[(pid, "root"), "elapsed_seconds"])
        arm_time = float(timing_lookup.loc[(pid, method), "elapsed_seconds"])
        rows.append({
            "pair_id": pid, "method": method, "dominant": dominant,
            "split": group.split.iloc[0], "primary": bool(group.primary.iloc[0]),
            "L": int(group.L.iloc[0]),
            **blind,
            "oracle_minor_enrichment": oracle["minority_enrichment"],
            "oracle_minor_recall": oracle["minority_recall"],
            "oracle_dual_hit": oracle["dual_contact_hit"],
            "gpu_seconds": root_time + arm_time,
            "generated_tokens": int(timing_lookup.loc[(pid, "root"), "generated_tokens"]
                                    + timing_lookup.loc[(pid, method), "generated_tokens"]),
        })
    per = pd.DataFrame(rows).sort_values(["pair_id", "method"])
    per.to_csv(DATA / "per_protein.csv", index=False)
    primary_test = per[(per.primary) & (per.split == "test")]
    iid = primary_test[primary_test.method == "iid"].set_index("pair_id")
    summary = []
    for method in ARMS:
        sub = primary_test[primary_test.method == method].set_index("pair_id")
        common = iid.index.intersection(sub.index)
        diff = (sub.loc[common, "minority_enrichment"]
                - iid.loc[common, "minority_enrichment"]).dropna().to_numpy()
        lo, hi = paired_interval(diff)
        summary.append({
            "method": method, "primary_comparison": method == PRIMARY_METHOD,
            "n_test": len(sub),
            "mean_minor_enrichment": sub.minority_enrichment.mean(),
            "mean_minor_recall": sub.minority_recall.mean(),
            "dual_contact_hits": int(sub.dual_contact_hit.sum()),
            "oracle_dual_hits": int(sub.oracle_dual_hit.sum()),
            "paired_delta_vs_iid": float(diff.mean()) if len(diff) else float("nan"),
            "paired_delta_lo": lo, "paired_delta_hi": hi,
            "paired_n_better": int((diff > 0).sum()),
            "paired_n_worse": int((diff < 0).sum()),
            "paired_n_tied": int((diff == 0).sum()),
            "mean_gpu_seconds": sub.gpu_seconds.mean(),
            "mean_generated_tokens": sub.generated_tokens.mean(),
        })
    report = pd.DataFrame(summary)
    report.to_csv(DATA / "summary.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
