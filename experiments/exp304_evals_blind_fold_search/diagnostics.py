#!/usr/bin/env python
"""Classify proposal, co-occurrence, and blind-ranking bottlenecks."""

from pathlib import Path

import pandas as pd

from search_policy import canonical

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
ARMS = ("iid", "temp", "random", "branch5", "branch10", "branch20")


def main() -> None:
    per = pd.read_csv(DATA / "per_protein.csv")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    raw = pd.concat(
        [pd.read_parquet(path) for path in sorted((HERE / "_cache" / "raw").glob("shard-*.parquet"))],
        ignore_index=True,
    )
    rows = []
    for record in per.itertuples():
        target = truth.loc[record.pair_id]
        fold1 = canonical(target.contacts_fold1)
        fold2 = canonical(target.contacts_fold2)
        unique = (fold2 - fold1) if record.dominant == "fold1" else (fold1 - fold2)
        lo, hi = int(target.fs_lo), int(target.fs_hi)
        minority = {pair for pair in unique if lo <= pair[0] < hi or lo <= pair[1] < hi}
        pool = raw[(raw.pair_id == record.pair_id) & raw.arm.isin(["root", record.method])]
        proposed = set().union(*(
            canonical(row.contacts) | canonical(row.given)
            for row in pool.itertuples() if row.finished
        ))
        union_recall = len(minority & proposed) / len(minority) if minority else float("nan")
        if record.dual_contact_hit:
            bottleneck = "blind_dual_hit"
        elif record.oracle_dual_hit:
            bottleneck = "blind_ranking"
        elif union_recall < 0.25:
            bottleneck = "proposal_coverage"
        else:
            bottleneck = "cooccurrence_or_specificity"
        rows.append({"pair_id": record.pair_id, "method": record.method,
                     "primary": record.primary, "split": record.split,
                     "minority_union_recall": union_recall,
                     "bottleneck": bottleneck})
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA / "diagnostics.csv", index=False)
    print(frame[frame.primary & (frame.split == "test")].groupby(
        ["method", "bottleneck"]
    ).size().to_string())


if __name__ == "__main__":
    main()
