#!/usr/bin/env python
"""Freeze sequence-only eval-val and fold-switching targets for beam decoding."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def main() -> None:
    val_root = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data"
    eval_sets = pd.read_csv(val_root / "eval_sets.csv")
    val_stems = set(eval_sets.loc[eval_sets.eval_set == "eval-val", "stem"])
    foldbench = pd.read_parquet(val_root / "eval_targets_foldbench_monomers.parquet")
    val = foldbench[foldbench.stem.isin(val_stems)].copy()
    if len(val) != 97 or val.stem.nunique() != 97:
        raise ValueError(f"expected 97 unique eval-val targets, got {len(val)}")
    val = val.rename(columns={"input_seq": "sequence"})
    val["cohort"] = "eval-val"
    val["split"] = "val"
    val["primary"] = False
    val["target_id"] = val.stem

    switch_root = EXPERIMENTS / "exp304_evals_blind_fold_search" / "data"
    switch = pd.read_parquet(switch_root / "search_targets.parquet")
    cohort = pd.read_csv(switch_root / "cohort.csv")
    switch = switch.merge(cohort[["pair_id", "split", "primary"]], on="pair_id",
                          validate="one_to_one")
    if len(switch) != 67:
        raise ValueError(f"expected 67 non-capped fold-switch targets, got {len(switch)}")
    switch["cohort"] = "foldswitch"
    switch["dataset"] = "foldswitch"
    switch["stem"] = switch.pair_id
    switch["target_id"] = switch.pair_id

    columns = ["cohort", "dataset", "stem", "target_id", "sequence", "L", "split", "primary"]
    targets = pd.concat([val[columns], switch[columns]], ignore_index=True)
    if targets.target_id.duplicated().any():
        raise ValueError("duplicate target IDs")
    if not (targets.sequence.str.len() == targets.L).all():
        raise ValueError("sequence length mismatch")
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    targets.sort_values(["cohort", "L", "target_id"]).to_csv(out / "targets.csv", index=False)
    print(targets.groupby("cohort").size().to_string())


if __name__ == "__main__":
    main()
