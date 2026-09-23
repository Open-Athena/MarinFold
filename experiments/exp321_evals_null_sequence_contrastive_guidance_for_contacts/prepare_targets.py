"""Freeze natural eval-val and fold-switching development/test targets."""

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXP304_REVISION = "f846626561aeb8eaaba99a6eb89d1c1e945632cd"
EXP304_ROOT = "experiments/exp304_evals_blind_fold_search"


def git_bytes(path: str) -> bytes:
    """Read a frozen file from exp304 without checking its whole branch out."""
    return subprocess.check_output(["git", "show", f"{EXP304_REVISION}:{path}"])


def eval_val_targets() -> pd.DataFrame:
    """Return all 97 natural eval-val proteins with 16 length-stratified dev."""
    root = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data"
    eval_sets = pd.read_csv(root / "eval_sets.csv")
    stems = set(eval_sets.loc[eval_sets.eval_set == "eval-val", "stem"])
    frame = pd.read_parquet(root / "eval_targets_foldbench_monomers.parquet")
    frame = frame[frame.stem.isin(stems)].copy().sort_values(["L", "stem"])
    if len(frame) != 97 or frame.stem.nunique() != 97:
        raise ValueError(f"expected 97 unique eval-val proteins, got {len(frame)}")
    development_indices = set(np.rint(np.linspace(0, len(frame) - 1, 16)).astype(int))
    if len(development_indices) != 16:
        raise ValueError("length-stratified development selection is not unique")
    frame["cohort"] = "eval-val"
    frame["dataset"] = "foldbench_monomer"
    frame["target_id"] = frame.stem
    frame["sequence"] = frame.input_seq
    frame["split"] = ["dev" if index in development_indices else "test"
                      for index in range(len(frame))]
    frame["primary"] = True
    return frame


def foldswitch_targets() -> pd.DataFrame:
    """Return exp304's frozen 67 non-capped fold-switch target sequences."""
    targets = pd.read_parquet(io.BytesIO(git_bytes(f"{EXP304_ROOT}/data/search_targets.parquet")))
    cohort = pd.read_csv(io.BytesIO(git_bytes(f"{EXP304_ROOT}/data/cohort.csv")))
    frame = targets.merge(
        cohort[["pair_id", "split", "primary"]], on="pair_id", validate="one_to_one"
    )
    if len(frame) != 67 or frame.pair_id.nunique() != 67:
        raise ValueError(f"expected 67 fold-switch targets, got {len(frame)}")
    frame["cohort"] = "foldswitch"
    frame["dataset"] = "foldswitch"
    frame["stem"] = frame.pair_id
    frame["target_id"] = frame.pair_id
    return frame


def main() -> None:
    """Write the immutable target table consumed by every worker arm."""
    columns = [
        "cohort", "dataset", "stem", "target_id", "sequence", "L", "split", "primary"
    ]
    targets = pd.concat(
        [eval_val_targets()[columns], foldswitch_targets()[columns]], ignore_index=True
    )
    if targets.target_id.duplicated().any():
        raise ValueError("duplicate target IDs")
    if not (targets.sequence.str.len() == targets.L).all():
        raise ValueError("sequence length mismatch")
    destination = HERE / "data"
    destination.mkdir(exist_ok=True)
    targets.sort_values(["cohort", "L", "target_id"]).to_csv(
        destination / "targets.csv", index=False
    )
    print(targets.groupby(["cohort", "split", "primary"]).size().to_string())


if __name__ == "__main__":
    main()
