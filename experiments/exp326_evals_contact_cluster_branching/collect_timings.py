"""Collect per-target predictor timings into the committed experiment table."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Concatenate all completed local timing records."""
    paths = sorted((HERE / "_cache").glob("*/*/*.timing.parquet"))
    if not paths:
        raise FileNotFoundError("no timing parquets under _cache")
    timings = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    keys = ["mode", "cohort", "stem"]
    if timings.duplicated(keys).any():
        duplicates = timings.loc[timings.duplicated(keys, keep=False), keys]
        raise ValueError(f"duplicate timing rows:\n{duplicates.to_string(index=False)}")
    timings = timings.sort_values(["cohort", "mode", "n_residues", "stem"])
    timings.to_csv(HERE / "data" / "timings.csv", index=False)
    print(
        timings.groupby(["cohort", "mode"])
        .agg(
            proteins=("stem", "nunique"),
            seconds=("elapsed_seconds", "sum"),
            tokens=("generated_tokens", "sum"),
            finished=("n_finished", "sum"),
            malformed=("n_malformed", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
