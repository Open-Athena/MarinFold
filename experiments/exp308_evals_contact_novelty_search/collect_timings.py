#!/usr/bin/env python
"""Collect full-run H100 timings from the saved novelty-search outputs."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Write one row per evaluated target and decoder mode."""
    paths = sorted((HERE / "_cache").glob("b*/*/*.timing.parquet"))
    if not paths:
        raise FileNotFoundError("no saved timing parquets")
    timings = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    timings = timings[timings.n_rollouts == 100].copy()
    if timings.duplicated(["stem", "mode"]).any():
        raise ValueError("duplicate target and mode timing rows")
    timings.sort_values(["mode", "stem"]).to_csv(HERE / "data" / "timings.csv", index=False)
    print(f"saved {len(timings)} predictor timing rows")


if __name__ == "__main__":
    main()
