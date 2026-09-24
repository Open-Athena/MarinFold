"""Collect per-target timing parquets into the committed timing table."""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Scan one raw artifact root and write a stable CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "_cache")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "timings.csv")
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/*/*.timing.parquet"))
    if not paths:
        raise ValueError(f"no timing parquets under {args.root}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame.sort_values(["cohort", "mode", "n_residues", "stem"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame.groupby(["cohort", "mode"])
        .agg(
            proteins=("stem", "nunique"),
            seconds=("elapsed_seconds", "sum"),
            tokens=("generated_tokens", "sum"),
            finished=("n_finished", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
