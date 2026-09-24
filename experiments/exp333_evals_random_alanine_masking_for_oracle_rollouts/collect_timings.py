"""Collect per-target predictor timings from locally mirrored parquets."""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Write the repository-standard compact timing CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "_cache")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "timings.csv")
    args = parser.parse_args()
    paths = sorted(args.root.glob("mask_*/*/*.timing.parquet"))
    if not paths:
        raise FileNotFoundError(f"no timing parquets under {args.root}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame.sort_values(["mode", "cohort", "n_residues", "stem"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame.groupby("mode")
        .agg(
            proteins=("stem", "nunique"),
            inference_seconds=("elapsed_seconds", "sum"),
            generated_tokens=("generated_tokens", "sum"),
            finished=("n_finished", "sum"),
            malformed=("n_malformed", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
