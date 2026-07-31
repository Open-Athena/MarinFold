"""Summarize per-document eval losses with document-level Poisson bootstrap."""

import argparse
import json
from pathlib import Path

import pandas as pd
from marinfold_evals import poisson_bootstrap_weighted_mean


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet per-document loss table."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input suffix {path.suffix!r}; expected .csv or .parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV/parquet with one row per validation document.")
    parser.add_argument("--output", required=True, type=Path, help="Output bootstrap summary JSON.")
    parser.add_argument("--loss-sum-column", default="loss_sum")
    parser.add_argument("--token-count-column", default="token_count")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    table = read_table(args.input)
    summary = poisson_bootstrap_weighted_mean(
        table[args.loss_sum_column].to_numpy(),
        table[args.token_count_column].to_numpy(),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    record = {
        "eval_loss": summary.estimate,
        "stderr": summary.stderr,
        "n_units": summary.n_units,
        "n_bootstrap": summary.n_bootstrap,
        "seed": summary.seed,
        "input": str(args.input),
        "loss_sum_column": args.loss_sum_column,
        "token_count_column": args.token_count_column,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
