"""CLI for document-level Poisson bootstrap stderr of eval loss."""

import argparse
import json
from pathlib import Path
from typing import Any

from marinfold_evals.bootstrap import poisson_bootstrap_weighted_mean


def _read_table(path: str):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension for {path!r}; expected .parquet or .csv")


def _write_json(path: str, record: dict[str, Any]) -> None:
    if path == "-":
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV or parquet with one row per eval unit/document.")
    parser.add_argument("--loss-sum-column", default="loss_sum")
    parser.add_argument("--token-count-column", default="token_count")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--output", default="-", help="Output JSON path, or '-' for stdout.")
    args = parser.parse_args(argv)

    table = _read_table(args.input)
    summary = poisson_bootstrap_weighted_mean(
        table[args.loss_sum_column].to_numpy(),
        table[args.token_count_column].to_numpy(),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    record = {
        "eval_loss": summary.estimate,
        "stderr": summary.stderr,
        "n_units": summary.n_units,
        "n_bootstrap": summary.n_bootstrap,
        "seed": summary.seed,
        "input": args.input,
        "loss_sum_column": args.loss_sum_column,
        "token_count_column": args.token_count_column,
    }
    _write_json(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
