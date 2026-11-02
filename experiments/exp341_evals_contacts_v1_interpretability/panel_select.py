# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A3/A4 panel, stage 3: stratify removed structures and match kept controls.

Joins stage 1's candidates with stage 2's relative counts, bins every
candidate by its kept-relative count under the plan's rule (``n_relatives``:
identity >= 0.30 and coverage >= 0.50) into the plan's log-spaced bins
(0, 1-3, 4-15, 16-63, 64+), and per bin:

1. draws ``--per-bin`` removed structures uniformly at random (seeded), and
2. pairs each with one kept structure from the same bin by optimal one-to-one
   matching (``scipy.optimize.linear_sum_assignment``) on standardized
   log length, global pLDDT and log(1 + relative count).

Relative count falls steeply with length and rises with pLDDT across the bins
(stage 2: median removed length 540 in bin 0, 138 in 64+), so the kept
control has to be matched within each bin rather than drawn at random. The
removed draw is not conditioned on match quality; where the kept pool is
short of long structures (bin 0 has 342 kept candidates) the residual
imbalance is reported rather than hidden by trimming the removed side.

Writes ``data/panel.parquet`` (one row per panel structure: ``set``, ``bin``,
``pair`` linking each removed structure to its matched control, the
candidate columns and the relative counts) and ``data/panel_balance.csv``
(per bin and covariate: removed and kept means and the standardized mean
difference).

    uv run python panel_select.py --candidates candidates.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BIN_EDGES = [-1, 0, 3, 15, 63, np.inf]
BIN_LABELS = ["0", "1-3", "4-15", "16-63", "64+"]
SEED = 341
DATA = Path(__file__).resolve().parent / "data"


def covariates(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_length": np.log(frame["seq_len"]),
        "global_plddt": frame["global_plddt"],
        "log1p_relatives": np.log1p(frame["n_relatives"]),
    }, index=frame.index)


def balance(removed: pd.DataFrame, kept: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    a, b = covariates(removed), covariates(kept)
    for name in a.columns:
        pooled = np.sqrt((a[name].var() + b[name].var()) / 2)
        rows.append({
            "bin": label,
            "covariate": name,
            "removed_mean": a[name].mean(),
            "kept_mean": b[name].mean(),
            "smd": (a[name].mean() - b[name].mean()) / pooled if pooled else 0.0,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True, help="stage 1 candidates.parquet")
    parser.add_argument("--relatives", type=Path, default=DATA / "relative_counts.parquet")
    parser.add_argument("--per-bin", type=int, default=200)
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates)
    relatives = pd.read_parquet(args.relatives)
    pool = candidates.merge(relatives, on=["entry_id", "set"], how="inner", validate="one_to_one")
    if len(pool) != len(candidates):
        raise ValueError(f"{len(candidates) - len(pool)} candidates have no relative count")
    pool["bin"] = pd.cut(pool["n_relatives"], BIN_EDGES, labels=BIN_LABELS).astype(str)

    # One scale for every bin, so a unit of distance means the same everywhere.
    features = covariates(pool)
    scaled = (features - features.mean()) / features.std()

    rng = np.random.default_rng(SEED)
    panel, balance_rows = [], []
    for label in BIN_LABELS:
        removed = pool[(pool["set"] == "removed") & (pool["bin"] == label)]
        kept = pool[(pool["set"] == "kept") & (pool["bin"] == label)]
        if len(removed) < args.per_bin or len(kept) < args.per_bin:
            raise ValueError(f"bin {label}: {len(removed)} removed, {len(kept)} kept, need {args.per_bin} of each")
        drawn = removed.iloc[np.sort(rng.choice(len(removed), size=args.per_bin, replace=False))]
        a, b = scaled.loc[drawn.index].to_numpy(), scaled.loc[kept.index].to_numpy()
        cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        rows, cols = linear_sum_assignment(cost)
        matched = kept.iloc[cols]
        pair = [f"{label}:{k:03d}" for k in range(args.per_bin)]
        panel += [drawn.iloc[rows].assign(pair=pair), matched.assign(pair=pair)]
        balance_rows += balance(drawn, matched, label)

    frame = pd.concat(panel, ignore_index=True)
    DATA.mkdir(exist_ok=True)
    frame.to_parquet(DATA / "panel.parquet", index=False)
    table = pd.DataFrame(balance_rows)
    table.to_csv(DATA / "panel_balance.csv", index=False, float_format="%.4g")
    print(frame.groupby(["bin", "set"]).size().unstack())
    print(table.to_string(index=False, float_format="%.3f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
