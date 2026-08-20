# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate + paired-difference table for the #169 checkpoints.

The three checkpoints in issue #169 sit within **0.008 nats** of each other on
`eval/tokenized/contacts-v1-val/loss`. At that separation an unpaired
mean±SEM over 554 proteins is far too blunt to say anything: the between-protein
spread of R-precision is ~0.3, so the SEM alone is ~0.013 — wider than the whole
effect we are trying to resolve.

Every checkpoint is scored on the *same* 554 proteins, though, so the paired
difference is the right statistic. Its standard error is an order of magnitude
smaller because the protein-to-protein variance cancels. This script therefore
reports, for every pair of models:

* the mean per-protein difference and its 95% CI (paired t, normal approximation),
* the win rate (fraction of proteins where A > B),

alongside the plain aggregate table (R / L / L2 / L5 / AUC by range).

    uv run python summarize_results.py --rows data/exp169_rows.csv.gz \\
        --out-summary data/exp169_summary.csv --out-paired data/exp169_paired.csv
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Display order and the val loss each checkpoint reported at its step, so the
# loss -> accuracy relationship can be read straight off the table. Losses are
# the `eval/tokenized/contacts-v1-val/loss` values quoted in issue #169 (and, for
# the #61/#75 reference row, in #89).
CHECKPOINTS = [
    ("marinfold-cv1-exp75-rollout", "#61/#75 · 1.5B · E8 · step 35679", "#75 E8", 2.756602),
    ("exp117_e16_final_step35679", "#117 · 1.5B · E16 · step 35679 (final)",
     "#117 final", 2.703709),
    ("exp117_e16_early_step33450", "#117 · 1.5B · E16 · step 33450 (early stop)",
     "#117 early", 2.696074),
    ("exp146_3b_e8_step17839", "#146 · 3B · E8 · step 17839", "#146 3B", 2.702478),
]
LABELS = {key: label for key, label, _, _ in CHECKPOINTS}
SHORT = {key: short for key, _, short, _ in CHECKPOINTS}
LOSSES = {key: loss for key, _, _, loss in CHECKPOINTS}


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (model, range, cut), plus the count it averaged."""
    grouped = rows.groupby(["model", "range", "cut"])["precision"]
    out = grouped.agg(mean_precision="mean", n_valid="count").reset_index()
    out["n_units"] = grouped.size().values
    return out


def paired_differences(rows: pd.DataFrame, *, cut: str, rng: str) -> pd.DataFrame:
    """Per-protein paired comparison of every model pair for one metric.

    Models are joined on (dataset, stem); only proteins where *both* models have
    a finite value contribute, and the number that did is reported so a silently
    shrunken comparison is visible.
    """
    sub = rows[(rows["cut"] == cut) & (rows["range"] == rng)]
    wide = sub.pivot_table(index=["dataset", "stem"], columns="model", values="precision")
    present = [k for k, _, _, _ in CHECKPOINTS if k in wide.columns]

    records = []
    for a, b in itertools.combinations(present, 2):
        pair = wide[[a, b]].dropna()
        delta = (pair[a] - pair[b]).to_numpy()
        n = delta.size
        if n < 2:
            continue
        sem = delta.std(ddof=1) / np.sqrt(n)
        records.append(dict(
            range=rng, cut=cut, model_a=a, model_b=b, n_paired=n,
            mean_a=float(pair[a].mean()), mean_b=float(pair[b].mean()),
            mean_delta=float(delta.mean()),
            ci_low=float(delta.mean() - 1.96 * sem), ci_high=float(delta.mean() + 1.96 * sem),
            sem=float(sem),
            win_rate_a=float((delta > 0).mean()), tie_rate=float((delta == 0).mean()),
            delta_loss=LOSSES.get(b, np.nan) - LOSSES.get(a, np.nan),
        ))
    return pd.DataFrame.from_records(records)


def print_aggregate(summary: pd.DataFrame) -> None:
    view = summary[summary["range"].isin(["all", "long"])]
    table = view.pivot_table(index="model", columns=["range", "cut"], values="mean_precision")
    order = [k for k, _, _, _ in CHECKPOINTS if k in table.index]
    columns = [(r, c) for r in ("all", "long") for c in ("R", "L", "L/2", "L/5", "AUC")]
    table = table.loc[order, [c for c in columns if c in table.columns]]

    print("\n=== Aggregate (mean over proteins) ===")
    print(f"{'checkpoint':<44}{'loss':>9}"
          + "".join(f"{rng[:3]}/{cut}".rjust(10) for rng, cut in table.columns))
    for key, row in table.iterrows():
        cells = "".join(f"{v:10.4f}" for v in row.to_numpy())
        print(f"{LABELS.get(key, key):<44}{LOSSES.get(key, float('nan')):9.4f}{cells}")


def print_paired(paired: pd.DataFrame) -> None:
    print("\n=== Paired per-protein differences, R-precision (range=all) ===")
    print(f"{'A':<13}{'B':<13}{'n':>5}{'A mean':>9}{'B mean':>9}{'Δ mean':>9}"
          f"{'95% CI':>20}{'A wins':>8}{'Δ loss':>9}")
    for _, r in paired.iterrows():
        ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        significant = " *" if r.ci_low * r.ci_high > 0 else ""
        print(f"{SHORT.get(r.model_a, r.model_a):<13}{SHORT.get(r.model_b, r.model_b):<13}"
              f"{int(r.n_paired):5d}{r.mean_a:9.4f}{r.mean_b:9.4f}{r.mean_delta:+9.4f}"
              f"{ci:>20}{r.win_rate_a:8.3f}{r.delta_loss:+9.4f}{significant}")
    print("  * = 95% CI excludes zero.  Δ loss > 0 means A had the lower val loss.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True, help="per-protein rows CSV")
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--out-paired", type=Path, required=True)
    a = ap.parse_args()

    rows = pd.read_csv(a.rows)
    known = set(LABELS)
    missing = known - set(rows["model"].unique())
    if missing:
        print(f"!! rows are missing model(s): {sorted(missing)}")

    summary = aggregate(rows)
    summary.to_csv(a.out_summary, index=False)
    print_aggregate(summary)

    paired = pd.concat(
        [paired_differences(rows, cut=c, rng=r)
         for r in ("all", "long") for c in ("R", "L", "AUC")],
        ignore_index=True)
    paired.to_csv(a.out_paired, index=False)
    print_paired(paired[(paired["cut"] == "R") & (paired["range"] == "all")])

    print(f"\nwrote {a.out_summary} and {a.out_paired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
