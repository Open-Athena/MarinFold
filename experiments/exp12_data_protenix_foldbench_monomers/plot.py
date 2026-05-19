# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render comparison PNGs from data/scores.csv.

Two plots per metric (MAE in Å, dRMSD in Å):

- ``plots/{metric}_per_protein.png`` — grouped bar chart, one bar pair
  (single_seq / msa) per protein, sorted by protein size.
- ``plots/{metric}_ss_vs_msa_scatter.png`` — paired scatter, x = MSA
  mode, y = single_seq mode, with the y=x diagonal. Each point is one
  protein.

A small summary CSV ``data/scores_summary.csv`` is also written: per-mode
mean / median of each metric. Useful headline numbers for the issue
comment.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    # Ensure numeric (csv module wrote strings if we hand-wrote).
    for col in ["mae_angstrom", "drmsd_angstrom", "n_residues", "ranking_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pair_modes(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to one row per protein with single_seq / msa columns for ``metric``."""
    sub = df[["pdb_id", "chain_id", "mode", "n_residues", metric]].copy()
    sub["stem"] = sub["pdb_id"] + "_" + sub["chain_id"]
    pivoted = sub.pivot_table(
        index=["stem", "n_residues"], columns="mode", values=metric, aggfunc="first"
    ).reset_index()
    return pivoted.dropna(subset=["single_seq", "msa"], how="all")


def _per_protein_bars(df: pd.DataFrame, metric: str, ylabel: str, out_png: Path) -> None:
    pivoted = _pair_modes(df, metric).sort_values("n_residues")
    if pivoted.empty:
        print(f"WARN: no data for {metric}; skipping bar plot.")
        return
    stems = pivoted["stem"].tolist()
    x = np.arange(len(stems))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(stems) + 2), 4))
    ss = pivoted["single_seq"].to_numpy()
    msa = pivoted["msa"].to_numpy()
    ax.bar(x - width / 2, ss, width=width, label="single_seq")
    ax.bar(x + width / 2, msa, width=width, label="msa")
    ax.set_xticks(x)
    labels = [f"{s}\n({int(n)}aa)" for s, n in zip(stems, pivoted["n_residues"].tolist())]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Protenix v2 — {ylabel} per protein (FoldBench monomers)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _ss_vs_msa_scatter(df: pd.DataFrame, metric: str, label: str, out_png: Path) -> None:
    pivoted = _pair_modes(df, metric).dropna(subset=["single_seq", "msa"])
    if pivoted.empty:
        print(f"WARN: no paired (single_seq, msa) data for {metric}; skipping scatter.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pivoted["msa"], pivoted["single_seq"], s=30, alpha=0.7)
    lo = min(pivoted["msa"].min(), pivoted["single_seq"].min())
    hi = max(pivoted["msa"].max(), pivoted["single_seq"].max())
    pad = (hi - lo) * 0.05 if hi > lo else 0.5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=1, label="y=x")
    ax.set_xlabel(f"{label} (MSA)")
    ax.set_ylabel(f"{label} (single-seq)")
    ax.set_title(f"Protenix v2 — single-seq vs MSA, {label}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _summary(df: pd.DataFrame, out_csv: Path) -> None:
    rows: list[dict] = []
    for mode, sub in df.groupby("mode"):
        for metric in ("mae_angstrom", "drmsd_angstrom"):
            rows.append({
                "mode": mode,
                "metric": metric,
                "n": int(sub[metric].notna().sum()),
                "mean": float(sub[metric].mean(skipna=True)),
                "median": float(sub[metric].median(skipna=True)),
                "min": float(sub[metric].min(skipna=True)),
                "max": float(sub[metric].max(skipna=True)),
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


def plot(*, scores_csv: Path, out_dir: Path) -> None:
    """Top-level: load scores, render PNGs + summary CSV."""
    df = _load(scores_csv)
    if df.empty:
        print(f"WARN: {scores_csv} is empty.")
        return
    _per_protein_bars(df, "mae_angstrom", "MAE (Å)", out_dir / "mae_per_protein.png")
    _per_protein_bars(df, "drmsd_angstrom", "dRMSD (Å)", out_dir / "drmsd_per_protein.png")
    _ss_vs_msa_scatter(df, "mae_angstrom", "MAE (Å)", out_dir / "mae_ss_vs_msa_scatter.png")
    _ss_vs_msa_scatter(df, "drmsd_angstrom", "dRMSD (Å)", out_dir / "drmsd_ss_vs_msa_scatter.png")
    summary_csv = scores_csv.parent / (scores_csv.stem + "_summary.csv")
    _summary(df, summary_csv)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("plot", help="Render PNGs + summary CSV from data/scores.csv.")
    p.add_argument("--scores", type=Path, required=True, help="data/scores.csv path.")
    p.add_argument("--out", type=Path, required=True, help="Output dir for PNGs (typically plots/).")
    p.set_defaults(func=lambda args: plot(scores_csv=args.scores, out_dir=args.out))
