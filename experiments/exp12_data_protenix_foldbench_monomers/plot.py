# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render comparison PNGs from data/scores.csv.

Two plots per metric, for each of the five score.py outputs:

- ``mae_distogram_cb_angstrom`` — distogram MAE on CB.
- ``mae_structure_ca_angstrom`` — structure-distance-derived MAE on CA.
- ``drmsd_ca_angstrom``         — dRMSD on CA.
- ``rmsd_ca_angstrom``          — Kabsch RMSD on CA.
- ``rmsd_all_heavy_angstrom``   — Kabsch RMSD on all matching heavy atoms.

Per metric:

- ``plots/{metric}_per_protein.png`` — grouped bar chart, one bar pair
  (single_seq / msa) per protein, sorted by protein size.
- ``plots/{metric}_ss_vs_msa_scatter.png`` — paired scatter, x = MSA
  mode, y = single_seq mode, with the y=x diagonal. Each point is one
  protein.

A small summary CSV ``data/scores_summary.csv`` is also written: per-mode
mean / median / min / max of each metric. Useful headline numbers for
the issue comment.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta


_METRICS = (
    # Distance-based, in-range pair set (option B).
    ("mae_distogram_cb_angstrom", "Distogram MAE on CB (Å, in-range)"),
    ("drmsd_distogram_cb_angstrom", "Distogram dRMSD on CB (Å, in-range)"),
    # Distance-based, contact-regime pair set (option C1).
    ("mae_distogram_cb_contact_angstrom", "Distogram MAE on CB (Å, contacts only)"),
    ("drmsd_distogram_cb_contact_angstrom", "Distogram dRMSD on CB (Å, contacts only)"),
    # Structure-based.
    ("mae_structure_ca_angstrom", "Structure-distance MAE on CA (Å)"),
    ("drmsd_ca_angstrom", "dRMSD on CA (Å)"),
    ("rmsd_ca_angstrom", "Kabsch RMSD on CA (Å)"),
    ("rmsd_all_heavy_angstrom", "Kabsch RMSD on all heavy atoms (Å)"),
    # CASP contact precision (option C2). Long-range only on the plots;
    # short and medium still in the CSV / summary.
    ("prec_long_L", "CASP precision @ top L, long range (sep ≥ 24)"),
    ("prec_long_L_5", "CASP precision @ top L/5, long range (sep ≥ 24)"),
    # LDDT (CASP convention: 15 Å inclusion, thresholds 0.5/1/2/4 Å).
    ("lddt_structure_ca", "LDDT-CA from predicted structure"),
    ("lddt_structure_cb", "LDDT-CB from predicted structure"),
    ("lddt_structure_all_heavy", "LDDT-all-heavy from predicted structure"),
    ("lddt_distogram_cb", "LDDT-CB from distogram (point estimate)"),
    ("lddt_distogram_cb_soft", "LDDT-CB from distogram (soft / probabilistic)"),
)


def _load(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    numeric_cols = (
        "ranking_score", "selected_as_best", "n_residues",
        "mae_distogram_cb_angstrom", "drmsd_distogram_cb_angstrom",
        "n_mae_distogram_pairs",
        "mae_distogram_cb_contact_angstrom", "drmsd_distogram_cb_contact_angstrom",
        "n_mae_distogram_contact_pairs",
        "mae_structure_ca_angstrom",
        "drmsd_ca_angstrom", "n_ca_pairs",
        "rmsd_ca_angstrom", "n_ca_atoms",
        "rmsd_all_heavy_angstrom", "n_heavy_atoms",
        "prec_short_L", "prec_short_L_2", "prec_short_L_5",
        "prec_medium_L", "prec_medium_L_2", "prec_medium_L_5",
        "prec_long_L", "prec_long_L_2", "prec_long_L_5",
        "n_short_contacts", "n_medium_contacts", "n_long_contacts",
        "lddt_structure_ca", "lddt_structure_cb", "lddt_structure_all_heavy",
        "lddt_distogram_cb", "lddt_distogram_cb_soft",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # When called on the all-samples CSV, restrict to top-1 rows for the
    # per-protein plots (those plots assume one (mode, stem) per row).
    if "selected_as_best" in df.columns and (df["selected_as_best"] == 0).any():
        df = df[df["selected_as_best"] == 1].reset_index(drop=True)
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
    save_plot_with_meta(
        fig, out_png,
        caption=(
            f"Per-protein {ylabel}, single_seq vs MSA. Bars sorted by "
            f"protein length. Top-1 sample per (protein, mode) by Protenix's "
            f"ranking_score."
        ),
        dpi=150,
    )
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
    save_plot_with_meta(
        fig, out_png,
        caption=(
            f"Paired {label}, y=single_seq, x=MSA. Each point is one "
            f"protein; dashed line is y=x. Below the diagonal: MSA mode "
            f"is better (typical for distance/structure metrics)."
        ),
        dpi=150,
    )
    plt.close(fig)
    print(f"Wrote {out_png}")


def _summary(df: pd.DataFrame, out_csv: Path) -> None:
    rows: list[dict] = []
    for mode, sub in df.groupby("mode"):
        for metric, _ in _METRICS:
            if metric not in sub.columns:
                continue
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
    """Top-level: load scores, render PNGs + summary CSV for all metrics."""
    df = _load(scores_csv)
    if df.empty:
        print(f"WARN: {scores_csv} is empty.")
        return
    for metric, label in _METRICS:
        if metric not in df.columns:
            print(f"WARN: {metric} not in scores CSV; skipping.")
            continue
        _per_protein_bars(df, metric, label, out_dir / f"{metric}_per_protein.png")
        _ss_vs_msa_scatter(df, metric, label, out_dir / f"{metric}_ss_vs_msa_scatter.png")
    summary_csv = scores_csv.parent / (scores_csv.stem + "_summary.csv")
    _summary(df, summary_csv)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("plot", help="Render PNGs + summary CSV from data/scores.csv.")
    p.add_argument("--scores", type=Path, required=True, help="data/scores.csv path.")
    p.add_argument("--out", type=Path, required=True, help="Output dir for PNGs (typically plots/).")
    p.set_defaults(func=lambda args: plot(scores_csv=args.scores, out_dir=args.out))
