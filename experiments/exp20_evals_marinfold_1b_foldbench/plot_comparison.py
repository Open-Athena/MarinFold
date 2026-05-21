# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plots for the 3-way comparison.

Reads ``data/scores.csv`` (output of ``score_comparison.py``) and
produces:

- ``plots/lddt_per_protein.png`` — per-protein grouped bar chart of
  LDDT-distogram-CB for all three methods.
- ``plots/mae_per_protein.png`` — same for MAE-distogram-CB.
- ``plots/headline_aggregate.png`` — bar chart of mean and median for
  each headline metric per method.
- ``plots/marinfold_vs_protenix_scatter.png`` — paired scatter,
  MarinFold on x, Protenix on y (one panel per Protenix mode, per
  metric).
- ``plots/prec_long_L_per_protein.png`` — CASP long-range contact
  precision @ top L, per protein per method.
- ``plots/timing_vs_sequence_length.png`` (if ``data/timings.csv``
  is present) — log-log scatter of per-protein runtime vs sequence
  length, one series per GPU.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_METHOD_COLORS = {
    "marinfold_1b": "#d95f02",
    "protenix_single_seq": "#7570b3",
    "protenix_msa": "#1b9e77",
}
_METHOD_ORDER = ("marinfold_1b", "protenix_single_seq", "protenix_msa")


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def per_protein_bar(
    df: pd.DataFrame, *, metric: str, ylabel: str, out_path: Path,
    higher_is_better: bool,
) -> None:
    """Grouped bar: x=protein, hue=method, y=metric."""
    proteins = sorted(df["pdb_id"].unique())
    if not proteins:
        return
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(10, 0.15 * len(proteins) + 4), 4.5))
    x_base = np.arange(len(proteins))
    for offset, method in enumerate(_METHOD_ORDER):
        sub = df[df["method"] == method].set_index("pdb_id")
        vals = [sub.loc[p, metric] if p in sub.index else np.nan for p in proteins]
        ax.bar(
            x_base + (offset - 1) * width, vals, width,
            label=method, color=_METHOD_COLORS[method], alpha=0.85,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(proteins, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} per protein — {'higher' if higher_is_better else 'lower'} is better")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def headline_aggregate(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of mean and median for the three headline metrics."""
    metrics = [
        ("lddt_distogram_cb", "LDDT-distogram-CB (higher=better)"),
        ("mae_distogram_cb_angstrom", "MAE-distogram-CB (Å, lower=better)"),
        ("drmsd_distogram_cb_angstrom", "dRMSD-distogram-CB (Å, lower=better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    methods = list(_METHOD_ORDER)
    x = np.arange(len(methods))
    width = 0.4
    for ax, (col, title) in zip(axes, metrics, strict=True):
        means = [df[df["method"] == m][col].mean() for m in methods]
        medians = [df[df["method"] == m][col].median() for m in methods]
        ax.bar(x - width / 2, means, width, label="mean", color="#666666", alpha=0.85)
        ax.bar(x + width / 2, medians, width, label="median", color="#bbbbbb", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def paired_scatter(df: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    """Paired per-protein scatter: x=marinfold_1b, y=protenix_{mode}.

    One panel per Protenix mode. y=x diagonal in grey for reference.
    """
    mf = df[df["method"] == "marinfold_1b"].set_index("pdb_id")[metric]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, mode in zip(axes, ("protenix_single_seq", "protenix_msa"), strict=True):
        px = df[df["method"] == mode].set_index("pdb_id")[metric]
        common = mf.index.intersection(px.index)
        x_vals = mf.loc[common].to_numpy()
        y_vals = px.loc[common].to_numpy()
        ax.scatter(x_vals, y_vals, s=14, color=_METHOD_COLORS[mode], alpha=0.7)
        lo = float(np.nanmin([x_vals.min(), y_vals.min()])) if len(common) else 0
        hi = float(np.nanmax([x_vals.max(), y_vals.max()])) if len(common) else 1
        ax.plot([lo, hi], [lo, hi], color="#bbbbbb", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"marinfold_1b  {metric}")
        ax.set_ylabel(f"{mode}  {metric}")
        ax.set_title(f"{mode} vs marinfold_1b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def timing_vs_sequence_length(timings_csv: Path, out_path: Path) -> None:
    """Log-log scatter: per-protein runtime vs sequence length, by GPU.

    Reads the CSV produced by ``collect_timings.py``. One color per
    distinct ``gpu_name``; ``runner_tag`` (local / modal-h100 / etc.)
    is shown as marker shape. Diagonal is omitted because the
    quadratic-pairs trend dominates; we just plot the points and
    label sufficient context for someone to extrapolate.
    """
    if not timings_csv.exists():
        return
    df = pd.read_csv(timings_csv)
    if df.empty or "elapsed_seconds" not in df.columns:
        return
    # Drop incomplete rows (e.g. ones from older provenance.json
    # files written before timing fields were added).
    df = df.dropna(subset=["elapsed_seconds", "n_residues"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    gpus = list(df["gpu_name"].fillna("unknown").unique())
    palette = plt.colormaps["tab10"].resampled(max(len(gpus), 1))
    markers = {"local": "o", "modal": "s"}
    for i, gpu in enumerate(gpus):
        sub = df[df["gpu_name"].fillna("unknown") == gpu]
        for tag, marker in markers.items():
            tag_sub = sub[sub["runner_tag"].fillna("local").str.startswith(tag)]
            if tag_sub.empty:
                continue
            ax.scatter(
                tag_sub["n_residues"], tag_sub["elapsed_seconds"],
                s=24, color=palette(i), marker=marker, alpha=0.85,
                label=f"{gpu} ({tag})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (residues)")
    ax.set_ylabel("inference wall-time (seconds)")
    ax.set_title("MarinFold 1B per-protein runtime")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render(*, scores_csv: Path, out_dir: Path, timings_csv: Path | None = None) -> None:
    df = pd.read_csv(scores_csv)
    _ensure_outdir(out_dir)
    per_protein_bar(
        df, metric="lddt_distogram_cb", ylabel="LDDT",
        out_path=out_dir / "lddt_per_protein.png", higher_is_better=True,
    )
    per_protein_bar(
        df, metric="mae_distogram_cb_angstrom", ylabel="MAE (Å)",
        out_path=out_dir / "mae_per_protein.png", higher_is_better=False,
    )
    per_protein_bar(
        df, metric="prec_long_L", ylabel="precision@L (long-range)",
        out_path=out_dir / "prec_long_L_per_protein.png", higher_is_better=True,
    )
    headline_aggregate(df, out_dir / "headline_aggregate.png")
    paired_scatter(
        df, metric="lddt_distogram_cb",
        out_path=out_dir / "lddt_marinfold_vs_protenix_scatter.png",
    )
    paired_scatter(
        df, metric="mae_distogram_cb_angstrom",
        out_path=out_dir / "mae_marinfold_vs_protenix_scatter.png",
    )
    if timings_csv is not None:
        timing_vs_sequence_length(timings_csv, out_dir / "timing_vs_sequence_length.png")
    print(f"Wrote plots to {out_dir}/")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=here / "data" / "scores.csv")
    parser.add_argument("--out", type=Path, default=here / "plots")
    parser.add_argument("--timings", type=Path, default=here / "data" / "timings.csv")
    args = parser.parse_args()
    render(scores_csv=args.scores, out_dir=args.out, timings_csv=args.timings)


if __name__ == "__main__":
    main()
