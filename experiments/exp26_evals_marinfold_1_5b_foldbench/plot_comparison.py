# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plots for the 4-way comparison (1.5B / 1B / Protenix single_seq / msa).

Reads ``data/scores.csv`` (output of ``score_comparison.py``) and
produces per-protein bar charts, an aggregate headline figure,
paired scatters anchored on ``marinfold_1_5b``, and the swarm plots.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


_METHOD_COLORS = {
    "marinfold_1_5b": "#e6550d",
    "marinfold_1b": "#d95f02",
    "protenix_single_seq": "#7570b3",
    "protenix_msa": "#1b9e77",
}
_METHOD_ORDER = (
    "marinfold_1_5b", "marinfold_1b", "protenix_single_seq", "protenix_msa",
)


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _first_nonempty(*values) -> str:
    """First value that's a non-empty string. Skips NaN (which is truthy in Python)."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return v
    return ""


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
    """Bar chart of mean and median for the four headline metrics."""
    metrics = [
        ("lddt_distogram_cb", "LDDT-distogram-CB (higher=better)"),
        ("mae_distogram_cb_angstrom", "MAE-distogram-CB (Å, lower=better)"),
        ("drmsd_distogram_cb_angstrom", "dRMSD-distogram-CB (Å, lower=better)"),
        ("prec_long_L", "prec_long_L (higher=better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4))
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
    """Paired per-protein scatter anchored on ``marinfold_1_5b``.

    Three panels (1.5B vs each of: 1B, protenix_single_seq,
    protenix_msa). Points above the y=x diagonal mean the comparator
    beats 1.5B; points below mean 1.5B beats the comparator. The
    1.5B-vs-1B panel is the headline view for this experiment.
    """
    anchor = df[df["method"] == "marinfold_1_5b"].set_index("pdb_id")[metric]
    others = ("marinfold_1b", "protenix_single_seq", "protenix_msa")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=False)
    for ax, comp in zip(axes, others, strict=True):
        comp_vals = df[df["method"] == comp].set_index("pdb_id")[metric]
        common = anchor.index.intersection(comp_vals.index)
        x_vals = anchor.loc[common].to_numpy()
        y_vals = comp_vals.loc[common].to_numpy()
        ax.scatter(x_vals, y_vals, s=14, color=_METHOD_COLORS[comp], alpha=0.7)
        lo = float(np.nanmin([x_vals.min(), y_vals.min()])) if len(common) else 0
        hi = float(np.nanmax([x_vals.max(), y_vals.max()])) if len(common) else 1
        ax.plot([lo, hi], [lo, hi], color="#bbbbbb", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"marinfold_1_5b  {metric}")
        ax.set_ylabel(f"{comp}  {metric}")
        ax.set_title(f"{comp} vs marinfold_1_5b")
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


def lddt_5way_swarm(
    *,
    merged_scores_csv: Path,
    protenix_scores_csv: Path,
    out_path: Path,
) -> None:
    """LDDT swarm across all 4 methods + Protenix structure categories.

    Per method:
      - MarinFold 1.5B, 1B: distogram + distogram-soft (2 categories each).
      - Protenix single_seq, msa: distogram + distogram-soft + structure-CB
        (3 categories each).

    The merged ``data/scores.csv`` only has distogram-LDDT (structure
    metrics were dropped from the schema). The Protenix-side
    structure-LDDT lives in ``protenix_data/.../scores.csv``
    (column ``lddt_structure_cb``, the CB / CA-for-GLY variant —
    apples-to-apples with the CB-CB distogram LDDT).

    Points are seaborn ``swarmplot`` (no overlap), overlaid with a
    light grey boxplot at alpha=0.5 so the median + quartiles are
    visible against the swarm.
    """
    merged = pd.read_csv(merged_scores_csv)
    protenix = pd.read_csv(protenix_scores_csv)

    rows: list[dict] = []
    # MarinFold (1.5B and 1B): distogram (point) + distogram-soft.
    for marin_method in ("marinfold_1_5b", "marinfold_1b"):
        mf = merged[merged["method"] == marin_method][
            ["pdb_id", "chain_id", "lddt_distogram_cb", "lddt_distogram_cb_soft"]
        ]
        for _, r in mf.iterrows():
            stem = f"{r['pdb_id']}_{r['chain_id']}"
            rows.append({"protein": stem,
                         "category": f"{marin_method} (distogram)",
                         "lddt": r["lddt_distogram_cb"]})
            rows.append({"protein": stem,
                         "category": f"{marin_method} (distogram, soft)",
                         "lddt": r["lddt_distogram_cb_soft"]})

    # Protenix single_seq + msa: distogram + distogram-soft + structure-CB.
    for mode in ("single_seq", "msa"):
        sub = protenix[protenix["mode"] == mode]
        for _, r in sub.iterrows():
            stem = f"{r['pdb_id']}_{r['chain_id']}"
            rows.append({"protein": stem,
                         "category": f"protenix_{mode} (distogram)",
                         "lddt": r["lddt_distogram_cb"]})
            rows.append({"protein": stem,
                         "category": f"protenix_{mode} (distogram, soft)",
                         "lddt": r["lddt_distogram_cb_soft"]})
            rows.append({"protein": stem,
                         "category": f"protenix_{mode} (structure)",
                         "lddt": r["lddt_structure_cb"]})

    df = pd.DataFrame(rows).dropna(subset=["lddt"])
    order = [
        "marinfold_1_5b (distogram)",
        "marinfold_1_5b (distogram, soft)",
        "marinfold_1b (distogram)",
        "marinfold_1b (distogram, soft)",
        "protenix_single_seq (distogram)",
        "protenix_single_seq (distogram, soft)",
        "protenix_single_seq (structure)",
        "protenix_msa (distogram)",
        "protenix_msa (distogram, soft)",
        "protenix_msa (structure)",
    ]
    palette = {cat: _METHOD_COLORS[cat.split(" (")[0]] for cat in order}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    # Boxplot first (so the swarm renders on top). Light grey
    # boxes at alpha=0.5; no outliers (the swarm shows them).
    sns.boxplot(
        data=df, x="category", y="lddt", order=order,
        showfliers=False, width=0.55,
        boxprops={"facecolor": "#cccccc", "alpha": 0.5, "edgecolor": "#888888"},
        medianprops={"color": "#222222", "linewidth": 2.0},
        whiskerprops={"color": "#888888"}, capprops={"color": "#888888"},
        ax=ax,
    )
    sns.swarmplot(
        data=df, x="category", y="lddt", order=order,
        palette=palette, hue="category", legend=False,
        size=3.5, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("LDDT")
    ax.set_title(f"LDDT-CB (n={df['protein'].nunique()} FoldBench monomers)")
    ax.set_ylim(-0.02, 1.02)
    # Wrap long x-tick labels onto two lines for readability.
    ax.set_xticklabels(
        [t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()],
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def marinfold_vs_protenix_timing(
    *,
    marinfold_1_5b_timings_csv: Path,
    marinfold_1b_timings_csv: Path | None,
    protenix_timings_csv: Path,
    out_path: Path,
) -> None:
    """4-way per-protein wall-time vs sequence length.

    Joins exp26's own 1.5B timings, exp20's 1B timings, and the
    Protenix exp12 timings (single_seq + msa). All use
    ``elapsed_seconds`` as the post-model-load inference time, so
    the comparison is steady-state per-protein cost.

    Hardware caveat — the three MarinFold/Protenix runs were on
    different accelerators (1.5B on TPU v5p-8 via iris, 1B on H100
    via Modal, Protenix on H100). The plot annotates each series
    with its hardware so the comparison is read carefully. Pair-sweep
    cost still scales O(N²) regardless of hardware, which is what
    the log-log shape captures.

    Both axes log scale; one marker per (protein, method).
    """
    if not protenix_timings_csv.exists():
        print(f"skip 4-way timing plot: {protenix_timings_csv} not found.")
        return
    mf_15b = pd.read_csv(marinfold_1_5b_timings_csv)
    px = pd.read_csv(protenix_timings_csv)
    rows: list[dict] = []
    for _, r in mf_15b.iterrows():
        rows.append({
            "method": "marinfold_1_5b",
            "n_residues": r["n_residues"],
            "elapsed_seconds": r["elapsed_seconds"],
            "hardware": _first_nonempty(r.get("gpu_name"), r.get("runner_tag")),
        })
    if marinfold_1b_timings_csv is not None and marinfold_1b_timings_csv.exists():
        mf_1b = pd.read_csv(marinfold_1b_timings_csv)
        for _, r in mf_1b.iterrows():
            rows.append({
                "method": "marinfold_1b",
                "n_residues": r["n_residues"],
                "elapsed_seconds": r["elapsed_seconds"],
                "hardware": _first_nonempty(r.get("gpu_name"), r.get("runner_tag")),
            })
    for _, r in px.iterrows():
        rows.append({
            "method": f"protenix_{r['mode']}",
            "n_residues": r["n_residues"],
            "elapsed_seconds": r["elapsed_seconds"],
            "hardware": r.get("gpu_name", ""),
        })
    df = pd.DataFrame(rows).dropna(subset=["elapsed_seconds", "n_residues"])

    # Hardware label per method (most common value in the series).
    def _hw_label(method: str) -> str:
        sub = df[df["method"] == method]
        if sub.empty:
            return ""
        top = sub["hardware"].dropna().value_counts()
        if top.empty:
            return ""
        return f", {top.idxmax().split()[0]}"  # first token, e.g. 'NVIDIA' or 'iris'

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in ("marinfold_1_5b", "marinfold_1b", "protenix_single_seq", "protenix_msa"):
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["n_residues"], sub["elapsed_seconds"],
            color=_METHOD_COLORS[method], alpha=0.7, s=22,
            label=f"{method} (n={len(sub)}{_hw_label(method)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (residues)")
    ax.set_ylabel("inference wall-time (seconds, post model load)")
    ax.set_title("Per-protein inference cost vs sequence length")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def prec_at_l_swarm(*, merged_scores_csv: Path, out_path: Path) -> None:
    """Swarm plot of CASP top-L contact precision per (method, sep class).

    9 categories: 3 methods × {short, medium, long} sequence separation.
    All three methods produce a distogram, so each method has one column
    per separation class (``prec_{short,medium,long}_L``). Structure-
    derived contact precision (binarize predicted CB-CB at 8 Å) isn't
    in exp12's score schema, so this plot is distogram-only.
    """
    merged = pd.read_csv(merged_scores_csv)
    rows: list[dict] = []
    methods = _METHOD_ORDER
    sep_classes = ("short", "medium", "long")
    for method in methods:
        sub = merged[merged["method"] == method]
        for _, r in sub.iterrows():
            stem = f"{r['pdb_id']}_{r['chain_id']}"
            for sep in sep_classes:
                rows.append({"protein": stem,
                             "category": f"{method} ({sep})",
                             "prec": r[f"prec_{sep}_L"]})

    df = pd.DataFrame(rows).dropna(subset=["prec"])
    order = [f"{m} ({s})" for s in sep_classes for m in methods]
    palette = {f"{m} ({s})": _METHOD_COLORS[m] for s in sep_classes for m in methods}

    fig, ax = plt.subplots(figsize=(15, 5.5))
    sns.boxplot(
        data=df, x="category", y="prec", order=order,
        showfliers=False, width=0.55,
        boxprops={"facecolor": "#cccccc", "alpha": 0.5, "edgecolor": "#888888"},
        medianprops={"color": "#222222", "linewidth": 2.0},
        whiskerprops={"color": "#888888"}, capprops={"color": "#888888"},
        ax=ax,
    )
    sns.swarmplot(
        data=df, x="category", y="prec", order=order,
        palette=palette, hue="category", legend=False,
        size=2.0, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("precision @ top L")
    ax.set_title(f"CASP contact precision @ L (n={df['protein'].nunique()} FoldBench monomers)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticklabels(
        [t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()],
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render(*, scores_csv: Path, out_dir: Path, timings_csv: Path | None = None,
           protenix_scores_csv: Path | None = None,
           protenix_timings_csv: Path | None = None,
           marinfold_1b_timings_csv: Path | None = None) -> None:
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
    if protenix_scores_csv is not None and protenix_scores_csv.exists():
        lddt_5way_swarm(
            merged_scores_csv=scores_csv,
            protenix_scores_csv=protenix_scores_csv,
            out_path=out_dir / "lddt_5way_swarm.png",
        )
    prec_at_l_swarm(
        merged_scores_csv=scores_csv,
        out_path=out_dir / "prec_L_swarm.png",
    )
    if timings_csv is not None and protenix_timings_csv is not None:
        marinfold_vs_protenix_timing(
            marinfold_1_5b_timings_csv=timings_csv,
            marinfold_1b_timings_csv=marinfold_1b_timings_csv,
            protenix_timings_csv=protenix_timings_csv,
            out_path=out_dir / "timing_4way_vs_sequence_length.png",
        )
    print(f"Wrote plots to {out_dir}/")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=here / "data" / "scores.csv")
    parser.add_argument("--out", type=Path, default=here / "plots")
    parser.add_argument("--timings", type=Path, default=here / "data" / "timings.csv")
    parser.add_argument(
        "--protenix-scores", type=Path,
        default=here / "protenix_data" / "data" / "protenix-foldbench-monomers" / "scores.csv",
        help="Source for Protenix structure-LDDT (5-way swarm plot).",
    )
    parser.add_argument(
        "--protenix-timings", type=Path,
        default=here.parent / "exp12_data_protenix_foldbench_monomers" / "data" / "timings.csv",
        help="Source for Protenix timings (4-way timing-vs-length plot).",
    )
    parser.add_argument(
        "--marinfold-1b-timings", type=Path,
        default=here.parent / "exp20_evals_marinfold_1b_foldbench" / "data" / "timings.csv",
        help="Source for exp20 MarinFold 1B timings (4-way timing-vs-length plot).",
    )
    args = parser.parse_args()
    render(scores_csv=args.scores, out_dir=args.out, timings_csv=args.timings,
           protenix_scores_csv=args.protenix_scores,
           protenix_timings_csv=args.protenix_timings,
           marinfold_1b_timings_csv=args.marinfold_1b_timings)


if __name__ == "__main__":
    main()
