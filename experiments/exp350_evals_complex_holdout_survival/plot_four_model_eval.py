"""Render matched-complex accuracy figures and a compact comparison PDF."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from build_summary import save_plot_with_meta
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from score_four_model_eval import MODELS

HERE = Path(__file__).resolve().parent
COLORS = ["#31688e", "#d07a29", "#328a6d", "#8664a0"]
LABELS = [
    "MarinFold\nmultichain",
    "MarinFold default\n+ 10 glycines",
    "ESMFold2\nsingle-sequence",
    "AlphaFold3\nColabFold MSAs",
]


def main() -> None:
    """Plot the identical 17 test complexes for each model and contact class."""
    rows = list(csv.DictReader((HERE / "data/four_model_v1/per_target.csv").open()))
    summaries = list(csv.DictReader((HERE / "data/four_model_v1/summary.csv").open()))
    rows = [r for r in rows if r["split"] == "test"]
    targets = sorted(
        {r["stem"] for r in rows},
        key=lambda s: next(int(r["n_residues"]) for r in rows if r["stem"] == s),
    )
    assert len(targets) == 17
    lookup = {
        (r["model"], r["region"], r["stem"]): float(r["r_precision"]) for r in rows
    }
    means = {(r["model"], r["region"]): r for r in summaries if r["split"] == "test"}
    assert all(
        int(r["n_targets"]) == 17
        for r in means.values()
        if r["region"] in ("intra", "inter")
    )
    figures = []
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, region, color in [
        (-0.18, "intra", "#2b7a78"),
        (0.18, "inter", "#cc6b49"),
    ]:
        vals = np.array([float(means[m, region]["mean_r_precision"]) for m in MODELS])
        low = np.array([float(means[m, region]["ci_low"]) for m in MODELS])
        high = np.array([float(means[m, region]["ci_high"]) for m in MODELS])
        positions = np.arange(4) + offset
        ax.bar(
            positions,
            vals,
            0.34,
            label="Within original chains"
            if region == "intra"
            else "Between original chains",
            color=color,
            alpha=0.85,
        )
        ax.errorbar(
            positions,
            vals,
            yerr=[vals - low, high - vals],
            fmt="none",
            ecolor="#252525",
            capsize=4,
            linewidth=1.3,
        )
        for x, y, top in zip(positions, vals, high, strict=True):
            ax.text(x, top + 0.025, f"{y:.1%}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(range(4), LABELS)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Mean per-complex R-precision")
    ax.set_title(
        "Within-chain and interface accuracy on the same 17 test complexes", pad=18
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Whiskers: 95% homology-group bootstrap intervals (13 groups). No monomer-only evaluation targets.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figures.append(
        (
            fig,
            "four_model_overview",
            "Same 17 test dimers in both contact classes and all four model arms; means and 95% group-bootstrap intervals.",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for ax, region, title in zip(
        axes,
        ["intra", "inter"],
        ["Within original chains", "Between original chains"],
        strict=True,
    ):
        matrix = np.array([[lookup[m, region, t] for m in MODELS] for t in targets])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), LABELS, fontsize=9)
        ax.set_yticks(range(17), [t.removesuffix("-assembly1") for t in targets])
        ax.set_title(title, pad=14)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="black" if value > 0.65 else "white",
                fontsize=9,
            )
    fig.suptitle("Per-complex R-precision · identical test cohort", fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.9, bottom=0.13, wspace=0.18)
    fig.colorbar(
        im,
        cax=fig.add_axes([0.92, 0.14, 0.015, 0.72]),
        format=PercentFormatter(1),
        label="R-precision",
    )
    figures.append(
        (
            fig,
            "four_model_per_complex",
            "Every test complex, ordered by total sequence length; all cells share a 0–100% color scale.",
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, model, label, color in zip(axes.flat, MODELS, LABELS, COLORS, strict=True):
        x = [lookup[model, "intra", t] for t in targets]
        y = [lookup[model, "inter", t] for t in targets]
        ax.scatter(
            x, y, s=52, alpha=0.8, color=color, edgecolors="white", linewidth=0.6
        )
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#aaaaaa")
        ax.set_title(label.replace("\n", " · "), fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.12)
    for ax in axes[-1]:
        ax.set_xlabel("Within-chain R-precision")
    for ax in axes[:, 0]:
        ax.set_ylabel("Inter-chain R-precision")
    fig.suptitle(
        "Does a well-predicted pair of chains imply a correct interface?", fontsize=15
    )
    fig.text(
        0.5,
        0.025,
        "One point per test complex. The dashed line marks equal intra- and inter-chain accuracy.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    figures.append(
        (
            fig,
            "four_model_intra_inter_scatter",
            "Paired within-chain and inter-chain accuracy for each full-complex prediction; common axes across all models.",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for ax, region, title in zip(
        axes,
        ["chain_A", "chain_B"],
        ["Original chain A", "Original chain B"],
        strict=True,
    ):
        matrix = np.array([[lookup[m, region, t] for m in MODELS] for t in targets])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), LABELS, fontsize=9)
        ax.set_yticks(range(17), [t.removesuffix("-assembly1") for t in targets])
        ax.set_title(title, pad=14)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:.0%}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color="black" if value > 0.65 else "white",
                fontsize=9,
            )
    fig.suptitle(
        "Accuracy within each partner, extracted from the same dimer predictions",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.08, right=0.89, top=0.9, bottom=0.13, wspace=0.18)
    fig.colorbar(
        im,
        cax=fig.add_axes([0.92, 0.14, 0.015, 0.72]),
        format=PercentFormatter(1),
        label="R-precision",
    )
    figures.append(
        (
            fig,
            "four_model_per_chain",
            "Per-partner R-precision using each chain’s own number of true contacts; chain labels follow the frozen input order.",
        )
    )

    for fig, stem, caption in figures:
        save_plot_with_meta(fig, HERE / "plots" / f"{stem}.png", caption=caption, dpi=180)
    with PdfPages(HERE / "plots/four_model_comparison.pdf") as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(
            0.07,
            0.9,
            "Contact prediction on matched protein complexes",
            fontsize=22,
            weight="bold",
        )
        lines = [
            "23 frozen FoldBench dimers: 6 development and 17 test targets. Figures show the 17 test targets.",
            "Each model predicts the whole dimer. Both contact classes use the same experimental residue mask.",
            "",
            f"{'Test mean R-precision':<41} {'Within chains':>14} {'Between chains':>17}",
        ]
        for model in MODELS:
            lines.append(
                f"{model:<41} {float(means[model, 'intra']['mean_r_precision']):>14.1%} {float(means[model, 'inter']['mean_r_precision']):>17.1%}"
            )
        lines.extend(
            [
                "",
                "Metric: top R pairs, where R is the native contact count in that class. Intra pools both partners.",
                "Ground truth: pyconfind native-only degree >= 0.001; separation >= 6 within a chain; no inter-chain exclusion.",
                "MarinFold ranking: consensus votes from 100 rollouts; T=1, top-p=0.95, full 8192-token context.",
                "Multichain: exp343 step 280154. Default: exp277 step 266344, A + 10G + B, linker omitted from scoring.",
                "ESMFold2: native chains, no MSA, 20 loops / 100 diffusion steps, best of 5 by 0.8 ipTM + 0.2 pTM.",
                "AlphaFold3: native chains, ColabFold paired/unpaired MSAs, no templates, 10 recycles, 1 seed x 5 samples.",
                "AF3 selects its own highest ranking_score. Structure contacts are ranked by pyconfind degree.",
                "Zero scores remain eligible; ties preserve canonical pair order. Random-tie expectations are in the CSV.",
                "Equivalent-sequence chains retain input order for every method; no ground-truth-based chain swapping.",
                "Uncertainty: 10,000 bootstrap samples of 13 homology groups. Same complexes in every comparison.",
                "AF3 has MSA information; this is a comparison of these specified pipelines, not a controlled MSA ablation.",
                "Baseline training-set decontamination has not been audited here. Pair holdout was audited for exp343.",
                "",
                "AF3: Abramson et al., Nature 630, 493–500 (2024). ESMFold2: Biohub, biohub/ESMFold2.",
                "Generated by: uv run python plot_four_model_eval.py",
            ]
        )
        fig.text(
            0.07,
            0.81,
            "\n".join(lines),
            fontsize=9.5,
            va="top",
            family="monospace",
            linespacing=1.65,
        )
        pdf.savefig(fig)
        plt.close(fig)
        for fig, stem, caption in figures:
            fig.text(
                0.01,
                0.002,
                "Generated by: uv run python plot_four_model_eval.py",
                fontsize=6,
                color="#555555",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(HERE / "plots/four_model_comparison.pdf")


if __name__ == "__main__":
    main()
