"""Plot completed MarinFold arms while native structure baselines are running."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Show means, group-bootstrap intervals, and all matched test complexes."""
    data = HERE / "data/four_model_v1"
    rows = list(csv.DictReader((data / "per_target.csv").open()))
    summary = list(csv.DictReader((data / "summary.csv").open()))
    models = ["MarinFold multichain", "MarinFold default + 10G"]
    labels = [
        "Multichain\nexp343 · step 280154",
        "Default + 10 glycines\nexp277 · step 266344",
    ]
    colors = ["#31688e", "#d07a29"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.22, wspace=0.3)
    fig.suptitle(
        "MarinFold predicts contacts within chains much better than between them",
        fontsize=15,
        x=0.08,
        ha="left",
        y=0.97,
    )
    fig.text(
        0.08,
        0.895,
        "Same 17 held-out test complexes · 100 rollouts per model · native resolved residues only",
        fontsize=11,
        color="#444444",
    )
    for ax, region, title in zip(
        axes,
        ["intra", "inter"],
        ["Within the original chains", "Between the original chains"],
        strict=True,
    ):
        means = []
        for index, (model, color) in enumerate(zip(models, colors, strict=True)):
            values = sorted(
                (
                    r
                    for r in rows
                    if r["split"] == "test"
                    and r["region"] == region
                    and r["model"] == model
                ),
                key=lambda r: r["stem"],
            )
            assert len(values) == 17
            s = next(
                r
                for r in summary
                if r["split"] == "test"
                and r["region"] == region
                and r["model"] == model
            )
            mean, low, high = (
                float(s[k]) * 100 for k in ["mean_r_precision", "ci_low", "ci_high"]
            )
            means.append(mean)
            ax.bar(index, mean, color=color, alpha=0.25, width=0.6, zorder=2)
            jitter = np.random.default_rng(350).uniform(-0.2, 0.2, len(values))
            ax.scatter(
                index + jitter,
                [float(r["r_precision"]) * 100 for r in values],
                s=24,
                color=color,
                alpha=0.65,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
            ax.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color=color,
                capsize=6,
                elinewidth=2.3,
                markersize=7,
                zorder=5,
            )
            ax.annotate(
                f"{mean:.1f}%",
                (index, high),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                weight="bold",
                color=color,
                fontsize=13,
            )
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_xticks([0, 1], labels, fontsize=10)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylabel("R-precision (%)")
        ax.set_ylim(0, 100 if region == "intra" else 20)
        if region == "inter":
            ax.set_yticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.08,
        0.08,
        "Dots: individual complexes. Diamonds: mean. Whiskers: 95% bootstrap interval over 13 homology groups.",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.08,
        0.045,
        "Different y-axis scales: the inter-chain panel is enlarged 5×. See four_model_comparison.pdf for all four models.",
        fontsize=9,
        color="#444444",
    )
    out = HERE / "plots/four_model_progress.png"
    fig.savefig(out, dpi=190, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    out.with_name(out.name + ".meta.json").write_text(
        json.dumps(
            {
                "script": "plot_four_model_progress.py",
                "args": [],
                "caption": "Interim comparison of the two completed MarinFold arms on the same 17 complex test targets; intra-chain scores use both original partners. 95% homology-group bootstrap intervals. Inter-chain axis is enlarged 5x.",
            },
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
