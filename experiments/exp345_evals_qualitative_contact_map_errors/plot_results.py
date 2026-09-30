"""Render contact-map case studies and diagnostic plots from saved inputs."""

import gzip
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze import HERE, INPUTS, unpack
from build_summary import save_plot_with_meta

CASES = [["7wz5_A", "8arl_A", "8bux_A", "8dmu_A"],
         ["7xz3_A", "7uk8_A", "8b6a_A", "7y8h_A"]]


def draw_map(ax, mask: np.ndarray, color: str, size: float = 3) -> None:
    """Draw both halves in common input-sequence coordinates."""
    i, j = np.where(mask)
    ax.scatter(np.r_[j+1, i+1], np.r_[i+1, j+1], s=size, c=color, marker="s", linewidths=0)


def main() -> None:
    """Create saved figures and their generating-command metadata."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    items = {p["truth"]["stem"]: p for p in json.loads(gzip.decompress(INPUTS.read_bytes()))["proteins"]}
    df = pd.read_csv(HERE / "data/per_protein.csv").set_index("stem")
    (HERE / "plots").mkdir(exist_ok=True)
    for page, cases in enumerate(CASES, 1):
        fig, axes = plt.subplots(4, 3, figsize=(12, 15), layout="constrained")
        for row, stem in enumerate(cases):
            item = items[stem]
            score, truth, i, j = unpack(item)
            r = int(truth[i, j].sum())
            chosen = np.argsort(-score[i, j], kind="mergesort")[:r]
            pred = np.zeros_like(truth)
            pred[i[chosen], j[chosen]] = True
            valid = np.zeros_like(truth)
            valid[i, j] = True
            truth &= valid
            for ax in axes[row]:
                ax.set_xlim(0, len(score)+1)
                ax.set_ylim(len(score)+1, 0)
                ax.set_aspect("equal")
                ax.set_xlabel("Residue j (1-based)")
                ax.set_ylabel("Residue i (1-based)")
                ax.set_facecolor("#fafaf8")
                missing = sorted(set(range(len(score))) - set(item["truth"]["resolved"]))
                for pos in missing:
                    ax.axhspan(pos+.5, pos+1.5, color="#dfe2e5", lw=0)
                    ax.axvspan(pos+.5, pos+1.5, color="#dfe2e5", lw=0)
            draw_map(axes[row, 0], pred, "#233d69")
            draw_map(axes[row, 1], truth, "#087e83")
            draw_map(axes[row, 2], truth & ~pred, "#ce81b6")
            draw_map(axes[row, 2], pred & ~truth, "#da612b")
            draw_map(axes[row, 2], pred & truth, "#087e83")
            p = df.loc[stem, "r_precision"]
            axes[row, 0].set_title(f"{stem} · predicted top-{r} · P@R {p:.1%}")
            axes[row, 1].set_title(f"True contacts · L={len(score)}")
            axes[row, 2].set_title("Teal: correct / orange: false / pink: missed")
        save_plot_with_meta(fig, HERE / f"plots/cases_{page}.png", caption="Selected exploratory cases: predicted top-R, truth, and errors. Both halves mirrored; gray bands are unresolved residues.")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    axes[0].hist(df.r_precision, bins=np.linspace(0, 1, 11), color="#233d69")
    axes[0].set(xlabel="Per-protein R-precision", ylabel="Proteins", title="Accuracy varies widely (97 proteins)")
    keys = ["r_precision", "one_to_one_1_precision", "one_to_one_2_precision", "one_to_one_5_precision"]
    axes[1].bar(range(4), [df[k].mean() for k in keys], color=["#233d69", "#087e83", "#087e83", "#8baead"])
    axes[1].set(xticks=range(4), xticklabels=["Exact", "±1", "±2", "±5"], ylim=(0, 1), ylabel="Matched fraction", title="Oracle one-to-one local correction")
    axes[1].text(.02, .98, "True positives fixed; FP matched to FN.\nDiagnostic upper bound, not model accuracy.", transform=axes[1].transAxes, va="top", fontsize=9)
    axes[2].scatter(df.true_seen_fraction, df.r_precision, color="#087e83", alpha=.7, s=24)
    axes[2].plot([0, 1], [0, 1], color="#abb0b5", ls="--")
    axes[2].set(xlim=(.5, 1.02), ylim=(0, 1), xlabel="True contacts seen in ≥1 of 100 rollouts", ylabel="R-precision", title="Seeing a contact does not rank it highly")
    save_plot_with_meta(fig, HERE / "plots/diagnostics.png", caption="All 97 eval-val proteins. Local correction uses a truth-informed maximum one-to-one matching, fixing exact matches first; it is an upper bound, not an improved predictor.")
    plt.close(fig)


if __name__ == "__main__":
    main()
