#!/usr/bin/env python
"""Plot paired blind-search results from the committed small CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"


def main() -> None:
    per = pd.read_csv(DATA / "per_protein.csv")
    sub = per[per.primary & (per.split == "test")]
    paired = sub[sub.method.isin(["iid", "branch10"])].pivot(
        index="pair_id", columns="method", values="minority_enrichment"
    ).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(paired.iid, paired.branch10, c="#306b9b", alpha=0.8, s=45)
    low = min(0, paired.min().min()) - 0.05
    high = max(0, paired.max().max()) + 0.05
    ax.plot([low, high], [low, high], color="0.45", linestyle="--", linewidth=1)
    ax.set(xlim=(low, high), ylim=(low, high),
           xlabel="Independent sampling: best minority enrichment",
           ylabel="10-contact branching: best minority enrichment",
           title=f"Blind top-16 shortlists, {len(paired)} held-out pairs")
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "paired_enrichment.png", dpi=180,
        caption="Each point is one held-out primary fold-switching pair. Both methods use "
                "the same 100 root rollouts plus 100 arm rollouts; the diagonal is parity.",
    )
    plt.close(fig)

    summary = pd.read_csv(DATA / "summary.csv")
    methods = ["iid", "temp", "random", "branch5", "branch10", "branch20"]
    tab = summary.set_index("method").loc[methods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    locations = range(len(tab))
    ax.bar(locations, tab.dual_contact_hits, color="#306b9b", label="blind top 16")
    ax.scatter(locations, tab.oracle_dual_hits, color="#b25a2d", marker="D",
               zorder=3, label="oracle best of all 200")
    ax.set_xticks(list(locations), methods)
    ax.set(ylabel="Held-out pairs with both contact-map modes",
           title="Blind discovery versus pool headroom")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "dual_contact_hits.png", dpi=180,
        caption="Bars count pairs meeting the prespecified contact-level dual-hit criterion "
                "among 16 blindly selected maps. Diamonds show the oracle ceiling in the "
                "same candidate pool, which cannot serve as an inference method.",
    )
    plt.close(fig)

    budget = pd.read_csv(DATA / "budget_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(budget.budget, budget.blind_dual_hits, marker="o", linewidth=2,
            color="#306b9b", label="blind top 16")
    ax.plot(budget.budget, budget.oracle_dual_hits, marker="D", linewidth=2,
            color="#b25a2d", label="oracle best of pool")
    ax.set(xlabel="Independent rollouts per protein", ylabel="Held-out dual-contact pairs",
           xticks=budget.budget.tolist(), ylim=(0, max(budget.oracle_dual_hits) + 1),
           title="Plain-sampling budget curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "budget_curve.png", dpi=180,
        caption="A separate 500-rollout run, evaluated at its first 100, 200, and 500 "
                "candidates. The blind selector returns 16 maps at each budget; the oracle "
                "line is diagnostic and cannot be used at inference time.",
    )
    plt.close(fig)

    structural = pd.read_csv(DATA / "helico_cross_reference.csv")
    case = structural[structural.pair_id == "3j7wb_3j7vg"]
    variants = ["true_fold1", "blind_fold1", "iid_fold1",
                "iid_fold2", "blind_fold2", "true_fold2"]
    labels = ["true\nFold1", "branch\nFold1", "iid\nFold1",
              "iid\nFold2", "branch\nFold2", "true\nFold2"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for fold, offset, color in [(1, -0.18, "#306b9b"), (2, 0.18, "#b25a2d")]:
        means = [case[(case.variant == variant) & (case.reference_fold == fold)]
                 .lddt_region_touch.mean() for variant in variants]
        ax.bar([index + offset for index in range(len(variants))], means, width=0.34,
               color=color, label=f"score vs Fold{fold}")
        for index, variant in enumerate(variants):
            values = case[(case.variant == variant) & (case.reference_fold == fold)]
            ax.scatter([index + offset] * len(values), values.lddt_region_touch,
                       color="black", s=11, zorder=3)
    ax.set_xticks(range(len(variants)), labels)
    ax.set(ylabel="Switching-region contact-neighborhood lDDT", ylim=(0, 0.85),
           title="3j7w/3j7v: both blind selectors recover two local folds")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "helico_3j7_structural_check.png", dpi=180,
        caption="Helico structures from two reference input chains; black points are the "
                "two inputs. Both 200-rollout shortlist methods yield fold-specific "
                "switching-region structures on this protein. True-contact controls show "
                "the attainable separation for this reconstruction model.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
