"""Compare four ten-structure arms using paired protein-level uncertainty."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import (
    LABELS,
    METRIC_LABELS,
    MODELS,
    SELECTORS,
    SELECTOR_LABELS,
    interval,
    selections,
)
from build_summary import save_plot_with_meta
from score import METRICS

HERE = Path(__file__).resolve().parent
ARMS = ("first10", "random10", "medoids10", "maxmin10")
ALABELS = {
    "first10": "Original first 10",
    "random10": "Random 10 / 1,000",
    "medoids10": "10 medoids / 1,000",
    "maxmin10": "10 farthest / 1,000",
}
COLORS = {
    "first10": "#9b9b9b",
    "random10": "#2475b3",
    "medoids10": "#d57827",
    "maxmin10": "#378a59",
}


def main() -> None:
    """Produce frozen-endpoint summaries, paired differences, and figures."""
    first = pd.read_csv(HERE / "data/per_rollout_metrics.csv").assign(arm="first10")
    frame = pd.concat(
        [
            first,
            pd.read_csv(HERE / "data/diversity_metrics.csv"),
            pd.read_csv(HERE / "data/maxmin_metrics.csv"),
        ],
        ignore_index=True,
    )
    assert len(frame) == 800
    rows = []
    for (model, stem, arm), group in frame.groupby(["model", "stem", "arm"]):
        assert sorted(group.rollout) == list(range(10))
        rows.extend(
            dict(model=model, stem=stem, arm=arm, **row) for row in selections(group)
        )
    selected = pd.DataFrame(rows)
    selected.to_csv(HERE / "data/diversity_per_protein_selection.csv", index=False)
    summary, deltas = [], []
    for (model, arm, metric, selector), group in selected.groupby(
        ["model", "arm", "metric", "selector"]
    ):
        values = group.sort_values("stem").score.to_numpy()
        low, high = interval(values)
        summary.append(
            dict(
                model=model,
                arm=arm,
                metric=metric,
                selector=selector,
                mean=values.mean(),
                ci_low=low,
                ci_high=high,
                n_proteins=len(values),
            )
        )
    for (model, metric, selector), group in selected.groupby(
        ["model", "metric", "selector"]
    ):
        pivot = group.pivot(index="stem", columns="arm", values="score").sort_index()
        for arm in ("medoids10", "maxmin10"):
            for reference in ("random10", "first10"):
                values = (pivot[arm] - pivot[reference]).to_numpy()
                low, high = interval(values)
                deltas.append(
                    dict(
                        model=model,
                        arm=arm,
                        metric=metric,
                        selector=selector,
                        reference=reference,
                        delta=values.mean(),
                        ci_low=low,
                        ci_high=high,
                        wins=int((values > 0).sum()),
                        ties=int((values == 0).sum()),
                        losses=int((values < 0).sum()),
                    )
                )
    summary, deltas = pd.DataFrame(summary), pd.DataFrame(deltas)
    summary.to_csv(HERE / "data/diversity_summary.csv", index=False)
    deltas.to_csv(HERE / "data/diversity_paired_deltas.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for i, model in enumerate(MODELS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for a, arm in enumerate(ARMS):
                group = (
                    summary[
                        (summary.model == model)
                        & (summary.metric == metric)
                        & (summary.arm == arm)
                    ]
                    .set_index("selector")
                    .loc[list(SELECTORS)]
                )
                ax.plot(
                    np.arange(4) + (a - 1.5) * 0.11,
                    group["mean"],
                    "o-",
                    color=COLORS[arm],
                    label=ALABELS[arm],
                    ms=5,
                )
            ax.set(
                xticks=range(4),
                xticklabels=[SELECTOR_LABELS[s] for s in SELECTORS],
                ylim=(0, 1),
                title=METRIC_LABELS[metric] if i == 0 else None,
            )
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel(LABELS[model] + "\nMean across ten proteins")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Selecting ten of 1,000 contacts: coverage versus pairwise diversity")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_plot_with_meta(
        fig,
        HERE / "plots/diversity_accuracy.png",
        caption="All arms fold ten contact sets per protein with matched per-slot Helico seeds. Mean across the same ten lowest-Neff eval-val proteins. Medoids, farthest points and random controls use a 1,000-rollout pool per checkpoint; original first ten are reused.",
    )
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for i, model in enumerate(MODELS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for a, arm in enumerate(("medoids10", "maxmin10")):
                group = (
                    deltas[
                        (deltas.model == model)
                        & (deltas.metric == metric)
                        & (deltas.reference == "random10")
                        & (deltas.arm == arm)
                    ]
                    .set_index("selector")
                    .loc[list(SELECTORS)]
                )
                means = group.delta.to_numpy()
                ax.errorbar(
                    means,
                    np.arange(4) + (a - 0.5) * 0.24,
                    xerr=[
                        means - group.ci_low.to_numpy(),
                        group.ci_high.to_numpy() - means,
                    ],
                    fmt="o",
                    capsize=4,
                    color=COLORS[arm],
                    label=ALABELS[arm],
                )
            ax.axvline(0, color="black", lw=0.8)
            ax.set(
                yticks=range(4),
                yticklabels=[SELECTOR_LABELS[s] for s in SELECTORS],
                title=METRIC_LABELS[metric] if i == 0 else None,
            )
            if i == 1:
                ax.set_xlabel("Selected arm minus random")
            if j == 0:
                ax.set_ylabel(LABELS[model])
            ax.grid(axis="x", alpha=0.2)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Paired accuracy changes · 95% protein-bootstrap intervals")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_plot_with_meta(
        fig,
        HERE / "plots/diversity_paired_changes.png",
        caption="10,000 paired bootstrap resamples of ten proteins (seed381). Medoid confidence-selected TM-score is the original primary endpoint; farthest-point comparisons are secondary. Intervals describe protein variation conditional on one fixed random subset and the selected inference seeds.",
    )
    plt.close(fig)
    diagnostics = pd.concat(
        [
            pd.read_csv(HERE / "data/diversity_contact_diagnostics.csv"),
            pd.read_csv(HERE / "data/maxmin_contact_diagnostics.csv"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, model in enumerate(MODELS):
        for j, (column, title) in enumerate(
            [
                ("pairwise_jaccard", "Pairwise contact diversity ↑"),
                ("mean_nearest_distance", "Pool-to-nearest distance ↓"),
                ("mean_contacts", "Contacts per selected rollout"),
            ]
        ):
            ax = axes[i, j]
            group = diagnostics[diagnostics.model == model].pivot(
                index="stem", columns="arm", values=column
            )
            for _, row in group.iterrows():
                ax.plot(range(4), row[list(ARMS)], color="#888888", alpha=0.4, lw=0.8)
            for a, arm in enumerate(ARMS):
                ax.scatter(np.full(10, a), group[arm], color=COLORS[arm], s=20)
                ax.scatter(
                    a, group[arm].mean(), marker="D", color="black", s=45, zorder=5
                )
            ax.set(
                xticks=range(4),
                xticklabels=["First 10", "Random 10", "Medoids", "Farthest"],
                title=title if i == 0 else None,
            )
            if j == 0:
                ax.set_ylabel(LABELS[model])
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "What the contact selector changes · each line is one protein; diamond is mean"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_plot_with_meta(
        fig,
        HERE / "plots/diversity_contact_geometry.png",
        caption="Jaccard distance on the exact contact sets supplied to Helico. Medoids optimize mean nearest-representative distance, rather than directly maximizing pairwise separation. All 1,000 draws retain their sampling weight, including duplicates.",
    )
    plt.close(fig)
    print(
        summary.pivot(
            index=["model", "arm", "selector"], columns="metric", values="mean"
        )
        .round(4)
        .to_string()
    )
    print(
        deltas[(deltas.metric == "tm_score") & (deltas.selector == "top_pick")]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
