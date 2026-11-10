"""Summarize oracle, median and confidence-selected structural accuracy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_summary import save_plot_with_meta
from score import METRICS

HERE = Path(__file__).resolve().parent
MODELS = ("default", "mpnn2-lr5e-4")
LABELS = {"default": "Current default", "mpnn2-lr5e-4": "MPNN-only · 2 epochs · 5e-4"}
COLORS = {"default": "#2378b5", "mpnn2-lr5e-4": "#de762b"}
METRIC_LABELS = {"tm_score": "TM-score", "lddt": "All-atom lDDT", "gdt_ts": "GDT-TS"}
SELECTORS = ("oracle", "median", "top_pick", "mean")
SELECTOR_LABELS = {
    "oracle": "Oracle",
    "median": "Median",
    "top_pick": "Confidence pick",
    "mean": "Rollout mean",
}


def selections(frame: pd.DataFrame) -> list[dict]:
    """Choose once by confidence, but compute a separate oracle for each metric."""
    frame = frame.sort_values("rollout")
    pick = frame.sort_values(
        ["ranking_score", "rollout"], ascending=[False, True]
    ).iloc[0]
    rows = []
    for metric in METRICS:
        oracle = frame.sort_values([metric, "rollout"], ascending=[False, True]).iloc[0]
        for selector, value, index in (
            ("oracle", oracle[metric], int(oracle.rollout)),
            ("median", frame[metric].median(), -1),
            ("top_pick", pick[metric], int(pick.rollout)),
            ("mean", frame[metric].mean(), -1),
        ):
            rows.append(
                dict(
                    metric=metric,
                    selector=selector,
                    score=float(value),
                    selected_rollout=index,
                )
            )
    return rows


def interval(values: np.ndarray) -> tuple[float, float]:
    """Bootstrap proteins, with the same sampled indices for every comparison."""
    rng = np.random.default_rng(381)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def main() -> None:
    """Build small source tables and figures from the complete 200-structure pilot."""
    frame = pd.read_csv(HERE / "data/per_rollout_metrics.csv")
    targets = pd.read_csv(HERE / "data/targets.csv")
    assert len(frame) == 200 and set(frame.model) == set(MODELS)
    assert set(frame.stem) == set(targets.stem)
    selected = []
    for (model, stem), group in frame.groupby(["model", "stem"]):
        assert sorted(group.rollout) == list(range(10))
        for k in range(1, 11):
            selected.extend(
                dict(model=model, stem=stem, k=k, **row)
                for row in selections(group.loc[group.rollout < k])
            )
    selected = pd.DataFrame(selected)
    selected.to_csv(HERE / "data/per_protein_selection.csv", index=False)
    summary = []
    for (model, k, metric, selector), group in selected.groupby(
        ["model", "k", "metric", "selector"]
    ):
        values = group.sort_values("stem").score.to_numpy()
        low, high = interval(values)
        summary.append(
            dict(
                model=model,
                k=k,
                metric=metric,
                selector=selector,
                mean=values.mean(),
                ci_low=low,
                ci_high=high,
                n_proteins=len(values),
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(HERE / "data/summary.csv", index=False)
    delta_rows, regret_rows, confidence_rows, gain_rows = [], [], [], []
    final = selected.loc[selected.k == 10]
    for (metric, selector), group in final.groupby(["metric", "selector"]):
        paired = group.pivot(index="stem", columns="model", values="score").sort_index()
        deltas = (paired[MODELS[1]] - paired[MODELS[0]]).to_numpy()
        low, high = interval(deltas)
        delta_rows.append(
            dict(
                metric=metric,
                selector=selector,
                mpnn_minus_default=deltas.mean(),
                ci_low=low,
                ci_high=high,
                wins=int((deltas > 0).sum()),
                ties=int((deltas == 0).sum()),
                losses=int((deltas < 0).sum()),
            )
        )
    pd.DataFrame(delta_rows).to_csv(HERE / "data/paired_model_deltas.csv", index=False)
    for (model, metric), group in final.groupby(["model", "metric"]):
        pivot = group.pivot(
            index="stem", columns="selector", values="score"
        ).sort_index()
        for first, second in (
            ("top_pick", "mean"),
            ("top_pick", "median"),
            ("oracle", "mean"),
            ("oracle", "top_pick"),
        ):
            gains = (pivot[first] - pivot[second]).to_numpy()
            low, high = interval(gains)
            gain_rows.append(
                dict(
                    model=model,
                    metric=metric,
                    comparison=f"{first}_minus_{second}",
                    mean=gains.mean(),
                    ci_low=low,
                    ci_high=high,
                    wins=int((gains > 0).sum()),
                    ties=int((gains == 0).sum()),
                )
            )
        for stem, row in pivot.iterrows():
            regret_rows.append(
                dict(
                    model=model,
                    metric=metric,
                    stem=stem,
                    oracle_minus_pick=row.oracle - row.top_pick,
                    pick_minus_median=row.top_pick - row["median"],
                    pick_minus_mean=row.top_pick - row["mean"],
                )
            )
    pd.DataFrame(regret_rows).to_csv(HERE / "data/selection_regret.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(HERE / "data/selection_gains.csv", index=False)
    for (model, stem), group in frame.groupby(["model", "stem"]):
        for metric in METRICS:
            confidence_rows.append(
                dict(
                    model=model,
                    stem=stem,
                    metric=metric,
                    spearman_r=float(
                        spearmanr(group.ranking_score, group[metric]).statistic
                    ),
                )
            )
    pd.DataFrame(confidence_rows).to_csv(
        HERE / "data/within_protein_confidence.csv", index=False
    )
    strata = final.merge(
        targets[["stem", "is_viral"]], on="stem", validate="many_to_one"
    )
    strata.groupby(["model", "metric", "selector", "is_viral"]).agg(
        mean=("score", "mean"), n_proteins=("stem", "nunique")
    ).reset_index().to_csv(HERE / "data/viral_status_summary.csv", index=False)
    contacts = pd.read_csv(HERE / "data/contact_inputs.csv")
    contacts.groupby("model")[
        [
            "n_canonical_contacts",
            "n_unresolved_removed",
            "n_mapped_near_removed",
            "n_effective_contacts",
        ]
    ].agg(["sum", "mean", "min", "max"]).to_csv(HERE / "data/contact_count_summary.csv")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        for mi, model in enumerate(MODELS):
            values = (
                summary.loc[
                    (summary.k == 10)
                    & (summary.metric == metric)
                    & (summary.model == model)
                ]
                .set_index("selector")
                .loc[list(SELECTORS)]
            )
            x = np.arange(4) + (mi - 0.5) * 0.16
            ax.errorbar(
                x,
                values["mean"],
                yerr=[values["mean"] - values.ci_low, values.ci_high - values["mean"]],
                fmt="o",
                capsize=3,
                color=COLORS[model],
                label=LABELS[model],
            )
        ax.set(
            title=METRIC_LABELS[metric],
            xticks=np.arange(4),
            xticklabels=["Oracle", "Median", "Confidence\npick", "Rollout\nmean"],
            ylim=(0, 1),
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean accuracy across 10 proteins")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "One Helico structure per MarinFold rollout · 10 rollouts per model and protein"
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    save_plot_with_meta(
        fig,
        plots / "accuracy_summary.png",
        caption="Equal-protein means at k=10; 95% protein-bootstrap intervals (10,000 draws). Oracle uses reference labels; confidence pick uses only Helico ranking_score.",
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    styles = {"oracle": "-", "top_pick": "--", "median": ":"}
    for ax, metric in zip(axes, METRICS, strict=True):
        for model in MODELS:
            for selector, style in styles.items():
                curve = summary.loc[
                    (summary.model == model)
                    & (summary.metric == metric)
                    & (summary.selector == selector)
                ].sort_values("k")
                ax.plot(
                    curve.k,
                    curve["mean"],
                    style,
                    color=COLORS[model],
                    linewidth=2,
                    label=f"{LABELS[model]} · {SELECTOR_LABELS[selector]}",
                )
        ax.set(
            title=METRIC_LABELS[metric],
            xlabel="Number of rollouts, k",
            xlim=(1, 10),
            xticks=[1, 2, 4, 6, 8, 10],
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean accuracy across 10 proteins")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Observed prefix curves: first k matched rollouts")
    fig.tight_layout(rect=(0, 0.23, 1, 0.95))
    save_plot_with_meta(
        fig,
        plots / "accuracy_vs_rollouts.png",
        caption="Oracle, confidence-selected, and median accuracy for the first k saved rollouts. Lines describe one fixed sequence of ten samples, not expected curves over all possible rollout orders.",
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.4), sharey=True)
    for ax, metric in zip(axes, METRICS, strict=True):
        for ti, target in targets.iterrows():
            for mi, model in enumerate(MODELS):
                y = ti + (mi - 0.5) * 0.24
                values = frame.loc[(frame.model == model) & (frame.stem == target.stem)]
                pick = selections(values)
                pick = {
                    r["selector"]: r["score"] for r in pick if r["metric"] == metric
                }
                ax.scatter(
                    values[metric], np.full(10, y), s=11, color=COLORS[model], alpha=0.3
                )
                ax.plot(
                    [values[metric].min(), pick["oracle"]],
                    [y, y],
                    color=COLORS[model],
                    alpha=0.4,
                )
                ax.scatter(pick["median"], y, marker="|", s=120, color=COLORS[model])
                ax.scatter(
                    pick["oracle"],
                    y,
                    marker="o",
                    s=43,
                    facecolor="white",
                    edgecolor=COLORS[model],
                )
                ax.scatter(
                    pick["top_pick"], y, marker="*", s=80, color=COLORS[model], zorder=5
                )
        ax.set(xlabel=METRIC_LABELS[metric], xlim=(0, 1), yticks=np.arange(10))
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticklabels(
        [f"{t.stem}  (Neff {t.msa_neff:.2f})" for _, t in targets.iterrows()]
    )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Per-protein variation · pale dots = rollouts · | = median · ○ = oracle · ★ = confidence pick"
    )
    fig.text(
        0.5,
        0.02,
        "Blue: current default     Orange: MPNN-only, 2 epochs, 5e-4",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_plot_with_meta(
        fig,
        plots / "per_protein_accuracy.png",
        caption="Every individual prediction, with median, oracle, and confidence-selected accuracy. Targets are ordered by Neff. Each structure contributes the same three metrics.",
    )
    plt.close(fig)
    print(
        summary.loc[summary.k == 10]
        .pivot(index=["model", "selector"], columns="metric", values="mean")
        .round(4)
        .to_string()
    )
    print(pd.DataFrame(delta_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
