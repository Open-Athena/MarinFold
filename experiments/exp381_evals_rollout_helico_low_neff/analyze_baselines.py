"""Matched-budget five-arm comparisons using the frozen exp381 selectors."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import METRIC_LABELS, SELECTORS, interval, selections
from build_summary import save_plot_with_meta
from score import METRICS

HERE = Path(__file__).resolve().parent
MODELS = ("default", "mpnn2-lr5e-4", "esmfold2", "af3_single", "af3_msa")
LABELS = {
    "default": "Default → Helico",
    "mpnn2-lr5e-4": "MPNN 2ep → Helico",
    "esmfold2": "ESMFold2",
    "af3_single": "AF3 single sequence",
    "af3_msa": "AF3 + MSA",
}
COLORS = dict(
    zip(MODELS, ("#2378b5", "#de762b", "#299860", "#9367b6", "#c44460"), strict=True)
)
SLABELS = {
    "oracle": "Oracle",
    "median": "Median",
    "top_pick": "Confidence pick",
    "mean": "Sample mean",
}


def main() -> None:
    """Compute all prefixes, protein-level uncertainty, and final matched plots."""
    baseline = pd.read_csv(HERE / "data/baseline_metrics.csv")
    frame = pd.concat(
        [pd.read_csv(HERE / "data/per_rollout_metrics.csv"), baseline],
        ignore_index=True,
    )
    if len(frame) != 500 or set(frame.model) != set(MODELS):
        raise ValueError("Require all 500 predictions")
    selected = []
    for (model, stem), group in frame.groupby(["model", "stem"]):
        if sorted(group.rollout) != list(range(10)):
            raise ValueError("Incomplete matched sample budget")
        for k in range(1, 11):
            selected.extend(
                dict(model=model, stem=stem, k=k, **row)
                for row in selections(group[group.rollout < k])
            )
    selected = pd.DataFrame(selected)
    selected.to_csv(HERE / "data/matched_per_protein_selection.csv", index=False)
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
    summary.to_csv(HERE / "data/matched_summary.csv", index=False)
    final = selected[selected.k == 10]
    deltas = []
    for metric in METRICS:
        for selector in SELECTORS:
            pivot = final[
                (final.metric == metric) & (final.selector == selector)
            ].pivot(index="stem", columns="model", values="score")
            for model in MODELS[1:]:
                values = (pivot[model] - pivot["default"]).to_numpy()
                lo, hi = interval(values)
                deltas.append(
                    dict(
                        model=model,
                        reference="default",
                        metric=metric,
                        selector=selector,
                        delta=values.mean(),
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
            values = (pivot.af3_msa - pivot.af3_single).to_numpy()
            lo, hi = interval(values)
            deltas.append(
                dict(
                    model="af3_msa",
                    reference="af3_single",
                    metric=metric,
                    selector=selector,
                    delta=values.mean(),
                    ci_low=lo,
                    ci_high=hi,
                )
            )
    pd.DataFrame(deltas).to_csv(HERE / "data/matched_paired_deltas.csv", index=False)
    regrets = []
    for model in MODELS:
        for metric in METRICS:
            p = final[(final.model == model) & (final.metric == metric)].pivot(
                index="stem", columns="selector", values="score"
            )
            for comparator in ("oracle", "mean", "median"):
                values = (p.top_pick - p[comparator]).to_numpy()
                lo, hi = interval(values)
                regrets.append(
                    dict(
                        model=model,
                        metric=metric,
                        comparison=f"top_pick_minus_{comparator}",
                        delta=values.mean(),
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    pd.DataFrame(regrets).to_csv(HERE / "data/matched_selection_gains.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    y = np.arange(len(MODELS))
    markers = {"oracle": "D", "median": "s", "top_pick": "o", "mean": "x"}
    offsets = dict(zip(SELECTORS, (-0.24, -0.08, 0.08, 0.24), strict=True))
    for ax, metric in zip(axes, METRICS, strict=True):
        for selector in SELECTORS:
            for i, model in enumerate(MODELS):
                row = summary[
                    (summary.model == model)
                    & (summary.k == 10)
                    & (summary.metric == metric)
                    & (summary.selector == selector)
                ].iloc[0]
                ax.errorbar(
                    row["mean"],
                    i + offsets[selector],
                    xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                    fmt=markers[selector],
                    color=COLORS[model],
                    ms=6,
                    elinewidth=0.9,
                    capsize=2,
                    label=SLABELS[selector] if i == 0 else None,
                )
        ax.set(
            title=METRIC_LABELS[metric], xlim=(0, 1), xlabel="Mean across 10 proteins"
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, [LABELS[m] for m in MODELS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Matched ten-sample accuracy · 95% protein bootstrap intervals", fontsize=14
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save_plot_with_meta(
        fig,
        HERE / "plots/matched_accuracy_summary.png",
        caption="All five arms use the same ten proteins and ten samples each. Oracle is metric-specific; confidence selects one structure. Whiskers bootstrap proteins, not rollouts.",
    )
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for i, metric in enumerate(METRICS):
        for j, selector in enumerate(SELECTORS):
            ax = axes[i, j]
            for model in MODELS:
                rows = summary[
                    (summary.model == model)
                    & (summary.metric == metric)
                    & (summary.selector == selector)
                ].sort_values("k")
                ax.plot(
                    rows.k,
                    rows["mean"],
                    marker=".",
                    color=COLORS[model],
                    label=LABELS[model],
                )
            ax.set(xlim=(1, 10), ylim=(0, 1), xticks=[1, 3, 5, 7, 10])
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(SLABELS[selector])
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if i == 2:
                ax.set_xlabel("Number of samples k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10)
    fig.suptitle("Accuracy as the sample budget increases · fixed observed prefixes")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_plot_with_meta(
        fig,
        HERE / "plots/matched_accuracy_vs_samples.png",
        caption="Macro averages of all fixed prefixes k=1..10. Only oracle is monotonic; confidence pick may worsen as more samples arrive. Sample mean is the per-protein individual-sample average.",
    )
    plt.close(fig)
    stems = pd.read_csv(HERE / "data/targets.csv").stem.tolist()
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharey=True)
    for ax, stem in zip(axes.flat, stems, strict=True):
        for i, model in enumerate(MODELS):
            rows = frame[(frame.stem == stem) & (frame.model == model)].sort_values(
                "rollout"
            )
            ax.scatter(
                i + np.linspace(-0.15, 0.15, 10),
                rows.tm_score,
                color=COLORS[model],
                alpha=0.5,
                s=22,
            )
            pick = final[
                (final.stem == stem)
                & (final.model == model)
                & (final.metric == "tm_score")
                & (final.selector == "top_pick")
            ].iloc[0]
            ax.scatter(
                i,
                pick.score,
                marker="*",
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.7,
                s=140,
                zorder=5,
            )
        ax.set(
            title=stem,
            ylim=(0, 1),
            xticks=range(5),
            xticklabels=["Default", "MPNN", "ESM2", "AF3 SS", "AF3 MSA"],
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("TM-score")
    axes[1, 0].set_ylabel("TM-score")
    fig.suptitle("Every predicted structure · star = native-confidence selection")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        HERE / "plots/matched_per_protein_tm.png",
        caption="Ten samples per protein per arm; star identifies the single confidence-selected structure. Targets appear in increasing original Neff order.",
    )
    plt.close(fig)
    report = dict(
        n_predictions=500,
        n_baseline_predictions=300,
        min_baseline_ca_coverage=float(baseline.ca_coverage.min()),
        min_baseline_atom_coverage=float(baseline.atom_coverage.min()),
        counts=frame.groupby("model").size().to_dict(),
        baseline_inference_seconds=baseline.groupby("model")
        .elapsed_seconds.sum()
        .to_dict(),
        n_distinct_cif_per_model=baseline.groupby("model")
        .prediction_sha256.nunique()
        .to_dict(),
    )
    (HERE / "data/baseline_run_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        summary[summary.k == 10]
        .pivot(index=["model", "selector"], columns="metric", values="mean")
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
