"""Render the experiment figures entirely from saved small source tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
COLORS = {"default": "#245c91", "mpnn": "#d35b39"}
LABELS = {"default": "Full-data default", "mpnn": "ProteinMPNN-only"}
TIERS = ["<10", "10–<100", "100–<1000", "≥1000"]


def read(name: str) -> pd.DataFrame:
    """Read a committed analysis table."""
    return pd.read_csv(DATA / name)


def save(fig: plt.Figure, name: str, caption: str) -> None:
    """Save a standalone figure and the summary PDF's generation sidecar."""
    fig.canvas.draw()
    fig.tight_layout(pad=1.5)
    fig.canvas.draw()
    save_plot_with_meta(fig, PLOTS / (name + ".png"), caption=caption)
    plt.close(fig)


def errorbar(ax: plt.Axes, x: np.ndarray, frame: pd.DataFrame, model: str) -> None:
    """Plot means and 95% protein-bootstrap intervals."""
    y = frame["mean"].to_numpy()
    err = np.array([y - frame.ci_low.to_numpy(), frame.ci_high.to_numpy() - y])
    ax.errorbar(
        x,
        y,
        yerr=err,
        color=COLORS[model],
        marker="o",
        capsize=3,
        label=LABELS[model],
        lw=1.8,
    )


def extended_curves() -> None:
    """Show 1,000-sample discovery and oracle F1, keeping cohorts separate."""
    curves = read("curve_summary_1000.csv")
    oracle = read("oracle_summary_1000.csv")
    for cohort, suffix, title in [
        ("eval-val", "", "97 natural proteins"),
        ("eval-denovo", "_designs", "19 designed proteins · secondary"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(title, fontsize=14)
        for row, region in enumerate(["all", "long"]):
            for col, (endpoint, label) in enumerate(
                [
                    ("r_precision", "Ensemble R-precision"),
                    ("true_coverage", "True-contact discovery / truth"),
                    ("false_discovered", "Distinct false contacts discovered"),
                ]
            ):
                ax = axes[row, col]
                for model in COLORS:
                    selected = curves[
                        curves.eval_set.eq(cohort)
                        & curves["range"].eq(region)
                        & curves.endpoint.eq(endpoint)
                        & curves.model.eq(model)
                    ].sort_values("k")
                    errorbar(ax, selected.k.to_numpy(), selected, model)
                ax.axvline(300, color=".65", lw=0.8, ls=":")
                ax.set(
                    xscale="log",
                    xlabel="Rollouts (k)",
                    ylabel=label,
                    title=f"{region.capitalize()} range",
                    xlim=(0.9, 1200),
                )
                ax.set_xticks([1, 10, 100, 300, 1000], [1, 10, 100, 300, 1000])
                if col == 0:
                    ax.legend(fontsize=9)
        save(
            fig,
            "ensemble_discovery" + suffix,
            "T=1, fixed prompts, extended to 1,000 samples. Dotted line: original 300-sample limit. Error bars: pointwise 95% protein-bootstrap intervals; cohorts remain separate.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            title + " · oracle selection versus individual-rollout mean", fontsize=13
        )
        for ax, region in zip(axes, ["all", "long"], strict=True):
            for model in COLORS:
                selected = oracle[
                    oracle.eval_set.eq(cohort)
                    & oracle["range"].eq(region)
                    & oracle.model.eq(model)
                ]
                best = selected[selected.endpoint.eq("oracle_f1")].sort_values("k")
                baseline = selected[selected.endpoint.eq("mean_f1")].iloc[0]
                ax.plot(
                    best.k,
                    best["mean"],
                    color=COLORS[model],
                    lw=2,
                    label=LABELS[model] + " · oracle best@k",
                )
                ax.fill_between(
                    best.k,
                    best.ci_low,
                    best.ci_high,
                    color=COLORS[model],
                    alpha=0.14,
                    lw=0,
                )
                ax.axhline(
                    baseline["mean"],
                    color=COLORS[model],
                    ls="--",
                    lw=1.6,
                    label=LABELS[model] + " · mean individual F1",
                )
                ax.axhspan(
                    baseline.ci_low,
                    baseline.ci_high,
                    color=COLORS[model],
                    alpha=0.06,
                    lw=0,
                )
            ax.axvline(300, color=".65", lw=0.8, ls=":")
            ax.set(
                xscale="log",
                xlim=(1, 1000),
                ylim=(0, 1),
                xlabel="Number of rollouts (k)",
                ylabel="F1 (mean over proteins)",
                title=f"{region.capitalize()} range",
            )
            ax.set_xticks([1, 10, 100, 300, 1000], [1, 10, 100, 300, 1000])
            ax.legend(fontsize=8, loc="lower right")
        save(
            fig,
            "oracle_f1" + suffix,
            "Oracle: best ground-truth F1 among the first k samples. Dashed: mean individual F1 over all 1,000. Bands: pointwise 95% protein CIs. Oracle selection needs truth and is not an inference-time selector.",
        )


def main() -> None:
    """Regenerate figures without inference or bootstrap recomputation."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "axes.titlesize": 12,
        }
    )
    meta = read("metadata.csv").set_index("stem")
    natural = set(meta[meta.eval_set.eq("eval-val")].index)
    div = read("diversity.csv")
    div = div[div.block.eq("pooled")]
    controls = read("count_controls.csv")
    contrasts = read("diversity_contrasts.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, region in enumerate(["all", "long"]):
        for col, (source, frame) in enumerate(
            [("diversity", div), ("count_controls", controls)]
        ):
            selected = frame[
                frame.stem.isin(natural)
                & frame.temperature.eq(1)
                & frame["mode"].eq("fixed")
                & frame["range"].eq(region)
            ]
            pair = selected.pivot(index="stem", columns="model", values="jaccard")
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color=".6", ls="--", lw=1)
            color = np.log10(meta.loc[pair.index, "msa_neff"])
            dots = ax.scatter(
                pair.default, pair.mpnn, c=color, cmap="viridis", s=27, alpha=0.8
            )
            result = contrasts[
                contrasts.source.eq(source)
                & contrasts.endpoint.eq("jaccard")
                & contrasts.eval_set.eq("eval-val")
                & contrasts.temperature.eq(1)
                & contrasts["mode"].eq("fixed")
                & contrasts["range"].eq(region)
            ].iloc[0]
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Default Jaccard distance",
                ylabel="MPNN-only Jaccard distance",
                title=f"{region.capitalize()} range · {'original counts' if col == 0 else 'matched counts'}\nΔ {result['mean']:+.3f} [{result.ci_low:+.3f}, {result.ci_high:+.3f}]",
            )
            fig.colorbar(dots, ax=ax, label="log₁₀ Neff", shrink=0.8)
    save(
        fig,
        "diversity",
        "97 natural proteins, T=1, 300 fixed-prompt samples. Points are proteins; Δ is MPNN-only minus default (paired 95% CI). Count control thins paired samples to equal contact counts.",
    )
    extended_curves()
    singles = read("single_rollout_summary.csv")
    metrics = read("accuracy_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for model in COLORS:
        points = []
        for temp in [0.7, 1, 1.3]:
            selected = div[
                div.stem.isin(natural)
                & div.temperature.eq(temp)
                & div["mode"].eq("fixed")
                & div["range"].eq("all")
                & div.model.eq(model)
            ]
            precision = singles[
                singles.eval_set.eq("eval-val")
                & singles.temperature.eq(temp)
                & singles["mode"].eq("fixed")
                & singles.model.eq(model)
            ].precision.mean()
            ensemble = metrics[
                metrics.eval_set.eq("eval-val")
                & metrics.temperature.eq(temp)
                & metrics["mode"].eq("fixed")
                & metrics["range"].eq("all")
                & metrics.model.eq(model)
            ]["mean"].iloc[0]
            points.append(
                [
                    temp,
                    selected.jaccard.mean(),
                    precision,
                    ensemble,
                    selected.mean_contacts.mean(),
                ]
            )
        p = np.array(points)
        for ax, y, label in [
            (axes[0], p[:, 2], "Single-rollout precision"),
            (axes[1], p[:, 3], "Ensemble R-precision (300)"),
            (axes[2], p[:, 4], "Contacts per rollout"),
        ]:
            ax.plot(p[:, 1], y, "o-", color=COLORS[model], label=LABELS[model])
            for x, yy, t in zip(p[:, 1], y, p[:, 0], strict=True):
                ax.annotate(
                    f"T={t:g}",
                    (x, yy),
                    xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set(xlabel="Mean Jaccard distance", ylabel=label)
    axes[0].legend(fontsize=9)
    save(
        fig,
        "temperature_tradeoff",
        "Natural proteins, fixed prompts. Means at temperatures 0.7, 1.0 and 1.3 show the tradeoff between diversity, precision, ensemble accuracy and output size. Descriptive comparison; no temperature was selected.",
    )
    tiers = read("tier_accuracy.csv")
    all_metrics = read("metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, region in enumerate(["all", "long"]):
        ax = axes[0, col]
        for model in COLORS:
            selected = (
                tiers[
                    tiers.depth_variable.eq("msa_neff")
                    & tiers["mode"].eq("randomized")
                    & tiers["range"].eq(region)
                    & tiers.model.eq(model)
                ]
                .set_index("tier")
                .reindex(TIERS)
            )
            errorbar(
                ax, np.arange(4) + (0 if model == "default" else 0.05), selected, model
            )
        counts = meta.loc[list(natural)].neff_tier.value_counts()
        ax.set(
            xticks=range(4),
            xticklabels=[f"{t}\nn={counts.get(t, 0)}" for t in TIERS],
            xlabel="Neff tier (80% identity)",
            ylabel="R-precision",
            title=f"{region.capitalize()} range · 100 randomized prompts",
            ylim=(0, 1),
        )
        ax.legend(fontsize=9)
        selected = all_metrics[
            all_metrics.stem.isin(natural)
            & all_metrics.temperature.eq(1)
            & all_metrics["mode"].eq("randomized")
            & all_metrics["range"].eq(region)
            & all_metrics.cut.eq("R")
        ]
        pair = selected.pivot(index="stem", columns="model", values="precision").join(
            meta
        )
        ax = axes[1, col]
        for viral, marker in [(0, "o"), (1, "^")]:
            part = pair[pair.is_viral.eq(viral)]
            ax.scatter(
                np.log10(part.msa_neff),
                part.mpnn - part.default,
                s=28,
                marker=marker,
                alpha=0.7,
                label="Viral" if viral else "Nonviral",
            )
        x = np.log10(pair.msa_neff.to_numpy())
        y = (pair.mpnn - pair.default).to_numpy()
        coefs = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, np.polyval(coefs, xx), color="black", lw=1.3)
        ax.axhline(0, color=".5", ls="--")
        ax.set(
            xlabel="log₁₀ Neff",
            ylabel="MPNN-only − default R-precision",
            title=f"Paired slope: {coefs[0]:+.3f} per Neff decade",
        )
        ax.legend(fontsize=9)
    save(
        fig,
        "neff_dependence",
        "Primary comparison: 97 natural proteins, T=1, 100 randomized prompts. Top: fixed tiers and bootstrap 95% CIs. Bottom: paired differences and unadjusted OLS.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, region in zip(axes, ["all", "long"], strict=True):
        for model in COLORS:
            selected = (
                tiers[
                    tiers.depth_variable.eq("msa_depth")
                    & tiers["mode"].eq("randomized")
                    & tiers["range"].eq(region)
                    & tiers.model.eq(model)
                ]
                .set_index("tier")
                .reindex(TIERS)
            )
            errorbar(ax, np.arange(4), selected, model)
        counts = meta.loc[list(natural)].depth_tier.value_counts()
        ax.set(
            xticks=range(4),
            xticklabels=[f"{t}\nn={counts.get(t, 0)}" for t in TIERS],
            xlabel="Raw MSA sequence count",
            ylabel="R-precision",
            ylim=(0, 1),
            title=f"{region.capitalize()} range",
        )
        ax.legend(fontsize=9)
    save(
        fig,
        "raw_depth_sensitivity",
        "Raw sequence-count sensitivity on the same 97 proteins and primary condition. Empty tiers are explicit; raw depth is distinct from Neff.",
    )
    shuffles = read("shuffle_controls.csv")
    means = shuffles.groupby(["eval_set", "model", "stem"], as_index=False).mean(
        numeric_only=True
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, model in zip(axes[:2], COLORS, strict=True):
        selected = means[means.eval_set.eq("eval-val") & means.model.eq(model)]
        ax.scatter(
            selected.null_jaccard_sd,
            selected.observed_jaccard_sd,
            s=25,
            alpha=0.7,
            color=COLORS[model],
        )
        upper = (
            max(selected.null_jaccard_sd.max(), selected.observed_jaccard_sd.max())
            * 1.05
        )
        ax.plot([0, upper], [0, upper], ls="--", color=".5")
        ax.set(
            xlabel="Shuffled Jaccard SD",
            ylabel="Observed Jaccard SD",
            title=LABELS[model],
        )
    for model in COLORS:
        selected = means[means.eval_set.eq("eval-val") & means.model.eq(model)]
        axes[2].scatter(
            selected.half_chain_jaccard_sd,
            selected.null_jaccard_sd,
            s=20,
            alpha=0.6,
            color=COLORS[model],
            label=LABELS[model],
        )
    upper = max(means.half_chain_jaccard_sd.max(), means.null_jaccard_sd.max()) * 1.05
    axes[2].plot([0, upper], [0, upper], ls="--", color=".5")
    axes[2].set(
        xlabel="Null SD after 5 trades / rollout",
        ylabel="Null SD after 10 trades / rollout",
        title="Mixing sensitivity",
    )
    axes[2].legend(fontsize=9)
    save(
        fig,
        "coordination_control",
        "Fixed prompts, T=1. Curveball shuffles preserve every sample size and contact frequency. Ten null replicates; independent half/full chains assess mixing sensitivity. Coordination does not establish conformations.",
    )
    panel = read("panel_contacts.csv.gz")
    order = read("panel_order.csv").stem.tolist()
    for group_index in range(4):
        group = order[group_index * 2 : (group_index + 1) * 2]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for row, stem in enumerate(group):
            part = panel[panel.stem.eq(stem)]
            length = int(meta.loc[stem, "L"])
            for col, (field, title) in enumerate(
                [
                    ("truth", "Ground truth"),
                    ("default", "Default frequency"),
                    ("mpnn", "MPNN-only frequency"),
                    ("difference", "MPNN − default"),
                ]
            ):
                ax = axes[row, col]
                matrix = np.zeros((length, length))
                matrix[part.i, part.j] = part[field]
                matrix += matrix.T
                ax.imshow(
                    matrix,
                    origin="lower",
                    vmin=-1 if field == "difference" else 0,
                    vmax=1,
                    cmap="RdBu_r" if field == "difference" else "magma",
                    interpolation="nearest",
                    rasterized=True,
                )
                ax.set_title(title if row == 0 else "", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        f"{stem}\nL={length}, Neff={meta.loc[stem, 'msa_neff']:.1f}",
                        fontsize=9,
                    )
        axes[1, 1].set_xlabel("Frequency: black 0 → yellow 1", fontsize=9)
        axes[1, 3].set_xlabel("Difference: blue −1 → white 0 → red +1", fontsize=9)
        save(
            fig,
            f"representative_contact_maps_{group_index + 1}",
            "Eight proteins preselected by Neff rank; two shown per page. Resolved-pair truth and frequencies from 300 fixed-prompt samples, T=1. Frequencies use 0–1 and signed differences −1 to +1.",
        )
    selected = all_metrics[
        all_metrics.stem.isin(natural)
        & all_metrics.temperature.eq(1)
        & all_metrics["mode"].eq("randomized")
        & all_metrics["range"].eq("all")
        & all_metrics.cut.eq("R")
    ]
    pair = selected.pivot(index="stem", columns="model", values="precision").join(meta)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cov in zip(axes, ["50", "40"], strict=True):
        x = pair["best_surviving_identity_cov" + cov]
        dots = ax.scatter(
            x,
            pair.mpnn - pair.default,
            c=np.log10(pair.msa_neff),
            cmap="viridis",
            s=28,
            alpha=0.8,
        )
        ax.axhline(0, color=".5", ls="--")
        ax.set(
            xlabel=f"Best surviving native identity (coverage ≥{cov}%)",
            ylabel="MPNN-only − default R-precision",
            title="Native AFDB + ESM-Atlas evidence",
        )
        fig.colorbar(dots, ax=ax, label="log₁₀ Neff")
    save(
        fig,
        "native_similarity_audit",
        "Exploratory exp245 native AFDB/ESM-Atlas alignment evidence. The 50% coverage filter restricts identity; 40% exposes local matches. Redesign sequences were not searched.",
    )


if __name__ == "__main__":
    main()
