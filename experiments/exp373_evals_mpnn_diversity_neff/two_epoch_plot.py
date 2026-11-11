"""Render the four-model report from committed protein-bootstrap source tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta
from summarize import TIERS
from two_epoch_summary import MODELS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/two_epoch"
COLORS = dict(zip(MODELS, ["#245c91", "#d35b39", "#238153", "#7957a0"], strict=True))
LABELS = dict(
    zip(
        MODELS,
        [
            "Full-data default",
            "MPNN 1 ep · 5e-4",
            "MPNN 2 ep · 5e-4",
            "MPNN 2 ep · 1e-3",
        ],
        strict=True,
    )
)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.dpi": 160,
    }
)


def save(fig: plt.Figure, name: str, caption: str) -> None:
    """Keep captions and the generating invocation with every publication figure."""
    fig.tight_layout(pad=1.6)
    save_plot_with_meta(fig, HERE / "plots" / (name + ".png"), caption=caption)
    plt.close(fig)


def band(ax: plt.Axes, rows: pd.DataFrame, model: str, x: str = "k") -> None:
    """Draw pointwise protein-bootstrap uncertainty around a model mean."""
    rows = rows.sort_values(x)
    ax.plot(rows[x], rows["mean"], color=COLORS[model], label=LABELS[model], lw=2)
    ax.fill_between(rows[x], rows.ci_low, rows.ci_high, color=COLORS[model], alpha=0.09)


def curves() -> None:
    """Show ensemble/discovery and individual-oracle accuracy through 1000."""
    curve = pd.read_csv(DATA / "curve_summary.csv")
    oracle = pd.read_csv(DATA / "oracle_summary.csv")
    for cohort, title, suffix in [
        ("eval-val", "97 natural proteins", ""),
        ("eval-denovo", "19 designs (secondary)", "_designs"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for row, region in enumerate(["all", "long"]):
            for col, (endpoint, label) in enumerate(
                [
                    ("r_precision", "Ensemble R-precision"),
                    ("true_coverage", "True-contact coverage"),
                    ("false_discovered", "Distinct false contacts"),
                ]
            ):
                ax = axes[row, col]
                for model in MODELS:
                    part = curve[
                        curve.eval_set.eq(cohort)
                        & curve["range"].eq(region)
                        & curve.endpoint.eq(endpoint)
                        & curve.model.eq(model)
                    ]
                    band(ax, part, model)
                ax.set(
                    xscale="log",
                    xlim=(1, 1000),
                    ylabel=label,
                    title=f"{region.capitalize()} range",
                )
                ax.axvline(300, color=".5", ls=":", lw=1)
                ax.grid(alpha=0.18)
                if row == 1:
                    ax.set_xlabel("Number of fixed-prompt rollouts, k")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(
            f"Accuracy and discovery through 1,000 rollouts · {title}", fontsize=16
        )
        save(
            fig,
            "ensemble_discovery" + suffix,
            "Temperature 1; fixed prompts. Means and pointwise 95% protein-bootstrap intervals. Dotted line: original 300-sample limit. False means absent from benchmark truth.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
        for ax, region in zip(axes, ["all", "long"], strict=True):
            for model in MODELS:
                selected = oracle[
                    oracle.eval_set.eq(cohort)
                    & oracle["range"].eq(region)
                    & oracle.model.eq(model)
                ]
                band(ax, selected[selected.endpoint.eq("oracle_f1")], model)
                mean = selected[selected.endpoint.eq("mean_f1")].iloc[0]
                ax.axhline(mean["mean"], color=COLORS[model], ls="--", lw=1.5)
                ax.axhspan(mean.ci_low, mean.ci_high, color=COLORS[model], alpha=0.035)
            ax.set(
                xscale="log",
                xlim=(1, 1000),
                ylim=(0, 1),
                xlabel="Number of fixed-prompt rollouts, k",
                title=f"{region.capitalize()} range",
                ylabel="Individual-rollout F1",
            )
            ax.axvline(300, color=".5", ls=":", lw=1)
            ax.grid(alpha=0.18)
        axes[0].legend(fontsize=9, loc="upper left")
        axes[1].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=".2",
                    lw=2,
                    label="Oracle best@k (uses ground truth)",
                ),
                Line2D(
                    [0], [0], color=".2", ls="--", label="Mean individual F1 over 1,000"
                ),
            ],
            fontsize=9,
            loc="upper left",
        )
        fig.suptitle(f"Oracle best@k and mean individual F1 · {title}", fontsize=16)
        save(
            fig,
            "oracle_f1" + suffix,
            "Temperature 1; seeded prefix order. Solid: oracle best single-rollout F1. Dashed: mean of 1,000 individual F1 scores per protein. Pointwise 95% intervals; each range has its own oracle.",
        )


def diversity_and_temperature() -> None:
    """Compare matched-sample diversity and the original three-temperature sweep."""
    div = pd.read_csv(DATA / "diversity_summary.csv")
    controls = pd.read_csv(DATA / "count_control_summary.csv")
    accuracy = pd.read_csv(DATA / "accuracy_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, region in enumerate(["all", "long"]):
        selected = div[
            div.eval_set.eq("eval-val")
            & div.temperature.eq(1)
            & div["mode"].eq("fixed")
            & div["range"].eq(region)
            & div.endpoint.eq("jaccard")
        ]
        for index, model in enumerate(MODELS):
            r = selected[selected.model.eq(model)].iloc[0]
            axes[row, 0].errorbar(
                index,
                r["mean"],
                yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]],
                fmt="o",
                color=COLORS[model],
                capsize=4,
            )
        axes[row, 0].set(
            xticks=range(4),
            xticklabels=[LABELS[m].replace(" · ", "\n") for m in MODELS],
            ylabel="Mean pairwise Jaccard distance",
            title=region.capitalize() + " range · 300 fixed samples",
        )
        selected = controls[
            controls.eval_set.eq("eval-val")
            & controls.temperature.eq(1)
            & controls["mode"].eq("fixed")
            & controls["range"].eq(region)
        ]
        for index, model in enumerate(MODELS[1:]):
            r = selected[selected.model.eq(model)].iloc[0]
            axes[row, 1].errorbar(
                index,
                r["mean"],
                yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]],
                fmt="o",
                color=COLORS[model],
                capsize=4,
            )
        axes[row, 1].axhline(0, color=".5", lw=1)
        axes[row, 1].set(
            xticks=range(3),
            xticklabels=[LABELS[m].replace(" · ", "\n") for m in MODELS[1:]],
            ylabel="Jaccard difference from default",
            title="After matching each rollout's contact count",
        )
    fig.suptitle(
        "Diversity and contact-count control · 97 natural proteins", fontsize=16
    )
    save(
        fig,
        "diversity",
        "First 300 fixed samples at temperature 1. Higher Jaccard distance means more diversity. Paired thinning matches each rollout's contact count to default. Bars: 95% protein-bootstrap intervals.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for model in MODELS:
        part = div[
            div.eval_set.eq("eval-val")
            & div["mode"].eq("fixed")
            & div["range"].eq("all")
            & div.model.eq(model)
            & div.endpoint.eq("jaccard")
        ]
        band(axes[0], part, model, "temperature")
        part = accuracy[
            accuracy.eval_set.eq("eval-val")
            & accuracy["mode"].eq("fixed")
            & accuracy["range"].eq("all")
            & accuracy.model.eq(model)
        ]
        band(axes[1], part, model, "temperature")
    for ax, label in zip(
        axes, ["Mean pairwise Jaccard distance", "Ensemble R-precision"], strict=True
    ):
        ax.set(xlabel="Sampling temperature", ylabel=label, xticks=[0.7, 1, 1.3])
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Temperature sweep · 300 fixed samples · 97 natural proteins", fontsize=16
    )
    save(
        fig,
        "temperature_tradeoff",
        "All-range metrics at matched sample counts. First 300 fixed samples at each temperature; pointwise 95% protein-bootstrap intervals. This sweep does not select a new temperature.",
    )


def neff() -> None:
    """Keep Neff and raw-depth tier counts visible and separate."""
    frame = pd.read_csv(DATA / "tier_accuracy.csv")
    for depth, title, name in [
        ("msa_neff", "Effective MSA depth (Neff)", "neff_dependence"),
        ("msa_depth", "Raw MSA sequence count", "raw_depth_sensitivity"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for col, region in enumerate(["all", "long"]):
            selected = frame[
                frame.depth_variable.eq(depth)
                & frame["mode"].eq("randomized")
                & frame["range"].eq(region)
            ]
            counts = []
            for model in MODELS:
                rows = selected[selected.model.eq(model)].set_index("tier").loc[TIERS]
                counts = rows.n.astype(int).tolist()
                for row, series in [
                    (0, rows),
                    (
                        1,
                        selected[selected.model.eq(model + " minus default")]
                        .set_index("tier")
                        .reindex(TIERS),
                    ),
                ]:
                    if row == 1 and model == "default":
                        continue
                    offset = (MODELS.index(model) - 1.5) * 0.06
                    axes[row, col].errorbar(
                        np.arange(4) + offset,
                        series["mean"],
                        yerr=[
                            series["mean"] - series.ci_low,
                            series.ci_high - series["mean"],
                        ],
                        color=COLORS[model],
                        marker="o",
                        capsize=3,
                        label=LABELS[model],
                    )
            axes[0, col].set(
                title=region.capitalize() + " range",
                ylabel="R-precision (100 randomized prompts)",
            )
            axes[1, col].axhline(0, color=".5", lw=1)
            axes[1, col].set(
                ylabel="Paired difference from default",
                xticks=range(4),
                xticklabels=[
                    f"{tier}\nn={n}" for tier, n in zip(TIERS, counts, strict=True)
                ],
                xlabel=title,
            )
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(title + " · 97 natural proteins · temperature 1", fontsize=16)
        save(
            fig,
            name,
            "Primary accuracy uses 100 randomized prompts. Tier means and paired differences from default, with 95% protein-bootstrap intervals. Neff <10 has four proteins; raw depth <10 has none.",
        )


def coordination() -> None:
    """Show dependencies beyond contact frequencies without claiming conformations."""
    frame = pd.read_csv(DATA / "shuffle_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, endpoint, label in zip(
        axes,
        ["jaccard_sd_excess", "overlap_sd_excess"],
        ["Excess Jaccard-distance SD", "Excess contact-overlap SD"],
        strict=True,
    ):
        for index, model in enumerate(MODELS):
            r = frame[
                frame.eval_set.eq("eval-val")
                & frame.endpoint.eq(endpoint)
                & frame.model.eq(model)
            ].iloc[0]
            ax.errorbar(
                index,
                r["mean"],
                yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]],
                fmt="o",
                color=COLORS[model],
                capsize=4,
            )
        ax.axhline(0, color=".5", lw=1)
        ax.set(
            ylabel=label,
            xticks=range(4),
            xticklabels=[LABELS[m].replace(" · ", "\n") for m in MODELS],
        )
    fig.suptitle(
        "Observed minus frequency/size-preserving shuffle · 97 natural proteins",
        fontsize=15,
    )
    save(
        fig,
        "coordination_control",
        "Temperature 1; first 300 fixed samples. Ten 3,000-trade curveball chains preserve every sample size and contact frequency. Positive excess indicates dependencies beyond these margins; 95% protein-bootstrap intervals.",
    )


def similarity_and_interactions() -> None:
    """Expose observational native-similarity and Neff evidence for each redesign model."""
    metrics = pd.read_csv(DATA / "metrics.csv")
    selected = metrics[
        metrics.eval_set.eq("eval-val")
        & metrics.temperature.eq(1)
        & metrics["mode"].eq("randomized")
        & metrics["range"].eq("all")
        & metrics.cut.eq("R")
    ]
    meta = pd.read_csv(HERE / "data/metadata.csv").set_index("stem")
    pair = selected.pivot(index="stem", columns="model", values="precision").join(meta)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, cov in zip(axes, ["50", "40"], strict=True):
        for model in MODELS[1:]:
            ax.scatter(
                pair["best_surviving_identity_cov" + cov],
                pair[model] - pair.default,
                s=23,
                alpha=0.55,
                color=COLORS[model],
                label=LABELS[model],
            )
        ax.axhline(0, color=".5", ls="--")
        ax.set(
            xlabel=f"Best native identity at coverage ≥{cov}%",
            ylabel="Paired R-precision difference from default",
            title="Historical native AFDB + ESM-Atlas search",
        )
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "Available sequence-similarity evidence · 97 natural proteins", fontsize=15
    )
    save(
        fig,
        "native_similarity_audit",
        "Exploratory native-sequence alignment evidence from exp245, with all three redesign checkpoints. The 50% coverage filter restricts identity. The redesign corpus remains unsearched.",
    )
    slopes = pd.read_csv(DATA / "slopes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    names = []
    for ax, region in zip(axes, ["all", "long"], strict=True):
        names = []
        index = 0
        for model in MODELS[1:]:
            for sensitivity, label in [
                ("unadjusted", "Unadjusted"),
                ("length_viral_adjusted", "Length + viral adjusted"),
                ("nonviral", "Nonviral only"),
            ]:
                r = slopes[
                    slopes["mode"].eq("randomized")
                    & slopes["range"].eq(region)
                    & slopes.depth_variable.eq("msa_neff")
                    & slopes.model.eq(model + " minus default")
                    & slopes.sensitivity.eq(sensitivity)
                ].iloc[0]
                ax.errorbar(
                    r["mean"],
                    index,
                    xerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]],
                    fmt="o",
                    color=COLORS[model],
                    capsize=3,
                )
                names.append(LABELS[model] + "\n" + label)
                index += 1
        ax.axvline(0, color=".5", ls="--")
        ax.set(
            title=region.capitalize() + " range",
            xlabel="Slope of paired R difference per log10 Neff",
            yticks=range(len(names)),
            yticklabels=names,
        )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Continuous Neff interaction versus default · 95% protein-bootstrap intervals",
        fontsize=15,
    )
    save(
        fig,
        "neff_interaction",
        "Primary randomized-prompt condition. Negative slope means a shallower Neff dependence than default. Length/viral adjustment and nonviral restriction are sensitivity checks; intervals are unadjusted.",
    )


def panels() -> None:
    """Plot the same eight examples selected before the original inference."""
    contacts = pd.read_csv(DATA / "panel_contacts.csv.gz")
    order = pd.read_csv(HERE / "data/panel_order.csv").stem.tolist()
    meta = pd.read_csv(HERE / "data/metadata.csv").set_index("stem")
    for page in range(4):
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        for row, stem in enumerate(order[2 * page : 2 * page + 2]):
            selected = contacts[contacts.stem.eq(stem)]
            length = int(meta.loc[stem, "L"])
            for col, key in enumerate(["truth", *MODELS]):
                matrix = np.zeros((length, length))
                matrix[selected.i, selected.j] = selected[key]
                matrix += matrix.T
                ax = axes[row, col]
                im = ax.imshow(
                    matrix,
                    vmin=0,
                    vmax=1,
                    cmap="magma_r",
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_title(
                    "Benchmark truth" if key == "truth" else LABELS[key], fontsize=10
                )
                if col == 0:
                    ax.set_ylabel(
                        f"{stem} · Neff {meta.loc[stem, 'msa_neff']:.1f}\nResidue"
                    )
                ax.set_xlabel("Residue")
        fig.suptitle(
            f"Prespecified contact-map examples {2 * page + 1}–{2 * page + 2} · 300 fixed samples",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))
        cax = fig.add_axes((0.96, 0.2, 0.01, 0.6))
        fig.colorbar(im, cax=cax, label="Contact frequency")
        save_plot_with_meta(
            fig,
            HERE / "plots" / f"representative_contact_maps_{page + 1}.png",
            caption="Eight proteins were selected by Neff rank before inference. Same 300-sample frequency scale for all four models; resolved contacts only. Examples are illustrative, not selected for effect size.",
        )
        plt.close(fig)


def main() -> None:
    """Regenerate the report figures without loading any raw prediction outputs."""
    curves()
    diversity_and_temperature()
    neff()
    coordination()
    similarity_and_interactions()
    panels()


if __name__ == "__main__":
    main()
