"""Build exp277's README figures from saved evaluation results.

Run: uv run --no-project --with pandas --with matplotlib python plot_results.py
No predictor runs or held-out evaluation are performed by this script.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DATA = HERE / "data/eval_rollout_v2"
PLOTS = HERE / "plots"
NEW = "MarinFold exp277"
OLD = "MarinFold exp232"
BASE = (
    EXPERIMENTS
    / "exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2/data/coreweave_results/marinfold_precision.csv.gz"
)
BASELINES = (
    EXPERIMENTS
    / "exp250_evals_exploration_notebook/figures/data/2_rprecision/per_protein.csv"
)
SETS = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
BASELINE_NAMES = [
    "Protenix-v2 single-seq",
    "ESMFold",
    "ESMFold2",
    "Protenix-v2 + MSA",
    "seq-KNN (decontaminated corpus)",
]


def interval(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and deterministic 95% percentile bootstrap interval."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(277).choice(values, (10000, len(values))).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def save(fig: plt.Figure, name: str, caption: str) -> None:
    """Write raster/vector figures and their reproduction sidecars."""
    save_plot_with_meta(
        fig,
        PLOTS / f"{name}.png",
        caption=caption,
        script="plot_results.py",
        args=[],
        dpi=180,
    )
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    new = pd.read_csv(DATA / "contact_precision_all.csv")
    old = pd.read_csv(BASE)
    old = old[old.model.eq("marinfold-exp232-decontam-train-m2-p06-step363000")]
    memberships = pd.read_csv(SETS)
    allowed = memberships[
        memberships.eval_set.isin(["eval-val", "eval-denovo"])
        & memberships.scorable.eq(1)
    ]
    assert allowed.groupby("eval_set").size().to_dict() == {
        "eval-denovo": 19,
        "eval-val": 97,
    }
    labels = dict(zip(allowed.stem, allowed.eval_set))
    for frame in (new, old):
        frame["subset"] = [
            labels[stem] if dataset == "foldbench_monomer" else "legacy_554"
            for dataset, stem in zip(frame.dataset, frame.stem)
        ]
    new = new[new.cut.eq("R") & new["range"].isin(["all", "long"])]
    old = old[old.cut.eq("R") & old["range"].isin(["all", "long"])]
    paired = new.merge(
        old,
        on=["dataset", "stem", "subset", "range", "cut"],
        suffixes=("_exp277", "_exp232"),
        validate="one_to_one",
    )
    assert len(paired) == 1340
    paired["delta"] = paired.precision_exp277 - paired.precision_exp232
    paired[
        [
            "dataset",
            "stem",
            "subset",
            "range",
            "precision_exp277",
            "precision_exp232",
            "delta",
        ]
    ].to_csv(DATA / "paired_r_precision.csv", index=False)
    comparison = []
    for (subset, distance_range), group in paired.groupby(["subset", "range"]):
        mean, low, high = interval(group.delta.to_numpy())
        comparison.append(
            dict(
                subset=subset,
                range=distance_range,
                n=group.delta.notna().sum(),
                delta=mean,
                ci_low=low,
                ci_high=high,
            )
        )
    pd.DataFrame(comparison).to_csv(DATA / "paired_bootstrap.csv", index=False)

    baseline = pd.read_csv(BASELINES)
    baseline = baseline[
        baseline.stem.isin(labels)
        & baseline.predictor.isin(BASELINE_NAMES)
        & baseline["range"].eq("all")
        & baseline.cut.eq("R")
    ].copy()
    baseline["subset"] = baseline.stem.map(labels)
    chunks = [baseline[["dataset", "stem", "subset", "predictor", "value"]]]
    for frame, name in ((new, NEW), (old, OLD)):
        chunk = frame[
            frame.dataset.eq("foldbench_monomer") & frame["range"].eq("all")
        ].copy()
        chunk["predictor"] = name
        chunks.append(
            chunk.rename(columns={"precision": "value"})[
                ["dataset", "stem", "subset", "predictor", "value"]
            ]
        )
    points = pd.concat(chunks, ignore_index=True)
    points.to_csv(DATA / "figure_per_protein.csv", index=False)
    summary = []
    for (subset, predictor), group in points.groupby(["subset", "predictor"]):
        expected = 97 if subset == "eval-val" else 19
        assert len(group) == expected and group.stem.nunique() == expected
        mean, low, high = interval(group.value.to_numpy())
        summary.append(
            dict(
                subset=subset,
                predictor=predictor,
                n=expected,
                mean=mean,
                ci_low=low,
                ci_high=high,
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(DATA / "figure_summary.csv", index=False)
    order = [
        "seq-KNN (decontaminated corpus)",
        "Protenix-v2 single-seq",
        OLD,
        NEW,
        "ESMFold",
        "ESMFold2",
        "Protenix-v2 + MSA",
    ]
    short = [
        "Sequence KNN (native corpus)",
        "Protenix-v2 single sequence",
        "MarinFold exp232 (previous)",
        "MarinFold exp277 (new)",
        "ESMFold",
        "ESMFold2",
        "Protenix-v2 + MSA",
    ]
    for subset, name, title in [
        ("eval-val", "rprecision_natural", "Natural FoldBench · eval-val (97)"),
        ("eval-denovo", "rprecision_designed", "De novo FoldBench · eval-denovo (19)"),
    ]:
        rows = summary[summary.subset.eq(subset)].set_index("predictor").loc[order]
        fig, ax = plt.subplots(figsize=(7.8, 4.6), layout="constrained")
        colors = [
            "#aaa" if p not in (NEW, OLD) else ("#b83c54" if p == NEW else "#365b8c")
            for p in order
        ]
        y = np.arange(len(order))
        ax.barh(y, rows["mean"], color=colors, height=0.62)
        ax.errorbar(
            rows["mean"],
            y,
            xerr=np.stack([rows["mean"] - rows.ci_low, rows.ci_high - rows["mean"]]),
            fmt="none",
            ecolor="#333",
            capsize=3,
        )
        for i, value in enumerate(rows["mean"]):
            ax.text(
                min(rows.ci_high.iloc[i] + 0.025, 0.98),
                i,
                f"{value:.3f}",
                va="center",
                fontsize=10,
            )
        ax.set(
            yticks=y,
            yticklabels=short,
            xlim=(0, 1.04),
            xlabel="Contact R-precision · 95% protein-bootstrap interval",
            title=title,
        )
        ax.invert_yaxis()
        save(
            fig,
            name,
            f"{title}. Identical proteins for all predictors; 10,000 bootstrap resamples. The KNN uses the native decontaminated corpus and does not index redesigns.",
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for ax, subset in zip(axes, ["legacy_554", "eval-val", "eval-denovo"]):
        group = paired[paired.subset.eq(subset) & paired["range"].eq("all")]
        ax.scatter(
            group.precision_exp232,
            group.precision_exp277,
            s=15,
            alpha=0.55,
            color="#b83c54",
        )
        ax.plot([0, 1], [0, 1], color="#777", lw=1)
        mean, low, high = interval(group.delta.to_numpy())
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="exp232 R-precision",
            ylabel="exp277 R-precision",
            title=f"{subset} (n={len(group)})\nΔ {mean:+.3f} [{low:+.3f}, {high:+.3f}]",
        )
        ax.set_aspect("equal")
    save(
        fig,
        "paired_comparison",
        "All-range R-precision on identical (dataset, stem) units; paired mean differences with 95% protein-bootstrap intervals. Legacy is only for comparing MarinFold checkpoints.",
    )

    low_members = pd.read_csv(
        EXPERIMENTS / "exp260_evals_msa_depth_stratified/data/low_msa_depth_set.csv"
    )
    available = low_members.merge(
        new[new["range"].eq("all")],
        on=["dataset", "stem"],
        how="left",
        validate="one_to_one",
        suffixes=("_membership", "_score"),
    )
    low_rows = []
    for label, members in [
        ("natural", available[~available.designed]),
        (
            "natural_foldbench",
            available[~available.designed & available.dataset.eq("foldbench_monomer")],
        ),
        ("designs", available[available.designed]),
    ]:
        values = members.precision.dropna().to_numpy()
        mean, low, high = interval(values)
        low_rows.append(
            dict(
                cut=label,
                n_available=len(values),
                n_frozen=len(members),
                mean=mean,
                ci_low=low,
                ci_high=high,
                note="Partial coverage: eval-test was not scored"
                if len(values) < len(members)
                else "Complete frozen membership",
            )
        )
    pd.DataFrame(low_rows).to_csv(DATA / "low_msa_coverage.csv", index=False)

    votes = np.loadtxt(DATA / "top7_votes.csv", delimiter=",")
    gt = json.loads((DATA / "top7_ground_truth.json").read_text())
    print("Top7 ground truth fields:", gt.keys())
    pairs = gt["contacts"]
    matrix = np.zeros_like(votes)
    for i, j, degree in pairs:
        if degree > 0.001 and abs(i - j) >= 6:
            matrix[i, j] = matrix[j, i] = 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].imshow(matrix, origin="lower", cmap="Greys", vmin=0, vmax=1)
    im = axes[1].imshow(votes / 100, origin="lower", cmap="magma_r", vmin=0, vmax=1)
    for ax, title in zip(
        axes, ["Top7 · experimental contacts", "Top7 · exp277 rollout consensus"]
    ):
        ax.set(title=title, xlabel="Residue index", ylabel="Residue index")
    fig.colorbar(im, ax=axes[1], label="Fraction of 100 rollouts", shrink=0.8)
    save(
        fig,
        "top7_maps",
        "Top7 (1qys_A, 92-residue benchmark sequence). Existing exp277 step-266344 rollouts; sequence and contact ground truth from exp89. No additional predictor run.",
    )

    progress = pd.read_csv(HERE / "data/epoch_validation_progress.csv")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(progress.global_step / 1000, progress.validation_loss, color="#b83c54")
    ax.axvline(213076 / 1000, color="#777", ls="--", label="WSD cooldown starts")
    ax.set(
        xlabel="Optimizer step (thousands)",
        ylabel="LM validation loss",
        title="First full-corpus epoch · exp277",
    )
    ax.legend()
    save(
        fig,
        "validation_loss",
        "Full language-model validations across the first epoch. Cooldown begins at step 213076; final checkpoint is step 266344.",
    )
    sources = [DATA / "contact_precision_all.csv", BASE, BASELINES, SETS]
    (DATA / "figure_provenance.json").write_text(
        json.dumps(
            {
                "sources": {
                    str(p.relative_to(EXPERIMENTS)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sources
                },
                "bootstrap_seed": 277,
                "bootstrap_replicates": 10000,
                "evaluation_sets": ["legacy_554", "eval-val", "eval-denovo"],
                "eval_test_scored": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
