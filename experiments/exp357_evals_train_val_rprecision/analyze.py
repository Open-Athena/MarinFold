"""Summarize measured contact accuracy and paired diagnostic contrasts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta
from protocol import stable_seed

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = DATA / "results"


def interval(values: np.ndarray, label: str) -> tuple[float, float, float]:
    """Bootstrap independent protein means with a fixed, contrast-specific seed."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed(label, "bootstrap"))
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(frame: pd.DataFrame, groups: list[str], column: str) -> pd.DataFrame:
    """Return macro means and uncertainty, preserving each valid denominator."""
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[column].dropna().to_numpy()
        mean, low, high = interval(values, str(key))
        rows.append(
            dict(zip(groups, key, strict=True))
            | dict(n=len(values), mean=mean, ci_low=low, ci_high=high)
        )
    return pd.DataFrame(rows)


def paired_delta(
    left: pd.DataFrame, right: pd.DataFrame, label: str, keys: list[str]
) -> dict:
    """Compute within-protein or predefined matched-pair differences."""
    joined = left[keys + ["precision"]].merge(
        right[keys + ["precision"]],
        on=keys,
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    joined = joined.dropna(subset=["precision_left", "precision_right"])
    values = (joined.precision_left - joined.precision_right).to_numpy()
    mean, low, high = interval(values, label)
    return dict(
        contrast=label, n=len(joined), mean_delta=mean, ci_low=low, ci_high=high
    )


def main() -> None:
    frame = pd.read_csv(RESULTS / "per_protein_corrected.csv")
    if frame.duplicated(
        ["dataset", "stem", "model", "mode", "budget", "range", "cut"]
    ).any():
        raise ValueError("Duplicated metric rows")
    metrics = frame[(frame.cut == "R") & frame["range"].isin(["all", "long"])].copy()
    headline = metrics[(metrics["mode"] == "unconditioned") & (metrics.budget == 100)]
    models = list(json.loads((DATA / "checkpoints.json").read_text()))
    labels = [r["label"] for r in models]
    expected = {"afdb_train": 256, "afdb_val": 256, "eval_val": 97}
    cohort = pd.read_csv(DATA / "cohort_manifest.csv")
    expected_units = {
        (model, row.dataset, row.stem, band)
        for model in labels
        for row in cohort.itertuples()
        for band in ("all", "long")
    }
    observed_units = set(
        headline[["model", "dataset", "stem", "range"]].itertuples(
            index=False, name=None
        )
    )
    if observed_units != expected_units:
        raise ValueError(
            f"Headline coverage mismatch: missing={expected_units - observed_units}, extra={observed_units - expected_units}"
        )
    for (model, dataset, band), g in headline.groupby(["model", "dataset", "range"]):
        if len(g) != expected[dataset]:
            raise ValueError(f"Missing headline units: {model}/{dataset}/{band}")
    summary = summarize(headline, ["model", "dataset", "range"], "precision")
    headline.to_csv(DATA / "headline_per_protein.csv", index=False)
    summary.to_csv(DATA / "headline.csv", index=False)
    distributions = []
    for key, group in headline.groupby(["model", "dataset", "range"]):
        values = group.precision.dropna()
        distributions.append(
            dict(zip(["model", "dataset", "range"], key, strict=True))
            | dict(
                n=len(values),
                median=values.median(),
                q10=values.quantile(0.1),
                q90=values.quantile(0.9),
                n_at_least_90pct=int((values >= 0.9).sum()),
                n_at_least_95pct=int((values >= 0.95).sum()),
            )
        )
    pd.DataFrame(distributions).to_csv(DATA / "score_distribution.csv", index=False)
    cohort.groupby("dataset")[["seq_len", "global_plddt"]].agg(
        ["count", "mean", "median", "min", "max"]
    ).to_csv(DATA / "cohort_summary.csv")
    joined = headline.merge(
        cohort[["dataset", "stem", "match_id", "global_plddt", "round"]],
        on=["dataset", "stem"],
        validate="many_to_one",
    )
    joined["length_bin"] = pd.cut(
        joined.L,
        [0, 150, 300, 500, np.inf],
        labels=["<=150", "151-300", "301-500", ">500"],
    ).astype(str)
    joined["confidence_bin"] = pd.cut(
        joined.global_plddt, [0, 80, 90, 100], labels=["<=80", "80-90", ">90"]
    ).astype(str)
    stratified = []
    for stratum in ("length_bin", "confidence_bin", "round"):
        eligible = (
            joined if stratum == "length_bin" else joined[joined.dataset != "eval_val"]
        )
        result = summarize(
            eligible, ["model", "dataset", "range", stratum], "precision"
        )
        stratified.append(
            result.rename(columns={stratum: "value"}).assign(stratum=stratum)
        )
    pd.concat(stratified, ignore_index=True).to_csv(
        DATA / "stratified.csv", index=False
    )
    density = headline[headline.model == labels[0]].copy()
    density["contact_density"] = density.n_true / density.n_candidate
    density["contacts_per_residue"] = density.n_true / density.L
    summarize(density, ["dataset", "range"], "contact_density").to_csv(
        DATA / "random_rprecision.csv", index=False
    )
    density[
        [
            "dataset",
            "stem",
            "range",
            "L",
            "n_true",
            "n_candidate",
            "contact_density",
            "contacts_per_residue",
        ]
    ].to_csv(DATA / "contact_density.csv", index=False)
    contrasts = []
    for label in labels:
        for band in ("all", "long"):
            sub = joined[(joined.model == label) & (joined["range"] == band)]
            contrasts.append(
                paired_delta(
                    sub[sub.dataset == "afdb_train"],
                    sub[sub.dataset == "afdb_val"],
                    f"train-minus-afdb-val/{label}/{band}",
                    ["match_id"],
                )
            )
    for dataset in expected:
        for band in ("all", "long"):
            sub = headline[(headline.dataset == dataset) & (headline["range"] == band)]
            contrasts.append(
                paired_delta(
                    sub[sub.model == labels[1]],
                    sub[sub.model == labels[0]],
                    f"epoch2-minus-epoch1/{dataset}/{band}",
                    ["stem"],
                )
            )
    for label in labels:
        for dataset in expected:
            for band in ("all", "long"):
                sub = metrics[
                    (metrics.model == label)
                    & (metrics.dataset == dataset)
                    & (metrics["range"] == band)
                    & metrics.diagnostic
                ]
                full = sub[sub["mode"] == "unconditioned"]
                contrasts.append(
                    paired_delta(
                        full[full.budget == 1000],
                        full[full.budget == 100],
                        f"1000-minus-100/{label}/{dataset}/{band}",
                        ["stem"],
                    )
                )
                contrasts.append(
                    paired_delta(
                        sub[sub["mode"] == "oracle_half_remaining"],
                        sub[
                            (sub["mode"] == "unconditioned_remaining")
                            & (sub.budget == 100)
                        ],
                        f"oracle-half-minus-unconditioned/{label}/{dataset}/{band}",
                        ["stem"],
                    )
                )
    pd.DataFrame(contrasts).to_csv(DATA / "contrasts.csv", index=False)
    summarize(
        metrics[metrics.diagnostic],
        ["model", "dataset", "range", "mode", "budget"],
        "precision",
    ).to_csv(DATA / "diagnostic_summary.csv", index=False)
    single = pd.read_csv(RESULTS / "single_samples_corrected.csv")
    summarize(single, ["model", "dataset", "range"], "mean_f1").to_csv(
        DATA / "single_sample_f1.csv", index=False
    )
    timing = pd.read_csv(DATA / "timings_corrected.csv")
    if timing.unfinished_rollouts.sum():
        raise ValueError(
            "Unfinished samples remain; validate the output budget before reporting"
        )
    expected_samples = 2 * (609 * 100 + 96 * 900 + 96 * 100)
    if timing.n_rollouts.sum() != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} generated samples, got {timing.n_rollouts.sum()}"
        )
    completeness = (
        timing.groupby(["model", "dataset", "mode"])[
            ["n_rollouts", "stopped_rollouts", "unfinished_rollouts", "elapsed_seconds"]
        ]
        .sum()
        .reset_index()
    )
    completeness.to_csv(DATA / "completeness.csv", index=False)
    # Capped samples never vote. Report the complete-protein sensitivity explicitly.
    capped = timing[
        (timing["mode"] == "unconditioned") & (timing.unfinished_rollouts > 0)
    ][["model", "dataset", "stem"]]
    sensitivity = headline.merge(
        capped.assign(capped=True), on=["model", "dataset", "stem"], how="left"
    )
    summarize(
        sensitivity[sensitivity.capped.isna()],
        ["model", "dataset", "range"],
        "precision",
    ).to_csv(DATA / "complete_protein_sensitivity.csv", index=False)
    baseline = pd.read_csv(
        HERE.parents[0]
        / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2_epochs/subset_aggregate_metrics.csv"
    )
    reference_models = [
        "marinfold-exp277-full-epoch-m2-p06-step266344",
        "marinfold-exp277-full-epoch2-from213072-step479417",
    ]
    gates = []
    for label, reference in zip(labels, reference_models, strict=True):
        for band in ("all", "long"):
            observed = summary[
                (summary.model == label)
                & (summary.dataset == "eval_val")
                & (summary["range"] == band)
            ]["mean"].item()
            expected_value = baseline[
                (baseline.model == reference)
                & (baseline.subset == "eval-val")
                & (baseline["range"] == band)
                & (baseline.cut == "R")
            ].precision.item()
            gates.append(
                dict(
                    model=label,
                    range=band,
                    observed=observed,
                    reference=expected_value,
                    delta=observed - expected_value,
                    passed=abs(observed - expected_value) <= 0.005,
                )
            )
    pd.DataFrame(gates).to_csv(DATA / "reproduction_gate.csv", index=False)
    print("REPRODUCTION\n", pd.DataFrame(gates).to_string(index=False))
    print("HEADLINE\n", summary.to_string(index=False))
    print("CONTRASTS\n", pd.DataFrame(contrasts).to_string(index=False))
    print("CAPS\n", completeness.to_string(index=False))
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    datasets = list(expected)
    colors = ["#2463a6", "#e38338"]
    for axis, band in zip(axes, ("all", "long"), strict=True):
        for index, label in enumerate(labels):
            sub = (
                summary[(summary.model == label) & (summary["range"] == band)]
                .set_index("dataset")
                .loc[datasets]
            )
            xs = np.arange(3) + (index - 0.5) * 0.32
            axis.bar(
                xs, sub["mean"], 0.3, label=f"Epoch {index + 1}", color=colors[index]
            )
            axis.errorbar(
                xs,
                sub["mean"],
                yerr=np.vstack([sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]]),
                fmt="none",
                color="black",
                capsize=3,
            )
        axis.set_xticks(
            range(3),
            [
                f"AFDB train\n(n={256 if band == 'all' else 252})",
                f"AFDB validation\n(n={256 if band == 'all' else 254})",
                "Natural eval-val\n(n=97)",
            ],
        )
        axis.set_title(f"{'All-range' if band == 'all' else 'Long-range'} contacts")
        axis.set_ylim(0, 1)
        axis.set_ylabel("R-precision, 100 rollouts")
        axis.legend(frameon=False)
    fig.suptitle("MarinFold: training versus validation contact accuracy")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        str(plots / "train_validation.png"),
        dpi=180,
        caption="Macro contact R-precision; 95% protein bootstrap intervals. AFDB cohorts matched on length, confidence and pLDDT round. Epoch 2 is the existing continuation from the first epoch's pre-cooldown checkpoint.",
    )
    plt.close(fig)
    diagnostic_summary = pd.read_csv(DATA / "diagnostic_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for row, band in enumerate(("all", "long")):
        for col, dataset in enumerate(datasets):
            axis = axes[row, col]
            for index, label in enumerate(labels):
                sub = diagnostic_summary[
                    (diagnostic_summary.model == label)
                    & (diagnostic_summary.dataset == dataset)
                    & (diagnostic_summary["range"] == band)
                    & (diagnostic_summary["mode"] == "unconditioned")
                ].sort_values("budget")
                axis.plot(
                    sub.budget,
                    sub["mean"],
                    "o-",
                    color=colors[index],
                    label=f"Epoch {index + 1}",
                )
                axis.fill_between(
                    sub.budget, sub.ci_low, sub.ci_high, color=colors[index], alpha=0.12
                )
            axis.set_xscale("log")
            axis.set_ylim(0, 1)
            axis.set_title(f"{dataset.replace('_', ' ')} / {band}")
            axis.set_xlabel("Unconditioned rollouts")
            axis.set_ylabel("R-precision")
            axis.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        plots / "rollout_budget.png",
        dpi=180,
        caption="Nested rollout budgets on 32 fixed diagnostic proteins per cohort. Same proteins at every budget; shaded 95% protein bootstrap intervals. Single-rollout rankings include ties and are distinct from sample F1.",
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, band in zip(axes, ("all", "long"), strict=True):
        for index, label in enumerate(labels):
            for col, dataset in enumerate(datasets):
                sub = (
                    diagnostic_summary[
                        (diagnostic_summary.model == label)
                        & (diagnostic_summary.dataset == dataset)
                        & (diagnostic_summary["range"] == band)
                        & (diagnostic_summary.budget == 100)
                        & diagnostic_summary["mode"].isin(
                            ["unconditioned_remaining", "oracle_half_remaining"]
                        )
                    ]
                    .set_index("mode")
                    .loc[["unconditioned_remaining", "oracle_half_remaining"]]
                )
                xs = col + (index - 0.5) * 0.30 + np.array([-0.06, 0.06])
                axis.plot(
                    xs,
                    sub["mean"],
                    "o-",
                    color=colors[index],
                    label=f"Epoch {index + 1}" if col == 0 else None,
                )
                axis.errorbar(
                    xs,
                    sub["mean"],
                    yerr=[sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]],
                    fmt="none",
                    ecolor=colors[index],
                    capsize=3,
                )
        axis.set_xticks(
            np.arange(3), ["AFDB train", "AFDB validation", "Natural eval-val"]
        )
        axis.set_ylim(0, 1)
        axis.set_title(f"{band.capitalize()} contacts: no prefix → half true contacts")
        axis.set_ylabel("R-precision on remaining contact candidates")
        axis.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        plots / "oracle_conditioning.png",
        dpi=180,
        caption="Each line connects unconditioned and true-contact-conditioned predictions for the same 32 proteins, 100 rollouts each. Supplied pairs are removed from both candidate universes; oracle information cannot earn direct credit. This is a diagnostic, not a deployable accuracy estimate.",
    )
    plt.close(fig)
    if not all(gate["passed"] for gate in gates):
        raise ValueError("Eval-val reproduction failed; investigate before reporting.")


if __name__ == "__main__":
    main()
