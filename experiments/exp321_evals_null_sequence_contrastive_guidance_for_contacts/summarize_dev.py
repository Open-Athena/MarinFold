"""Summarize the frozen exp321 development screen and render its plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
BASELINE = "dev_g0_pa_pos"
MODE_ORDER = [
    "dev_g0_pa_pos",
    "dev_g025_pa_pos",
    "dev_g05_pa_pos",
    "dev_g1_pa_pos",
    "dev_g2_pa_pos",
    "dev_ratio_pa_pos",
    "dev_g05_pa_all",
    "dev_t11_pa_pos",
]
LABELS = {
    "dev_g0_pa_pos": "paired gamma=0",
    "dev_g025_pa_pos": "gamma=0.25",
    "dev_g05_pa_pos": "gamma=0.5",
    "dev_g1_pa_pos": "gamma=1",
    "dev_g2_pa_pos": "gamma=2",
    "dev_ratio_pa_pos": "pure ratio",
    "dev_g05_pa_all": "gamma=0.5, all tokens",
    "dev_t11_pa_pos": "ordinary T=1.1",
}
COLORS = {
    "dev_g0_pa_pos": "#222222",
    "dev_g025_pa_pos": "#4c78a8",
    "dev_g05_pa_pos": "#2b8cbe",
    "dev_g1_pa_pos": "#7b3294",
    "dev_g2_pa_pos": "#c51b7d",
    "dev_ratio_pa_pos": "#d73027",
    "dev_g05_pa_all": "#fdae61",
    "dev_t11_pa_pos": "#1a9850",
}
METRICS = [
    "consensus_r_precision",
    "true_union_recall",
    "union_recall_log_n_auc",
    "mean_rollout_precision",
    "mean_pairwise_jaccard",
    "mean_contacts",
]


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 50_000) -> tuple[float, float]:
    """Return a deterministic percentile CI for a mean across proteins."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def paired_deltas(natural: pd.DataFrame) -> pd.DataFrame:
    """Compute paired protein bootstrap intervals against gamma=0 at N=20."""
    final = natural[natural.N == natural.N.max()]
    rows = []
    seed = 321_000
    for region in ("all", "long"):
        baseline = final[(final["mode"] == BASELINE) & (final["range"] == region)].set_index("stem")
        for mode in MODE_ORDER[1:]:
            arm = final[(final["mode"] == mode) & (final["range"] == region)].set_index("stem")
            if set(arm.index) != set(baseline.index):
                raise ValueError(f"unpaired stems for {mode}/{region}")
            arm = arm.loc[baseline.index]
            for metric in METRICS:
                values = (arm[metric] - baseline[metric]).to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(values, seed)
                rows.append({
                    "mode": mode,
                    "label": LABELS[mode],
                    "range": region,
                    "N": int(final.N.max()),
                    "metric": metric,
                    "n": len(values),
                    "baseline_mean": float(baseline[metric].mean()),
                    "arm_mean": float(arm[metric].mean()),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                })
                seed += 1
            first_baseline = natural[
                (natural["mode"] == BASELINE) & (natural["range"] == region)
                & (natural.N == natural.N.min())
            ].set_index("stem").loc[baseline.index]
            first_arm = natural[
                (natural["mode"] == mode) & (natural["range"] == region)
                & (natural.N == natural.N.min())
            ].set_index("stem").loc[baseline.index]
            baseline_increment = baseline.true_union_recall - first_baseline.true_union_recall
            arm_increment = arm.true_union_recall - first_arm.true_union_recall
            values = (arm_increment - baseline_increment).to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, seed)
            rows.append({
                "mode": mode,
                "label": LABELS[mode],
                "range": region,
                "N": int(final.N.max()),
                "metric": "incremental_true_union_recall_1_to_20",
                "n": len(values),
                "baseline_mean": float(baseline_increment.mean()),
                "arm_mean": float(arm_increment.mean()),
                "mean_delta": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
            seed += 1
    return pd.DataFrame(rows)


def diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap mean statement-level sequence-dependence diagnostics."""
    rows = []
    seed = 322_000
    for mode in MODE_ORDER:
        frame = diagnostics[diagnostics["mode"] == mode]
        for metric in ("log_ratio_auc", "true_false_log_ratio_delta"):
            values = frame[metric].dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, seed)
            rows.append({
                "mode": mode,
                "label": LABELS[mode],
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
            seed += 1
    return pd.DataFrame(rows)


def curve_summary(natural: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rollout-budget curves with across-protein standard errors."""
    metrics = ["consensus_r_precision", "true_union_recall", "mean_pairwise_jaccard"]
    rows = []
    for keys, frame in natural.groupby(["mode", "range", "N"], sort=False):
        mode, region, budget = keys
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            rows.append({
                "mode": mode,
                "label": LABELS[mode],
                "range": region,
                "N": int(budget),
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))),
            })
    return pd.DataFrame(rows)


def collect_timings() -> pd.DataFrame:
    """Collect the captured per-input predictor timings from local mirrors."""
    paths = sorted((HERE / "_cache").glob("dev_*/*/*.timing.parquet"))
    if not paths:
        raise FileNotFoundError("no mirrored timing parquets")
    timings = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = len(MODE_ORDER) * 31
    if len(timings) != expected:
        raise ValueError(f"expected {expected} timing rows, found {len(timings)}")
    return timings.sort_values(["mode", "cohort", "n_residues", "stem"])


def plot_curves(curves: pd.DataFrame) -> None:
    """Plot useful coverage and consensus accuracy across rollout budgets."""
    shown = [
        "dev_g0_pa_pos", "dev_g05_pa_pos", "dev_g1_pa_pos", "dev_g2_pa_pos",
        "dev_ratio_pa_pos", "dev_t11_pa_pos",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [
        ("all", "true_union_recall", "True-contact union recall"),
        ("long", "true_union_recall", "Long-range true-contact union recall"),
        ("all", "consensus_r_precision", "Consensus R-precision"),
        ("long", "consensus_r_precision", "Long-range consensus R-precision"),
    ]
    for ax, (region, metric, title) in zip(axes.flat, specs):
        for mode in shown:
            frame = curves[
                (curves["mode"] == mode)
                & (curves["range"] == region)
                & (curves.metric == metric)
            ].sort_values("N")
            ax.plot(frame.N, frame["mean"], marker="o", label=LABELS[mode], color=COLORS[mode])
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlabel("rollouts N")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Development screen: diversity gains trade against consensus accuracy", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "01_dev_curves.png",
        caption=(
            "Mean over 16 length-stratified eval-val development proteins. Pure-ratio and "
            "temperature sampling cover more true contacts, but both reduce consensus accuracy; "
            "moderate CFG-style guidance does not broaden the rollout pool."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_tradeoff(deltas: pd.DataFrame) -> None:
    """Plot paired accuracy/diversity deltas against the preregistered floor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    specs = [
        (
            "incremental_true_union_recall_1_to_20",
            "Delta incremental true-union recall, N=1 to 20",
            "higher means more useful additions after the first map",
        ),
        ("mean_pairwise_jaccard", "Delta pairwise Jaccard", "lower is more diverse"),
    ]
    region_markers = {"all": "o", "long": "s"}
    for ax, (metric, xlabel, subtitle) in zip(axes, specs):
        ax.axhspan(-0.005, 0.03, color="#d9f0d3", alpha=0.55)
        ax.axhline(-0.005, color="#238b45", linestyle="--", linewidth=1)
        ax.axvline(0, color="#777777", linewidth=0.8)
        for mode in MODE_ORDER[1:]:
            for region in ("all", "long"):
                xrow = deltas[
                    (deltas["mode"] == mode) & (deltas["range"] == region)
                    & (deltas.metric == metric)
                ].iloc[0]
                yrow = deltas[
                    (deltas["mode"] == mode) & (deltas["range"] == region)
                    & (deltas.metric == "consensus_r_precision")
                ].iloc[0]
                ax.scatter(
                    xrow.mean_delta, yrow.mean_delta, color=COLORS[mode],
                    marker=region_markers[region], s=55,
                    label=f"{LABELS[mode]} ({region})" if ax is axes[0] else None,
                )
        ax.set_xlabel(f"{xlabel}\n({subtitle})")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Delta consensus R-precision")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("No nonzero guidance arm satisfies the accuracy-diversity gate at N=20", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "02_accuracy_diversity_tradeoff.png",
        caption=(
            "Paired mean deltas versus gamma=0 on the same 16 proteins. Incremental coverage "
            "subtracts each arm's N=1 coverage from its N=20 coverage, separating rollout-pool "
            "breadth from first-map quality. The green band is the "
            "preregistered <=0.005 consensus-regression allowance. Circles are all-range and "
            "squares long-range. No guided arm combines lower overlap with preserved accuracy."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_reliability(fold_summary: pd.DataFrame) -> None:
    """Show fold-switch completion and malformed-statement stress failures."""
    frame = fold_summary.set_index("mode").loc[MODE_ORDER].reset_index()
    total = frame.n * 20
    completion = 100 * frame.finished / total
    malformed_per_100 = 100 * frame.malformed / total
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x, completion, color=[COLORS[mode] for mode in frame["mode"]])
    axes[0].set_ylabel("completed rollouts (%)")
    axes[0].set_ylim(90, 100.5)
    axes[0].axhline(100, color="#555555", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, malformed_per_100, color=[COLORS[mode] for mode in frame["mode"]])
    axes[1].set_ylabel("malformed statements / 100 rollouts")
    axes[1].set_xticks(x, [LABELS[mode] for mode in frame["mode"]], rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Pure-ratio decoding becomes unreliable on fold-switching sequences", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "03_foldswitch_reliability.png",
        caption=(
            "Fifteen primary fold-switch development pairs, 20 rollouts each. Pure ratio had "
            "25/300 unfinished rollouts and 648 malformed contact statements; gamma=2 had one "
            "unfinished rollout and 58 malformed statements."
        ),
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    """Write aggregate tables, decision record, timings, and summary plots."""
    natural = pd.read_csv(DATA / "dev_natural.csv")
    diagnostics = pd.read_csv(DATA / "dev_log_ratio.csv")
    fold_summary = pd.read_csv(DATA / "dev_foldswitch_summary.csv")
    deltas = paired_deltas(natural)
    curves = curve_summary(natural)
    diagnostic = diagnostic_summary(diagnostics)
    timings = collect_timings()
    deltas.to_csv(DATA / "dev_paired_deltas.csv", index=False)
    curves.to_csv(DATA / "dev_curve_summary.csv", index=False)
    diagnostic.to_csv(DATA / "dev_log_ratio_summary.csv", index=False)
    timings.to_csv(DATA / "timings.csv", index=False)

    decision = {
        "status": "frozen_before_heldout",
        "frozen_at_utc": "2026-09-23T01:14:08Z",
        "heldout_eval_val_accessed_at_freeze": False,
        "foldswitch_test_accessed_at_freeze": False,
        "selected_mode": "full_g05_pa_all",
        "parameters": {
            "null_kind": "polyala",
            "guidance_scope": "all",
            "gamma": 0.5,
            "pure_ratio": False,
            "temperature": 1.0,
            "top_p": 0.95,
        },
        "selection_metric": "accuracy/diversity Pareto frontier on development",
        "reason": (
            "Among reliable nonzero arms, all-token gamma=0.5 had the largest mean improvement "
            "in all- and long-range true-union log-N AUC while improving consensus R-precision."
        ),
        "caveat": (
            "Its pairwise Jaccard increased and its N=1-to-N=20 incremental union coverage "
            "decreased, so the held-out test must distinguish first-map quality from genuine "
            "rollout-pool diversity."
        ),
    }
    (DATA / "frozen_choice.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest = {
        "issue": 321,
        "checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "checkpoint_source": "s3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344",
        "source_revision": "359c3b72",
        "working_artifacts": "s3://marin-us-east-02a/MarinFold/exp321/null-sequence-guidance-v1",
        "public_artifacts": "hf://buckets/open-athena/MarinFold/data/exp321/null-sequence-guidance-v1",
        "natural_dev_proteins": 16,
        "foldswitch_dev_pairs": 15,
        "rollouts_per_target_mode": 20,
        "modes": MODE_ORDER,
        "total_rollouts": int(len(MODE_ORDER) * (16 + 15) * 20),
        "iris_cluster": "cw-rno2a",
        "gpu": "NVIDIA H100 80GB HBM3",
        "job_prefix": "/bizon/exp321-dev-",
        "wandb": None,
    }
    (DATA / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_curves(curves)
    plot_tradeoff(deltas)
    plot_reliability(fold_summary)
    print(f"wrote {len(deltas)} paired deltas, {len(timings)} timing rows, and 3 plots")


if __name__ == "__main__":
    main()
