"""Summarize held-out exp321 results, equal-time controls, and timings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import (
    TRUTH,
    mean_pairwise_jaccard,
    ordered_true_pairs,
    predicted_maps,
    r_precision,
    rollout_r_precision,
    vote_matrix,
)
from build_summary import save_plot_with_meta
from summarize_dev import bootstrap_mean_ci

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
GUIDED = "full_g05_pa_all"
PAIRED = "full_g0_pa_pos"
IID = "full_iid_single"
TEMPERATURE = "full_t08_single"
RATIO = "full_ratio_pa_pos"
HIGH_TEMPERATURE = "full_t11_single"
MODES = [PAIRED, GUIDED, TEMPERATURE, IID, HIGH_TEMPERATURE, RATIO]
LABELS = {
    PAIRED: "paired gamma=0",
    GUIDED: "frozen guidance",
    TEMPERATURE: "ordinary T=0.8",
    IID: "ordinary iid T=1.0",
    HIGH_TEMPERATURE: "ordinary T=1.1",
    RATIO: "pure native/null ratio",
}
COLORS = {
    PAIRED: "#222222",
    GUIDED: "#e66101",
    TEMPERATURE: "#5e3c99",
    IID: "#1b9e77",
    HIGH_TEMPERATURE: "#2166ac",
    RATIO: "#b2182b",
}
METRICS = [
    "validity_gated_oracle_r_precision",
    "oracle_r_precision",
    "consensus_r_precision",
    "true_union_recall",
    "union_recall_log_n_auc",
    "mean_rollout_precision",
    "mean_pairwise_jaccard",
    "mean_contacts",
]


def paired_comparisons(natural: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap frozen-guidance deltas against all declared controls."""
    final_n = int(natural.N.max())
    final = natural[natural.N == final_n]
    first = natural[natural.N == natural.N.min()]
    rows = []
    seed = 323_000
    for reference in (PAIRED, TEMPERATURE, IID, HIGH_TEMPERATURE, RATIO):
        for region in ("all", "long"):
            ref = final[
                (final["mode"] == reference) & (final["range"] == region)
            ].set_index("stem")
            arm = final[
                (final["mode"] == GUIDED) & (final["range"] == region)
            ].set_index("stem").loc[ref.index]
            for metric in METRICS:
                values = (arm[metric] - ref[metric]).to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(values, seed)
                rows.append({
                    "mode": GUIDED,
                    "reference": reference,
                    "reference_label": LABELS[reference],
                    "range": region,
                    "N": final_n,
                    "metric": metric,
                    "n": len(values),
                    "reference_mean": float(ref[metric].mean()),
                    "guided_mean": float(arm[metric].mean()),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                })
                seed += 1
            ref_first = first[
                (first["mode"] == reference) & (first["range"] == region)
            ].set_index("stem").loc[ref.index]
            arm_first = first[
                (first["mode"] == GUIDED) & (first["range"] == region)
            ].set_index("stem").loc[ref.index]
            ref_increment = ref.true_union_recall - ref_first.true_union_recall
            arm_increment = arm.true_union_recall - arm_first.true_union_recall
            values = (arm_increment - ref_increment).to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, seed)
            rows.append({
                "mode": GUIDED,
                "reference": reference,
                "reference_label": LABELS[reference],
                "range": region,
                "N": final_n,
                "metric": f"incremental_true_union_recall_1_to_{final_n}",
                "n": len(values),
                "reference_mean": float(ref_increment.mean()),
                "guided_mean": float(arm_increment.mean()),
                "mean_delta": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
            seed += 1
    return pd.DataFrame(rows)


def collect_timings() -> pd.DataFrame:
    """Collect every non-smoke development and full timing row."""
    paths = sorted((HERE / "_cache").glob("dev_*/*/*.timing.parquet"))
    paths += sorted((HERE / "_cache").glob("full_*/*/*.timing.parquet"))
    timings = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return timings.sort_values(["mode", "cohort", "n_residues", "stem"])


def truth_records() -> dict[str, dict]:
    """Load natural-protein truth keyed by stem."""
    records = {}
    with TRUTH.open() as source:
        for line in source:
            record = json.loads(line)
            if record["dataset"] == "foldbench_monomer":
                records[record["stem"]] = record
    return records


def equal_time_rows(natural: pd.DataFrame, timings: pd.DataFrame) -> pd.DataFrame:
    """Compare 100 guided rollouts with single-stream counts fitting its measured time."""
    targets = pd.read_csv(DATA / "targets.csv")
    targets = targets[(targets.cohort == "eval-val") & (targets.split == "test")]
    records = truth_records()
    final_guided = natural[(natural["mode"] == GUIDED) & (natural.N == 100)].set_index(
        ["stem", "range"]
    )
    timing_index = timings[timings.cohort == "eval-val"].set_index(["mode", "stem"])
    rows = []
    for target in targets.itertuples():
        guided_seconds = float(timing_index.loc[(GUIDED, target.stem), "elapsed_seconds"])
        record = records[target.stem]
        for control in (IID, HIGH_TEMPERATURE):
            control_seconds_200 = float(
                timing_index.loc[(control, target.stem), "elapsed_seconds"]
            )
            control_n = int(
                np.clip(np.floor(200 * guided_seconds / control_seconds_200), 1, 200)
            )
            frame = pd.read_parquet(
                HERE / "_cache" / control / "eval-val" / f"{target.stem}.parquet"
            ).sort_values("rollout").iloc[:control_n]
            valid = (
                frame.finished.to_numpy(dtype=bool)
                & (frame.malformed_contacts.to_numpy(dtype=int) == 0)
            )
            for region in ("all", "long"):
                maps = predicted_maps(frame, record, region)
                true = ordered_true_pairs(record, region)
                union = set().union(*(set(contacts) for contacts in maps))
                rollout_scores = [rollout_r_precision(contacts, true) for contacts in maps]
                gated_scores = [
                    score if is_valid else 0.0
                    for score, is_valid in zip(rollout_scores, valid, strict=True)
                ]
                guided = final_guided.loc[(target.stem, region)]
                rows.append({
                    "stem": target.stem,
                    "L": int(target.L),
                    "range": region,
                    "control": control,
                    "control_label": LABELS[control],
                    "guided_N": 100,
                    "control_equal_time_N": control_n,
                    "guided_seconds": guided_seconds,
                    "control_seconds_200": control_seconds_200,
                    "estimated_control_seconds": control_seconds_200 * control_n / 200,
                    "guided_validity_gated_oracle_r_precision": (
                        guided.validity_gated_oracle_r_precision
                    ),
                    "control_validity_gated_oracle_r_precision": float(
                        np.max(gated_scores)
                    ),
                    "guided_consensus_r_precision": guided.consensus_r_precision,
                    "control_consensus_r_precision": r_precision(
                        vote_matrix(maps, int(target.L)), record, region
                    ),
                    "guided_true_union_recall": guided.true_union_recall,
                    "control_true_union_recall": (
                        len(union & true) / len(true) if true else np.nan
                    ),
                    "guided_mean_pairwise_jaccard": guided.mean_pairwise_jaccard,
                    "control_mean_pairwise_jaccard": mean_pairwise_jaccard(maps),
                })
    return pd.DataFrame(rows)


def equal_time_summary(equal_time: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap guided-minus-iid equal-time metric differences."""
    rows = []
    seed = 324_000
    for control in (IID, HIGH_TEMPERATURE):
        for region in ("all", "long"):
            frame = equal_time[
                (equal_time["control"] == control) & (equal_time["range"] == region)
            ]
            for metric in (
                "validity_gated_oracle_r_precision",
                "consensus_r_precision",
                "true_union_recall",
                "mean_pairwise_jaccard",
            ):
                values = (
                    frame[f"guided_{metric}"] - frame[f"control_{metric}"]
                ).to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(values, seed)
                rows.append({
                    "control": control,
                    "control_label": LABELS[control],
                    "range": region,
                    "metric": metric,
                    "n": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "guided_mean": float(frame[f"guided_{metric}"].mean()),
                    "control_equal_time_mean": float(frame[f"control_{metric}"].mean()),
                    "median_control_equal_time_N": float(
                        frame.control_equal_time_N.median()
                    ),
                    "mean_control_equal_time_N": float(frame.control_equal_time_N.mean()),
                })
                seed += 1
    return pd.DataFrame(rows)


def validation_summary(dev: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Check the paired gamma=0 implementation on all 97 eval-val proteins."""
    combined = pd.concat([dev, heldout], ignore_index=True)
    final = combined[(combined["mode"] == PAIRED) & (combined.N == 100)]
    expected = {"all": 0.5538, "long": 0.5380}
    rows = []
    for region in ("all", "long"):
        frame = final[final["range"] == region]
        observed = float(frame.consensus_r_precision.mean())
        rows.append({
            "range": region,
            "n": frame.stem.nunique(),
            "observed_r_precision": observed,
            "exp306_reference": expected[region],
            "difference": observed - expected[region],
            "passes_0.005_gate": abs(observed - expected[region]) <= 0.005,
        })
    return pd.DataFrame(rows)


def curve_summary(natural: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out curves with protein-level standard errors."""
    rows = []
    for keys, frame in natural.groupby(["mode", "range", "N"], sort=False):
        mode, region, budget = keys
        for metric in (
            "validity_gated_oracle_r_precision",
            "consensus_r_precision",
            "true_union_recall",
            "mean_pairwise_jaccard",
        ):
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


def plot_curves(curves: pd.DataFrame) -> None:
    """Plot held-out oracle and consensus accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [
        ("all", "validity_gated_oracle_r_precision", "Oracle best-of-N R-precision"),
        (
            "long",
            "validity_gated_oracle_r_precision",
            "Long-range oracle best-of-N R-precision",
        ),
        ("all", "consensus_r_precision", "Consensus R-precision"),
        ("long", "consensus_r_precision", "Long-range consensus R-precision"),
    ]
    for ax, (region, metric, title) in zip(axes.flat, specs):
        for mode in MODES:
            frame = curves[
                (curves["mode"] == mode) & (curves["range"] == region)
                & (curves.metric == metric)
            ].sort_values("N")
            ax.plot(frame.N, frame["mean"], marker="o", color=COLORS[mode], label=LABELS[mode])
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100], labels=["1", "2", "5", "10", "25", "50", "100"])
        ax.set_title(title)
        ax.set_xlabel("rollouts N")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Oracle-first evaluation on 81 untouched eval-val proteins", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "04_heldout_curves.png",
        caption=(
            "Mean curves on the 81 non-development eval-val proteins. The top row is the primary "
            "endpoint: the best valid rollout in each prefix, with unfinished or malformed "
            "rollouts assigned zero. Ordinary iid and T=1.1 use their first 100 of 200 rollouts."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_primary(comparisons: pd.DataFrame) -> None:
    """Plot primary paired deltas and bootstrap intervals."""
    metrics = ["validity_gated_oracle_r_precision", "consensus_r_precision"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    references = [PAIRED, TEMPERATURE, IID, HIGH_TEMPERATURE, RATIO]
    positions = np.arange(len(references))
    offsets = {"all": -0.12, "long": 0.12}
    for ax, metric in zip(axes, metrics):
        for region, marker in (("all", "o"), ("long", "s")):
            frame = comparisons[
                (comparisons.metric == metric) & (comparisons["range"] == region)
            ].set_index("reference").loc[references]
            y = frame.mean_delta.to_numpy()
            low = y - frame.ci95_low.to_numpy()
            high = frame.ci95_high.to_numpy() - y
            ax.errorbar(
                positions + offsets[region], y, yerr=np.vstack([low, high]),
                fmt=marker, capsize=4, label=region,
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(positions, [LABELS[value] for value in references], rotation=18, ha="right")
        ax.set_title(
            "Validity-gated oracle R-precision"
            if metric.startswith("validity") else "Consensus R-precision (secondary)"
        )
        ax.set_ylabel("frozen guidance minus control")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Held-out paired effects with 95% protein-bootstrap intervals", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "05_heldout_primary_deltas.png",
        caption=(
            "Frozen-guidance deltas on 81 proteins. Oracle best-of-100 is primary; consensus "
            "accuracy is retained only as context. Error bars are paired protein-bootstrap 95% "
            "intervals."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_equal_time(summary: pd.DataFrame) -> None:
    """Plot paired equal-time oracle differences against single-stream controls."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    frame = summary[summary.metric == "validity_gated_oracle_r_precision"]
    order = [
        (IID, "all"),
        (IID, "long"),
        (HIGH_TEMPERATURE, "all"),
        (HIGH_TEMPERATURE, "long"),
    ]
    frame = frame.set_index(["control", "range"]).loc[order]
    y = frame.mean_delta.to_numpy()
    low = y - frame.ci95_low.to_numpy()
    high = frame.ci95_high.to_numpy() - y
    labels = [f"{LABELS[control]}\n{region}" for control, region in order]
    ax.errorbar(np.arange(len(order)), y, yerr=np.vstack([low, high]), fmt="o", capsize=4)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel("frozen guidance (N=100) minus equal-time control")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Equal-H100-time comparison using measured per-protein throughput", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "06_equal_time.png",
        caption=(
            "Each protein compares 100 paired guided rollouts with the number of ordinary T=1.0 "
            "or T=1.1 rollouts fitting the measured guided inference time, interpolated from each "
            "200-rollout timing. The endpoint is validity-gated oracle R-precision."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_foldswitch(common: pd.DataFrame, extended: pd.DataFrame) -> None:
    """Plot fixed fold-switch dual-mode coverage."""
    rows = []
    for mode in MODES:
        frame = common[common["mode"] == mode]
        rows.append({
            "label": LABELS[mode],
            "oracle": int(frame.oracle_dual.sum()),
            "blind": int(frame.blind_dual.sum()),
        })
    for mode in (IID, HIGH_TEMPERATURE):
        frame = extended[extended["mode"] == mode]
        rows.append({
            "label": f"{LABELS[mode]} (N=200)",
            "oracle": int(frame.oracle_dual.sum()),
            "blind": int(frame.blind_dual.sum()),
        })
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.35
    ax.bar(x - width / 2, frame.oracle, width, label="oracle pool")
    ax.bar(x + width / 2, frame.blind, width, label="reference-blind top 16")
    ax.set_xticks(x, frame.label, rotation=20, ha="right")
    ax.set_ylabel("dual-mode pairs out of 29")
    ax.set_ylim(0, 29)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fold-switching stress test", fontsize=14)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "07_foldswitch_modes.png",
        caption=(
            "Dual-mode coverage on the fixed 29-pair fold-switching stress set. This cohort is not "
            "a pristine benchmark; oracle and reference-blind top-16 definitions match exp304."
        ),
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    """Write final aggregate tables and plots from already-scored results."""
    heldout = pd.read_csv(DATA / "heldout_natural.csv")
    dev = pd.read_csv(DATA / "full_dev_natural.csv")
    folds = pd.read_csv(DATA / "heldout_foldswitch.csv")
    extended_folds = pd.read_csv(DATA / "heldout_200_foldswitch.csv")
    comparisons = paired_comparisons(heldout)
    curves = curve_summary(heldout)
    timings = collect_timings()
    equal_time = equal_time_rows(heldout, timings)
    equal_summary = equal_time_summary(equal_time)
    validation = validation_summary(dev, heldout)
    comparisons.to_csv(DATA / "heldout_paired_deltas.csv", index=False)
    curves.to_csv(DATA / "heldout_curve_summary.csv", index=False)
    timings.to_csv(DATA / "timings.csv", index=False)
    equal_time.to_csv(DATA / "equal_time_natural.csv", index=False)
    equal_summary.to_csv(DATA / "equal_time_summary.csv", index=False)
    validation.to_csv(DATA / "gamma0_validation.csv", index=False)
    plot_curves(curves)
    plot_primary(comparisons)
    plot_equal_time(equal_summary)
    plot_foldswitch(folds, extended_folds)
    print(
        f"wrote {len(comparisons)} held-out comparisons, {len(equal_time)} equal-time rows, "
        f"and {len(timings)} timing rows"
    )


if __name__ == "__main__":
    main()
