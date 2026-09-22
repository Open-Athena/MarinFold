"""Compare exp311's three Helico policies with external structure predictors."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
SELECTIONS = DATA / "per_target_selection.csv"
BASELINES = DATA / "predictor_baseline_gdt_ts.csv"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 311
EVAL_SETS = ("eval-val", "eval-denovo")
METHODS = (
    ("protenix_v2_msa", "Protenix-v2 + MSA", "external"),
    ("esmfold2", "ESMFold2", "external"),
    ("esmfold", "ESMFold", "external"),
    ("protenix_v2_single_seq", "Protenix-v2 single sequence", "external"),
    ("helico_top_l", "Helico + top-L MarinFold contacts", "helico"),
    (
        "helico_confidence",
        "Helico + top-N MarinFold contacts (confidence selected)",
        "helico",
    ),
    ("helico_oracle", "Helico + top-N MarinFold contacts (oracle)", "oracle"),
)
METHOD_LABEL = {method_id: label for method_id, label, _ in METHODS}
METHOD_GROUP = {method_id: group for method_id, _, group in METHODS}
METHOD_IDS = tuple(method_id for method_id, _, _ in METHODS)
COLORS = {
    "external": "#5470C6",
    "helico": "#E8752E",
    "oracle": "#C23B6E",
}
Y_POSITIONS = np.asarray((7.0, 6.0, 5.0, 4.0, 2.4, 1.4, 0.4))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty table with a stable header."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_comparison_rows() -> list[dict]:
    """Join the seven methods on the exact exp311 target cohorts."""
    rows = []
    targets = set()
    with SELECTIONS.open() as stream:
        for row in csv.DictReader(stream):
            if row["metric"] != "gdt_ts":
                continue
            targets.add((row["eval_set"], row["stem"]))
            values = {
                "helico_top_l": row["top_l_confidence"],
                "helico_confidence": row["confidence_top"],
                "helico_oracle": row["oracle_best"],
            }
            for method_id, value in values.items():
                rows.append({
                    "eval_set": row["eval_set"],
                    "stem": row["stem"],
                    "method_id": method_id,
                    "method_label": METHOD_LABEL[method_id],
                    "method_group": METHOD_GROUP[method_id],
                    "gdt_ts": float(value),
                })
    with BASELINES.open() as stream:
        for row in csv.DictReader(stream):
            target = (row["eval_set"], row["stem"])
            if target not in targets:
                raise ValueError(f"baseline target not present in exp311: {target}")
            method_id = row["method_id"]
            if method_id not in METHOD_LABEL:
                raise ValueError(f"unknown baseline method: {method_id}")
            rows.append({
                "eval_set": row["eval_set"],
                "stem": row["stem"],
                "method_id": method_id,
                "method_label": METHOD_LABEL[method_id],
                "method_group": METHOD_GROUP[method_id],
                "gdt_ts": float(row["gdt_ts"]),
            })

    by_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["eval_set"], row["stem"])
        if row["method_id"] in by_target[key]:
            raise ValueError(f"duplicate method for {key}: {row['method_id']}")
        by_target[key].add(row["method_id"])
    expected = set(METHOD_IDS)
    incomplete = {key: sorted(expected - methods) for key, methods in by_target.items() if methods != expected}
    if incomplete:
        raise ValueError(f"incomplete paired comparison: {incomplete}")
    counts = {eval_set: sum(dataset == eval_set for dataset, _ in by_target) for eval_set in EVAL_SETS}
    if counts != {"eval-val": 96, "eval-denovo": 19}:
        raise ValueError(f"unexpected target counts: {counts}")
    return sorted(rows, key=lambda row: (row["eval_set"], row["stem"], row["method_id"]))


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compute target means, marginal intervals, and paired top-L deltas."""
    lookup = {
        (row["eval_set"], row["stem"], row["method_id"]): row["gdt_ts"]
        for row in rows
    }
    summary = []
    deltas = []
    for eval_set in EVAL_SETS:
        stems = sorted({stem for dataset, stem, _ in lookup if dataset == eval_set})
        matrix = np.asarray([
            [lookup[eval_set, stem, method_id] for method_id in METHOD_IDS]
            for stem in stems
        ], dtype=np.float64)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, len(stems), size=(BOOTSTRAP_DRAWS, len(stems)))
        boot_means = matrix[indices].mean(axis=1)
        top_l_index = METHOD_IDS.index("helico_top_l")
        paired = matrix - matrix[:, [top_l_index]]
        boot_deltas = paired[indices].mean(axis=1)
        for method_index, method_id in enumerate(METHOD_IDS):
            low, high = np.quantile(boot_means[:, method_index], (0.025, 0.975))
            summary.append({
                "eval_set": eval_set,
                "method_id": method_id,
                "method_label": METHOD_LABEL[method_id],
                "method_group": METHOD_GROUP[method_id],
                "n_targets": len(stems),
                "mean_gdt_ts": statistics.mean(matrix[:, method_index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
            delta_low, delta_high = np.quantile(boot_deltas[:, method_index], (0.025, 0.975))
            deltas.append({
                "eval_set": eval_set,
                "method_id": method_id,
                "method_label": METHOD_LABEL[method_id],
                "method_group": METHOD_GROUP[method_id],
                "reference_method_id": "helico_top_l",
                "n_targets": len(stems),
                "mean_gdt_ts_delta": statistics.mean(paired[:, method_index]),
                "ci_low": float(delta_low),
                "ci_high": float(delta_high),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    return summary, deltas


def style_for(method_id: str) -> tuple[str, str, float]:
    """Return color, marker, and size for a method."""
    group = METHOD_GROUP[method_id]
    if group == "oracle":
        return COLORS[group], "*", 11
    if method_id == "helico_confidence":
        return COLORS[group], "D", 7
    if method_id == "helico_top_l":
        return COLORS[group], "o", 7
    return COLORS[group], "o", 6


def add_group_guides(ax: plt.Axes) -> None:
    """Visually separate external predictors from Helico policies."""
    ax.axhspan(-0.15, 2.9, color="#FFF3E8", zorder=-3)
    ax.axhline(3.2, color="0.82", lw=1)
    ax.text(0.01, 0.975, "External predictors", transform=ax.transAxes, va="top", color="0.35", fontsize=9)
    ax.text(0.01, 0.385, "Helico policies", transform=ax.transAxes, va="top", color="#9A4D1C", fontsize=9)


def plot_means(summary: list[dict]) -> None:
    """Plot paired-cohort mean GDT-TS with target-bootstrap intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8), sharex=True, sharey=True)
    for ax, eval_set in zip(axes, EVAL_SETS, strict=True):
        by_method = {row["method_id"]: row for row in summary if row["eval_set"] == eval_set}
        add_group_guides(ax)
        for y, method_id in zip(Y_POSITIONS, METHOD_IDS, strict=True):
            row = by_method[method_id]
            color, marker, size = style_for(method_id)
            mean = row["mean_gdt_ts"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - row["ci_low"]], [row["ci_high"] - mean]],
                fmt=marker,
                ms=size,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
            if row["ci_high"] < 0.965:
                label_x = row["ci_high"] + 0.012
                horizontal = "left"
            else:
                label_x = row["ci_low"] - 0.012
                horizontal = "right"
            ax.text(label_x, y, f"{mean:.3f}", va="center", ha=horizontal, fontsize=9, color=color)
        n_targets = next(iter(by_method.values()))["n_targets"]
        ax.set_title(f"{eval_set}  ·  n={n_targets}", fontsize=13)
        ax.set_xlabel("Mean GDT-TS")
        ax.set_xlim(0.08, 1.01)
        ax.set_ylim(-0.15, 7.55)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(Y_POSITIONS, [METHOD_LABEL[method_id] for method_id in METHOD_IDS])
    fig.suptitle("Structure accuracy on identical target cohorts", fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.012,
        "Points are target means; bars are 95% protein-bootstrap intervals. "
        "The oracle uses ground truth to choose across contact counts and diffusion samples.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save_plot_with_meta(
        fig,
        PLOTS / "gdt_ts_predictor_comparison.png",
        caption=(
            "Target-mean GDT-TS for four external structure predictors and three exp311 Helico policies. "
            "All methods use the same 96 eval-val and 19 eval-denovo targets; intervals resample proteins."
        ),
    )
    plt.close(fig)


def plot_deltas(deltas: list[dict]) -> None:
    """Plot paired GDT-TS differences from the usual top-L Helico policy."""
    compared_ids = tuple(method_id for method_id in METHOD_IDS if method_id != "helico_top_l")
    compared_positions = np.asarray((6.0, 5.0, 4.0, 3.0, 1.3, 0.3))
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharex=True, sharey=True)
    for ax, eval_set in zip(axes, EVAL_SETS, strict=True):
        by_method = {row["method_id"]: row for row in deltas if row["eval_set"] == eval_set}
        ax.axhspan(-0.15, 1.8, color="#FFF3E8", zorder=-3)
        ax.axhline(2.2, color="0.82", lw=1)
        ax.axvline(0, color="0.3", lw=1.2, zorder=1)
        ax.text(0.01, 0.975, "External predictors", transform=ax.transAxes, va="top", color="0.35", fontsize=9)
        ax.text(0.01, 0.34, "Helico sweep policies", transform=ax.transAxes, va="top", color="#9A4D1C", fontsize=9)
        for y, method_id in zip(compared_positions, compared_ids, strict=True):
            row = by_method[method_id]
            color, marker, size = style_for(method_id)
            mean = row["mean_gdt_ts_delta"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - row["ci_low"]], [row["ci_high"] - mean]],
                fmt=marker,
                ms=size,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
            if mean >= 0:
                label_x = row["ci_high"] + 0.012
                horizontal = "left"
            else:
                label_x = row["ci_low"] - 0.012
                horizontal = "right"
            ax.text(label_x, y, f"{mean:+.3f}", va="center", ha=horizontal, fontsize=9, color=color)
        n_targets = next(iter(by_method.values()))["n_targets"]
        ax.set_title(f"{eval_set}  ·  n={n_targets}", fontsize=13)
        ax.set_xlabel("Paired Δ GDT-TS vs Helico + top-L contacts")
        ax.set_xlim(-0.45, 0.47)
        ax.set_ylim(-0.15, 6.55)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(compared_positions, [METHOD_LABEL[method_id] for method_id in compared_ids])
    fig.suptitle("Difference from the usual top-L Helico scheme", fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.012,
        "Positive values beat top-L on the same proteins; bars are paired 95% protein-bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_plot_with_meta(
        fig,
        PLOTS / "gdt_ts_delta_vs_top_l.png",
        caption=(
            "Paired mean GDT-TS difference from Helico with exact top-L MarinFold contacts. "
            "Positive values are better; intervals resample matched proteins."
        ),
    )
    plt.close(fig)


def main() -> None:
    comparison = load_comparison_rows()
    summary, deltas = summarize(comparison)
    write_csv(DATA / "gdt_ts_predictor_comparison_per_target.csv", comparison)
    write_csv(DATA / "gdt_ts_predictor_comparison_summary.csv", summary)
    write_csv(DATA / "gdt_ts_predictor_comparison_deltas.csv", deltas)
    plot_means(summary)
    plot_deltas(deltas)
    for eval_set in EVAL_SETS:
        print(eval_set)
        for row in summary:
            if row["eval_set"] == eval_set:
                print(
                    f"  {row['method_label']:58s} {row['mean_gdt_ts']:.4f} "
                    f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                )


if __name__ == "__main__":
    main()
