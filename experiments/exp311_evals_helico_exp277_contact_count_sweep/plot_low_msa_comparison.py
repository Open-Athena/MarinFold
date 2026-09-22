"""Plot the seven-method GDT-TS comparison at low MSA depth."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_summary import save_plot_with_meta
from plot_predictor_comparison import (
    COLORS,
    EVAL_SETS,
    METHOD_GROUP,
    METHOD_IDS,
    METHOD_LABEL,
    Y_POSITIONS,
    add_group_guides,
    style_for,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
COMPARISON = DATA / "gdt_ts_predictor_comparison_per_target.csv"
DEPTHS = DATA / "msa_depth.csv"
THRESHOLDS = (10, 100)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 311


def load_rows() -> tuple[list[dict], list[dict]]:
    """Join each paired comparison row to its measured MSA depth."""
    depths = {}
    with DEPTHS.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["eval_set"], row["stem"])
            if key in depths:
                raise ValueError(f"duplicate MSA depth: {key}")
            depths[key] = int(row["msa_depth"])

    source_rows = []
    with COMPARISON.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["eval_set"], row["stem"])
            if key not in depths:
                raise ValueError(f"missing MSA depth: {key}")
            row["gdt_ts"] = float(row["gdt_ts"])
            row["msa_depth"] = depths[key]
            source_rows.append(row)

    subsets = []
    members = []
    for threshold in THRESHOLDS:
        selected = set()
        for row in source_rows:
            if row["msa_depth"] > threshold:
                continue
            subsets.append({"msa_depth_threshold": threshold, **row})
            selected.add((row["eval_set"], row["stem"], row["msa_depth"]))
        members.extend({
            "msa_depth_threshold": threshold,
            "eval_set": eval_set,
            "stem": stem,
            "msa_depth": msa_depth,
        } for eval_set, stem, msa_depth in sorted(selected))
    return subsets, members


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Compute means and paired top-L deltas within each cumulative depth cut."""
    lookup = {
        (row["msa_depth_threshold"], row["eval_set"], row["stem"], row["method_id"]): row["gdt_ts"]
        for row in rows
    }
    summary = []
    deltas = []
    counts = []
    for threshold in THRESHOLDS:
        for eval_set in EVAL_SETS:
            stems = sorted({
                stem for cut, dataset, stem, _ in lookup
                if cut == threshold and dataset == eval_set
            })
            counts.append({
                "msa_depth_threshold": threshold,
                "eval_set": eval_set,
                "n_targets": len(stems),
            })
            if not stems:
                for method_id in METHOD_IDS:
                    common = {
                        "msa_depth_threshold": threshold,
                        "eval_set": eval_set,
                        "method_id": method_id,
                        "method_label": METHOD_LABEL[method_id],
                        "method_group": METHOD_GROUP[method_id],
                        "n_targets": 0,
                    }
                    summary.append({
                        **common,
                        "mean_gdt_ts": "",
                        "ci_low": "",
                        "ci_high": "",
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                    })
                    deltas.append({
                        **common,
                        "reference_method_id": "helico_top_l",
                        "mean_gdt_ts_delta": "",
                        "ci_low": "",
                        "ci_high": "",
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                    })
                continue

            matrix = np.asarray([
                [lookup[threshold, eval_set, stem, method_id] for method_id in METHOD_IDS]
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
                common = {
                    "msa_depth_threshold": threshold,
                    "eval_set": eval_set,
                    "method_id": method_id,
                    "method_label": METHOD_LABEL[method_id],
                    "method_group": METHOD_GROUP[method_id],
                    "n_targets": len(stems),
                }
                summary.append({
                    **common,
                    "mean_gdt_ts": float(matrix[:, method_index].mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                })
                delta_low, delta_high = np.quantile(boot_deltas[:, method_index], (0.025, 0.975))
                deltas.append({
                    **common,
                    "reference_method_id": "helico_top_l",
                    "mean_gdt_ts_delta": float(paired[:, method_index].mean()),
                    "ci_low": float(delta_low),
                    "ci_high": float(delta_high),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                })
    return summary, deltas, counts


def panel_rows(rows: list[dict], eval_set: str, threshold: int) -> dict[str, dict]:
    """Index one plot panel by method."""
    return {
        row["method_id"]: row for row in rows
        if row["eval_set"] == eval_set and row["msa_depth_threshold"] == threshold
    }


def plot_means(summary: list[dict]) -> None:
    """Plot mean GDT-TS within the two cumulative low-depth cuts."""
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.2), sharex=True, sharey=True)
    for row_index, eval_set in enumerate(EVAL_SETS):
        for col_index, threshold in enumerate(THRESHOLDS):
            ax = axes[row_index, col_index]
            by_method = panel_rows(summary, eval_set, threshold)
            n_targets = by_method[METHOD_IDS[0]]["n_targets"]
            if not n_targets:
                ax.set_axis_off()
                ax.set_title(f"{eval_set}  ·  MSA depth ≤{threshold}  ·  n=0", fontsize=13)
                continue
            add_group_guides(ax)
            for y, method_id in zip(Y_POSITIONS, METHOD_IDS, strict=True):
                record = by_method[method_id]
                mean = float(record["mean_gdt_ts"])
                low = float(record["ci_low"])
                high = float(record["ci_high"])
                color, marker, size = style_for(method_id)
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt=marker,
                    ms=size,
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=3,
                    zorder=3,
                )
                ax.text(high + 0.012, y, f"{mean:.3f}", va="center", fontsize=9, color=color)
            ax.set_title(f"{eval_set}  ·  MSA depth ≤{threshold}  ·  n={n_targets}", fontsize=13)
            ax.set_xlabel("Mean GDT-TS")
            ax.set_xlim(0.02, 1.08)
            ax.set_ylim(-0.15, 7.55)
            ax.grid(axis="x", alpha=0.2)
    method_labels = [METHOD_LABEL[method_id] for method_id in METHOD_IDS]
    axes[0, 1].set_yticks(Y_POSITIONS, method_labels)
    axes[0, 1].tick_params(labelleft=True)
    axes[1, 0].set_yticks(Y_POSITIONS, method_labels)
    axes[1, 1].tick_params(labelleft=False)
    fig.suptitle("Structure accuracy at low MSA depth", fontsize=18, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Inclusive, cumulative depth cuts. Points are target means; bars are 95% protein-bootstrap intervals. "
        "The oracle uses ground truth.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_plot_with_meta(
        fig,
        PLOTS / "gdt_ts_low_msa_predictor_comparison.png",
        caption=(
            "GDT-TS at inclusive MSA-depth thresholds of 10 and 100 sequences. Depth is measured on the exact "
            "A3M supplied to Protenix-v2 + MSA; thresholds are cumulative and all methods use paired targets."
        ),
    )
    plt.close(fig)


def plot_deltas(deltas: list[dict]) -> None:
    """Plot paired low-depth differences from exact top-L Helico."""
    compared_ids = tuple(method_id for method_id in METHOD_IDS if method_id != "helico_top_l")
    positions = np.asarray((6.0, 5.0, 4.0, 3.0, 1.3, 0.3))
    numeric = [row for row in deltas if row["n_targets"] and row["method_id"] != "helico_top_l"]
    x_low = min(0.0, min(float(row["ci_low"]) for row in numeric)) - 0.14
    x_high = max(0.0, max(float(row["ci_high"]) for row in numeric)) + 0.14
    x_low = math.floor(x_low * 10) / 10
    x_high = math.ceil(x_high * 10) / 10

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.6), sharex=True, sharey=True)
    for row_index, eval_set in enumerate(EVAL_SETS):
        for col_index, threshold in enumerate(THRESHOLDS):
            ax = axes[row_index, col_index]
            by_method = panel_rows(deltas, eval_set, threshold)
            n_targets = by_method[METHOD_IDS[0]]["n_targets"]
            if not n_targets:
                ax.set_axis_off()
                ax.set_title(f"{eval_set}  ·  MSA depth ≤{threshold}  ·  n=0", fontsize=13)
                continue
            ax.axhspan(-0.15, 1.8, color="#FFF3E8", zorder=-3)
            ax.axhline(2.2, color="0.82", lw=1)
            ax.axvline(0, color="0.3", lw=1.2, zorder=1)
            ax.text(0.01, 0.975, "External predictors", transform=ax.transAxes, va="top", color="0.35", fontsize=9)
            ax.text(0.01, 0.34, "Helico sweep policies", transform=ax.transAxes, va="top", color="#9A4D1C", fontsize=9)
            for y, method_id in zip(positions, compared_ids, strict=True):
                record = by_method[method_id]
                mean = float(record["mean_gdt_ts_delta"])
                low = float(record["ci_low"])
                high = float(record["ci_high"])
                color, marker, size = style_for(method_id)
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    fmt=marker,
                    ms=size,
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=3,
                    zorder=3,
                )
                if mean >= 0:
                    label_x, horizontal = high + 0.012, "left"
                else:
                    label_x, horizontal = low - 0.012, "right"
                ax.text(label_x, y, f"{mean:+.3f}", va="center", ha=horizontal, fontsize=9, color=color)
            ax.set_title(f"{eval_set}  ·  MSA depth ≤{threshold}  ·  n={n_targets}", fontsize=13)
            ax.set_xlabel("Paired Δ GDT-TS vs Helico + top-L contacts")
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(-0.15, 6.55)
            ax.grid(axis="x", alpha=0.2)
    method_labels = [METHOD_LABEL[method_id] for method_id in compared_ids]
    axes[0, 1].set_yticks(positions, method_labels)
    axes[0, 1].tick_params(labelleft=True)
    axes[1, 0].set_yticks(positions, method_labels)
    axes[1, 1].tick_params(labelleft=False)
    fig.suptitle("Low-MSA difference from the usual top-L Helico scheme", fontsize=18, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Positive values beat exact top-L on the same low-depth proteins; "
        "bars are paired 95% protein-bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_plot_with_meta(
        fig,
        PLOTS / "gdt_ts_low_msa_delta_vs_top_l.png",
        caption=(
            "Paired GDT-TS difference from exact top-L Helico within inclusive MSA-depth cuts of 10 and 100. "
            "Thresholds are cumulative; intervals resample matched proteins."
        ),
    )
    plt.close(fig)


def main() -> None:
    rows, members = load_rows()
    summary, deltas, counts = summarize(rows)
    write_csv(DATA / "gdt_ts_low_msa_comparison_per_target.csv", rows)
    write_csv(DATA / "gdt_ts_low_msa_comparison_summary.csv", summary)
    write_csv(DATA / "gdt_ts_low_msa_comparison_deltas.csv", deltas)
    write_csv(DATA / "msa_depth_subset_members.csv", members)
    write_csv(DATA / "msa_depth_subset_counts.csv", counts)
    plot_means(summary)
    plot_deltas(deltas)
    for threshold in THRESHOLDS:
        for eval_set in EVAL_SETS:
            panel = panel_rows(summary, eval_set, threshold)
            n_targets = panel[METHOD_IDS[0]]["n_targets"]
            print(f"{eval_set} MSA depth <= {threshold}: n={n_targets}")
            if not n_targets:
                continue
            for method_id in METHOD_IDS:
                row = panel[method_id]
                print(
                    f"  {row['method_label']:58s} {float(row['mean_gdt_ts']):.4f} "
                    f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"
                )


if __name__ == "__main__":
    main()
