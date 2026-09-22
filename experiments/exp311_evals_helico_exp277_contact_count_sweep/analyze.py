"""Summarize exp311 per-sample structures without weighting longer sweeps more.

Run ``uv run python analyze.py`` after run_sweep.py. Each target contributes
once to every set-level summary. The oracle optimum is selected separately for
each metric; the confidence pick is one structure per target selected by
Helico's ranking score across every cut and diffusion sample.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "scratch" / "results" / "samples.csv"
DATA = HERE / "data"
PLOTS = HERE / "plots"
LOW_MSA_SET = HERE.parent / "exp260_evals_msa_depth_stratified" / "data" / "low_msa_depth_set.csv"
METRICS = ("gdt_ts", "lddt", "tm_score", "rmsd")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a table with an explicit, stable header."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_samples() -> dict[tuple[str, str], list[dict]]:
    """Read and validate the expected 3 samples at every recorded cut."""
    by_target: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with SAMPLES.open() as stream:
        for row in csv.DictReader(stream):
            for key in ("n_residues", "L", "n_pairs", "sample_idx", "n_matched_atoms"):
                row[key] = int(row[key])
            for key in ("ranking_score", "ptm", "iptm", *METRICS):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"nonfinite {key} for {row['stem']}")
            by_target[(row["eval_set"], row["stem"])].append(row)
    for (eval_set, stem), rows in by_target.items():
        length = rows[0]["L"]
        expected = sorted({*range(0, length + 1, 10), length})
        cuts: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            if row["sample_idx"] in cuts[row["n_pairs"]]:
                raise ValueError(f"duplicate sample {eval_set}/{stem} k={row['n_pairs']}")
            cuts[row["n_pairs"]].add(row["sample_idx"])
        if sorted(cuts) != expected or any(indices != {0, 1, 2} for indices in cuts.values()):
            raise ValueError(f"incomplete cut/sample grid for {eval_set}/{stem}")
    counts = {eval_set: sum(dataset == eval_set for dataset, _ in by_target) for eval_set in ("eval-val", "eval-denovo")}
    if counts != {"eval-val": 96, "eval-denovo": 19}:
        raise ValueError(f"incomplete target coverage: {counts}")
    return by_target


def best_for_metric(rows: list[dict], metric: str) -> dict:
    """Select a metric oracle with stable cut/sample tie-breaking."""
    sign = -1 if metric == "rmsd" else 1
    return max(rows, key=lambda row: (sign * row[metric], -row["n_pairs"], -row["sample_idx"]))


def confidence_pick(rows: list[dict]) -> dict:
    """Select the one Helico-ranked structure, breaking ties by smaller cut."""
    return max(rows, key=lambda row: (row["ranking_score"], -row["n_pairs"], -row["sample_idx"]))


def paired_bootstrap_ci(differences: list[float]) -> tuple[float, float]:
    """Return a 95% protein-bootstrap interval for a paired mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(311)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def summarize(by_target: dict[tuple[str, str], list[dict]]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Produce per-target selection, set summaries, and per-cut curves."""
    rankings = json.loads((HERE / "scratch" / "targets" / "ranked_pairs.json").read_text())
    per_target = []
    per_cut_target = []
    effective_counts = []
    for (eval_set, stem), rows in sorted(by_target.items()):
        length = rows[0]["L"]
        ranked = confidence_pick(rows)
        top_zero = confidence_pick([row for row in rows if row["n_pairs"] == 0])
        top_l = confidence_pick([row for row in rows if row["n_pairs"] == length])
        effective_by_cut = {
            k: sum(abs(a - b) >= 6 for a, b in rankings[stem][:k])
            for k in sorted({row["n_pairs"] for row in rows})
        }
        for k, effective in effective_by_cut.items():
            effective_counts.append({
                "eval_set": eval_set, "stem": stem, "L": length,
                "requested_pairs": k, "effective_pairs": effective,
                "dropped_short_pairs": k - effective,
            })
        for metric in METRICS:
            values = [row[metric] for row in rows]
            oracle = best_for_metric(rows, metric)
            per_target.append({
                "eval_set": eval_set, "stem": stem, "L": length, "metric": metric,
                "n_predictions": len(rows),
                "oracle_best": oracle[metric], "oracle_cut": oracle["n_pairs"],
                "mean_prediction": statistics.mean(values),
                "median_prediction": statistics.median(values),
                "confidence_top": ranked[metric], "confidence_cut": ranked["n_pairs"],
                "confidence_effective_cut": effective_by_cut[ranked["n_pairs"]],
                "confidence_score": ranked["ranking_score"],
                "top_zero_confidence": top_zero[metric],
                "top_l_confidence": top_l[metric],
            })
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["n_pairs"]].append(row)
        for k, cut_rows in grouped.items():
            picked = confidence_pick(cut_rows)
            for metric in METRICS:
                per_cut_target.append({
                    "eval_set": eval_set, "stem": stem, "L": length,
                    "n_pairs": k, "fraction_L": k / length, "metric": metric,
                    "n_effective_pairs": effective_by_cut[k],
                    "mean_prediction": statistics.mean(row[metric] for row in cut_rows),
                    "oracle_best": best_for_metric(cut_rows, metric)[metric],
                    "confidence_top": picked[metric],
                })

    grouped_summary: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_target:
        grouped_summary[(row["eval_set"], row["metric"])].append(row)
    summary = []
    for (eval_set, metric), rows in sorted(grouped_summary.items()):
        conf_vs_top_l = [row["confidence_top"] - row["top_l_confidence"] for row in rows]
        conf_vs_zero = [row["confidence_top"] - row["top_zero_confidence"] for row in rows]
        oracle_vs_conf = [row["oracle_best"] - row["confidence_top"] for row in rows]
        conf_ci = paired_bootstrap_ci(conf_vs_top_l)
        zero_ci = paired_bootstrap_ci(conf_vs_zero)
        oracle_ci = paired_bootstrap_ci(oracle_vs_conf)
        summary.append({
            "eval_set": eval_set, "metric": metric, "n_targets": len(rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "oracle_best", "mean_prediction", "median_prediction",
                "confidence_top", "top_zero_confidence", "top_l_confidence"
            )},
            "confidence_minus_top_l": statistics.mean(conf_vs_top_l),
            "confidence_minus_zero": statistics.mean(conf_vs_zero),
            "confidence_minus_zero_ci_low": zero_ci[0],
            "confidence_minus_zero_ci_high": zero_ci[1],
            "confidence_cut_mean": statistics.mean(row["confidence_cut"] for row in rows),
            "confidence_cut_median": statistics.median(row["confidence_cut"] for row in rows),
            "confidence_picked_zero_fraction": statistics.mean(row["confidence_cut"] == 0 for row in rows),
            "confidence_picked_top_l_fraction": statistics.mean(row["confidence_cut"] == row["L"] for row in rows),
            "confidence_minus_top_l_ci_low": conf_ci[0],
            "confidence_minus_top_l_ci_high": conf_ci[1],
            "oracle_minus_confidence": statistics.mean(oracle_vs_conf),
            "oracle_minus_confidence_ci_low": oracle_ci[0],
            "oracle_minus_confidence_ci_high": oracle_ci[1],
        })

    grouped_cut: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in per_cut_target:
        grouped_cut[(row["eval_set"], row["n_pairs"], row["metric"])].append(row)
    cut_curve = []
    for (eval_set, k, metric), rows in sorted(grouped_cut.items()):
        cut_curve.append({
            "eval_set": eval_set, "n_pairs": k, "metric": metric,
            "n_targets": len(rows),
            "mean_effective_pairs": statistics.mean(row["n_effective_pairs"] for row in rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "mean_prediction", "oracle_best", "confidence_top"
            )},
        })

    # A common fraction-of-L axis preserves the full target cohort. At each
    # grid point use the nearest actual 10-contact cut (smaller cut on a tie),
    # so no synthetic prediction is interpolated into the measurement.
    by_cut_target: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_cut_target:
        by_cut_target[(row["eval_set"], row["stem"], row["metric"])].append(row)
    relative_target: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for (eval_set, _stem, metric), rows in by_cut_target.items():
        length = rows[0]["L"]
        for decile in range(11):
            desired = length * decile / 10
            selected = min(rows, key=lambda row: (abs(row["n_pairs"] - desired), row["n_pairs"]))
            relative_target[(eval_set, decile, metric)].append(selected)
    relative_curve = []
    for (eval_set, decile, metric), rows in sorted(relative_target.items()):
        relative_curve.append({
            "eval_set": eval_set, "fraction_L": decile / 10,
            "metric": metric, "n_targets": len(rows),
            "mean_requested_pairs": statistics.mean(row["n_pairs"] for row in rows),
            "mean_effective_pairs": statistics.mean(row["n_effective_pairs"] for row in rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "mean_prediction", "oracle_best", "confidence_top"
            )},
        })
    return per_target, summary, cut_curve, effective_counts, relative_curve


def plot_curve(summary: list[dict], cut_curve: list[dict], relative_curve: list[dict]) -> None:
    """Show GDT-TS and lDDT against absolute contact count with cohort size."""
    for metric in ("gdt_ts", "lddt"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, eval_set in zip(axes, ("eval-val", "eval-denovo"), strict=True):
            rows = [row for row in cut_curve if row["eval_set"] == eval_set and row["metric"] == metric]
            rows.sort(key=lambda row: row["n_pairs"])
            # At large k only long proteins remain, so expose the changing
            # cohort in the lower panel rather than implying a paired curve.
            for key, label in (("confidence_top", "Helico rank at each k"),
                               ("mean_prediction", "sample mean"),
                               ("oracle_best", "oracle within k")):
                ax.plot([r["n_pairs"] for r in rows], [r[key] for r in rows], label=label)
            top_l = next(r["top_l_confidence"] for r in summary if r["eval_set"] == eval_set and r["metric"] == metric)
            ax.axhline(top_l, ls="--", color="0.5", label="top-L rank mean")
            ax.set_title(f"{eval_set} (n={96 if eval_set == 'eval-val' else 19})")
            ax.set_xlabel("Top k MarinFold contacts")
            ax.set_xlim(left=0)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("GDT-TS" if metric == "gdt_ts" else "lDDT")
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_plot_with_meta(
            fig, PLOTS / f"{metric}_by_contact_count.png",
            caption="Unpaired mean at each absolute k; targets with L < k leave the cohort. Dashed line is the all-target top-L mean.",
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, eval_set in zip(axes, ("eval-val", "eval-denovo"), strict=True):
            rows = [row for row in relative_curve if row["eval_set"] == eval_set and row["metric"] == metric]
            rows.sort(key=lambda row: row["fraction_L"])
            for key, label in (("confidence_top", "Helico rank at each cut"),
                               ("mean_prediction", "sample mean"),
                               ("oracle_best", "oracle within cut")):
                ax.plot([r["fraction_L"] for r in rows], [r[key] for r in rows], label=label)
            ax.set_title(f"{eval_set} (n={rows[0]['n_targets']})")
            ax.set_xlabel("Fraction of top-L MarinFold contacts")
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("GDT-TS" if metric == "gdt_ts" else "lDDT")
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_plot_with_meta(
            fig, PLOTS / f"{metric}_by_fraction_L.png",
            caption="Paired target cohort at every fraction of L; each point uses the nearest actual 10-contact cut.",
        )
        plt.close(fig)


def plot_selection_gap(per_target: list[dict]) -> None:
    """Compare Helico's confidence pick with the per-target GDT-TS oracle."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for ax, eval_set in zip(axes, ("eval-val", "eval-denovo"), strict=True):
        rows = [row for row in per_target if row["eval_set"] == eval_set and row["metric"] == "gdt_ts"]
        fractions = [row["confidence_cut"] / row["L"] for row in rows]
        scatter = ax.scatter(
            [row["oracle_best"] for row in rows],
            [row["confidence_top"] for row in rows],
            c=fractions, cmap="viridis", vmin=0, vmax=1, s=28, alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
        ax.set_title(f"{eval_set} (n={len(rows)})")
        ax.set_xlabel("Oracle best GDT-TS")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Helico-ranked GDT-TS")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Selected contact count / L", shrink=0.82)
    save_plot_with_meta(
        fig, PLOTS / "gdt_ts_selection_gap.png",
        caption="Each protein's best GDT-TS in the full sweep versus the structure Helico confidence ranks first.",
    )
    plt.close(fig)


def low_msa_designed_summary(per_target: list[dict]) -> list[dict]:
    """Score the frozen low-MSA-depth designed subset without touching eval-test."""
    with LOW_MSA_SET.open() as stream:
        stems = {
            row["stem"] for row in csv.DictReader(stream)
            if row["eval_set"] == "eval-denovo" and row["subset"] == "foldbench_designed"
        }
    if len(stems) != 13:
        raise ValueError(f"expected 13 frozen low-MSA designs, found {len(stems)}")
    rows_out = []
    for metric in METRICS:
        rows = [row for row in per_target if row["metric"] == metric and row["stem"] in stems]
        if len(rows) != 13:
            raise ValueError(f"missing low-MSA design for {metric}: {len(rows)}/13")
        rows_out.append({
            "subset": "eval-denovo-low-msa", "metric": metric, "n_targets": len(rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "oracle_best", "mean_prediction", "median_prediction", "confidence_top",
                "top_zero_confidence", "top_l_confidence"
            )},
        })
    return rows_out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    per_target, summary, cut_curve, effective_counts, relative_curve = summarize(load_samples())
    write_csv(DATA / "per_target_selection.csv", per_target)
    write_csv(DATA / "metric_summary.csv", summary)
    write_csv(DATA / "contact_count_curve.csv", cut_curve)
    write_csv(DATA / "effective_contacts.csv", effective_counts)
    write_csv(DATA / "contact_fraction_curve.csv", relative_curve)
    write_csv(DATA / "low_msa_designed_summary.csv", low_msa_designed_summary(per_target))
    plot_curve(summary, cut_curve, relative_curve)
    plot_selection_gap(per_target)
    for row in summary:
        print(
            f"{row['eval_set']:11s} {row['metric']:8s} n={row['n_targets']:3d} "
            f"oracle={row['oracle_best']:.4f} mean={row['mean_prediction']:.4f} "
            f"median={row['median_prediction']:.4f} rank={row['confidence_top']:.4f} "
            f"top0={row['top_zero_confidence']:.4f} topL={row['top_l_confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
