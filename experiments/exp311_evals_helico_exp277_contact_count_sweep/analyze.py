"""Summarize exp311 per-sample structures without weighting longer sweeps more.

Run ``uv run python analyze.py`` after run_sweep.py. Each target contributes
once to every set-level summary. The oracle optimum is selected separately for
each metric; the confidence pick is one structure per target selected by
Helico's ranking score across every cut and diffusion sample.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "scratch" / "results" / "samples.csv"
DATA = HERE / "data"
PLOTS = HERE / "plots"
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


def summarize(by_target: dict[tuple[str, str], list[dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Produce per-target selection, set summaries, and per-cut curves."""
    per_target = []
    per_cut_target = []
    for (eval_set, stem), rows in sorted(by_target.items()):
        length = rows[0]["L"]
        ranked = confidence_pick(rows)
        top_l = confidence_pick([row for row in rows if row["n_pairs"] == length])
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
                "confidence_score": ranked["ranking_score"],
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
                    "mean_prediction": statistics.mean(row[metric] for row in cut_rows),
                    "oracle_best": best_for_metric(cut_rows, metric)[metric],
                    "confidence_top": picked[metric],
                })

    grouped_summary: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_target:
        grouped_summary[(row["eval_set"], row["metric"])].append(row)
    summary = []
    for (eval_set, metric), rows in sorted(grouped_summary.items()):
        summary.append({
            "eval_set": eval_set, "metric": metric, "n_targets": len(rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "oracle_best", "mean_prediction", "median_prediction",
                "confidence_top", "top_l_confidence"
            )},
            "confidence_minus_top_l": statistics.mean(
                row["confidence_top"] - row["top_l_confidence"] for row in rows
            ),
        })

    grouped_cut: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in per_cut_target:
        grouped_cut[(row["eval_set"], row["n_pairs"], row["metric"])].append(row)
    cut_curve = []
    for (eval_set, k, metric), rows in sorted(grouped_cut.items()):
        cut_curve.append({
            "eval_set": eval_set, "n_pairs": k, "metric": metric,
            "n_targets": len(rows),
            **{key: statistics.mean(row[key] for row in rows) for key in (
                "mean_prediction", "oracle_best", "confidence_top"
            )},
        })
    return per_target, summary, cut_curve


def plot_curve(summary: list[dict], cut_curve: list[dict]) -> None:
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
        axes[0].set_ylabel(metric.upper() if metric == "gdt_ts" else "lDDT")
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_plot_with_meta(
            fig, PLOTS / f"{metric}_by_contact_count.png",
            caption="Unpaired mean at each absolute k; targets with L < k leave the cohort. Dashed line is the all-target top-L mean.",
        )
        plt.close(fig)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    per_target, summary, cut_curve = summarize(load_samples())
    write_csv(DATA / "per_target_selection.csv", per_target)
    write_csv(DATA / "metric_summary.csv", summary)
    write_csv(DATA / "contact_count_curve.csv", cut_curve)
    plot_curve(summary, cut_curve)
    for row in summary:
        print(
            f"{row['eval_set']:11s} {row['metric']:8s} n={row['n_targets']:3d} "
            f"oracle={row['oracle_best']:.4f} mean={row['mean_prediction']:.4f} "
            f"median={row['median_prediction']:.4f} rank={row['confidence_top']:.4f} "
            f"topL={row['top_l_confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
