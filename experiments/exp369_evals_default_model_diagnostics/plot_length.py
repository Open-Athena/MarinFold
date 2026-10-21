"""Plot the default model's archived eval-val accuracy against sequence length."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCE = EXPERIMENTS / "exp277_models_single_mpnn_pilot/data/eval_rollout_v2/contact_precision_all.csv"
SETS = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
MODEL = "marinfold-exp277-full-epoch-m2-p06-step266344"


def interval(values: np.ndarray, seed: int = 369) -> tuple[float, float]:
    """Bootstrap a protein-macro mean with a fixed seed."""
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def main() -> None:
    """Validate the archived subset and write plot, source CSVs, and provenance."""
    sets = pd.read_csv(SETS).query("eval_set == 'eval-val' and scorable == 1")
    data = pd.read_csv(SOURCE)
    data = data.loc[
        (data.model == MODEL) & (data.dataset == "foldbench_monomer")
        & (data.cut == "R") & data["range"].isin(["all", "long"])
    ].merge(sets[["stem", "is_viral", "seq_len"]], on="stem", validate="many_to_one")
    if len(sets) != 97 or len(data) != 194 or data.duplicated(["stem", "range"]).any():
        raise ValueError("Expected exactly 97 eval-val proteins in each contact range")
    if not data.precision.between(0, 1).all() or not (data.n_residues == data.seq_len).all():
        raise ValueError("Invalid scores or inconsistent sequence lengths")
    data["random_r_precision"] = data.n_true / data.n_candidate
    data["length_bin"] = pd.cut(data.n_residues, [0, 100, 200, 300, 500, 1000, np.inf], right=False)
    output = HERE / "data"
    output.mkdir(exist_ok=True)
    columns = ["dataset", "stem", "n_residues", "range", "precision", "n_true", "n_candidate", "random_r_precision", "is_viral"]
    data[columns].to_csv(output / "length_per_protein.csv", index=False)
    rows = []
    statistics = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.9), sharey=True, layout="constrained")
    for ax, contact_range, color in zip(axes, ("all", "long"), ("#007A87", "#6B54A6")):
        frame = data.loc[data["range"] == contact_range]
        rho = spearmanr(frame.n_residues, frame.precision)
        statistics[contact_range] = {
            "n": len(frame), "mean_r_precision": float(frame.precision.mean()),
            "mean_ci95": interval(frame.precision.to_numpy()),
            "spearman_rho": float(rho.statistic), "spearman_pvalue": float(rho.pvalue),
        }
        ax.scatter(frame.n_residues, frame.precision, color=color, alpha=.55, s=25, label="One protein")
        for label, group in frame.groupby("length_bin", observed=True):
            lower, upper = interval(group.precision.to_numpy())
            row = {"range": contact_range, "length_bin": str(label), "n": len(group),
                   "mean_length": group.n_residues.mean(), "mean_r_precision": group.precision.mean(),
                   "ci_lower": lower, "ci_upper": upper, "random_r_precision": group.random_r_precision.mean()}
            rows.append(row)
            ax.errorbar(row["mean_length"], row["mean_r_precision"],
                        yerr=[[row["mean_r_precision"] - lower], [upper - row["mean_r_precision"]]],
                        color="#152B38", marker="D", markersize=5, capsize=4, linewidth=1.8)
            ax.annotate(f"n={len(group)}", (row["mean_length"], upper), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
        ordered = frame.sort_values("n_residues")
        ax.scatter(ordered.n_residues, ordered.random_r_precision, color="#B0B7BE", s=12, marker="x", label="Random-ranking expectation")
        ax.set_title(f"{'All' if contact_range == 'all' else 'Long'}-range contacts\nmean={frame.precision.mean():.3f}; Spearman ρ={rho.statistic:.2f}")
        ax.set_xlabel("Protein length (residues)")
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("R-precision")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Default MarinFold • exp277 step 266344 • eval-val (97 natural proteins)", fontsize=13)
    save_plot_with_meta(fig, str(HERE / "plots/accuracy_vs_length.png"),
                        caption="100-rollout consensus; diamonds are length-bin protein means with 95% bootstrap intervals. All: sequence separation ≥6; long: ≥24. No new inference; no eval-test scores.")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "length_bins.csv", index=False)
    statistics["source_files_sha256"] = {str(p.relative_to(EXPERIMENTS.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE, SETS)}
    (output / "length_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
