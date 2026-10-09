"""Aggregate all 97 matched proteins with paired bootstrap uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
LABELS = ["5", "10", "20", "L5", "L2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    artifacts = [json.loads(p.read_text()) for p in sorted((args.results / "complete").glob("*.json"))]
    if len(artifacts) != 97 or len({a["stem"] for a in artifacts}) != 97:
        raise ValueError("comparison requires exactly 97 distinct completed eval-val proteins")
    rows = pd.DataFrame([r for a in artifacts for r in a["metrics"]])
    timings = pd.DataFrame([r for a in artifacts for r in a["timings"]])
    assert len(rows) == 97 * 11 * 20
    assert not rows.duplicated(["stem", "mode", "range", "cut"]).any()
    assert not timings.duplicated(["stem", "mode"]).any()
    data = HERE / "data"
    rows.to_csv(data / "per_protein.csv", index=False)
    timings.to_csv(data / "timings.csv", index=False)
    summary = rows.groupby(["mode", "range", "cut"]).precision.agg(["mean", "count"]).reset_index()
    summary.to_csv(data / "summary.csv", index=False)
    rng = np.random.default_rng(354)
    indices = rng.integers(0, 97, size=(20_000, 97))
    comparisons = []
    for (distance, cut), group in rows.groupby(["range", "cut"]):
        pivot = group.pivot(index="stem", columns="mode", values="precision").sort_index()
        for mode in pivot.columns:
            if mode == "full_n100":
                continue
            valid = pivot[[mode, "full_n100"]].notna().all(axis=1)
            delta = (pivot[mode] - pivot.full_n100).to_numpy()
            # All and long R-precision are defined for all 97. Undefined
            # secondary cuts (e.g. no short-range truth) are paired omissions.
            sample = delta[valid.to_numpy()]
            local_indices = indices if len(sample) == 97 else rng.integers(0, len(sample), size=(20_000, len(sample)))
            draws = sample[local_indices].mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            lo99, hi99 = np.quantile(draws, [0.005, 0.995])
            comparisons.append(dict(mode=mode, range=distance, cut=cut, n=len(sample),
                mean=pivot.loc[valid, mode].mean(), baseline=pivot.loc[valid, "full_n100"].mean(),
                delta=sample.mean(), ci95_low=lo, ci95_high=hi,
                ci99_low=lo99, ci99_high=hi99, wins=int((sample > 0).sum()),
                losses=int((sample < 0).sum()), ties=int((sample == 0).sum())))
    paired = pd.DataFrame(comparisons)
    paired.to_csv(data / "paired_deltas.csv", index=False)
    timing_summary = timings.groupby("mode").agg(
        generated_tokens=("generated_tokens", "sum"), input_tokens=("input_tokens", "sum"),
        early_eos=("early_eos", "sum"), unfinished=("unfinished_rollouts", "sum"),
        valid_contact_votes=("valid_contact_votes", "sum"),
        shared_generation_seconds=("elapsed_seconds", "sum"),
    ).reset_index()
    timing_summary.to_csv(data / "sampling_summary.csv", index=False)
    metadata = pd.read_csv(HERE.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv")
    stratified = rows.merge(metadata[["stem", "is_viral"]], on="stem", validate="many_to_one")
    stratified.groupby(["is_viral", "mode", "range", "cut"]).precision.agg(["mean", "count"]).to_csv(data / "viral_split.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, distance in zip(axes, ("all", "long"), strict=True):
        for n, color in ((100, "#778da9"), (1000, "#007f73")):
            points = paired[(paired.range == distance) & (paired.cut == "R")].set_index("mode").loc[[f"short_{cap}_n{n}" for cap in LABELS]]
            offset = -0.08 if n == 100 else 0.08
            ax.errorbar(np.arange(5) + offset, points.delta,
                yerr=[points.delta - points.ci95_low, points.ci95_high - points.delta],
                label=f"{n:,} short rollouts", marker="o", color=color, capsize=3)
        ax.axhline(0, color="black", linewidth=1)
        ax.axhspan(-0.005, 0.005, color="grey", alpha=0.12)
        ax.set(xticks=np.arange(5), xticklabels=["5", "10", "20", "⌊L/5⌋", "⌊L/2⌋"],
               xlabel="Maximum emitted contacts per rollout", title=f"{distance.capitalize()} range")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("R-precision change from 100 full rollouts")
    axes[0].legend(frameon=False)
    fig.suptitle("Exp277 default · eval-val (97 proteins) · paired 95% bootstrap intervals")
    fig.tight_layout()
    save_plot_with_meta(fig, HERE / "plots/paired_rprecision.png",
                       caption="All five caps, with 100-short controls; shading marks ±0.005. Intervals are exploratory and pointwise.")
    plt.close(fig)
    print(paired[(paired.cut == "R") & paired.range.isin(["all", "long"])].to_string(index=False))
    print(timing_summary.to_string(index=False))


if __name__ == "__main__":
    main()
