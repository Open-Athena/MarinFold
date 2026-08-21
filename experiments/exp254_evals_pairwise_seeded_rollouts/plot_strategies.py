# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed-strategy figure: does biasing the seeds toward long range help?

exp254's first result left one live lever. Consensus over 100 rollouts already
matches the oracle best single rollout at all-range, but at **long** range the
oracle is 0.037 ahead -- so whatever headroom the sampler has left is in the
long-separation contacts. This figure asks whether pointing the seeds there
reaches it.

Left panel: what each strategy actually handed the model. This is the framing
the question needs. "Top 100 overall" is **already 56.8 % long-range**, because
long-separation pairs dominate the candidate universe, so an equal-thirds split
*lowers* the long-range share to 34 % rather than raising it. Only the
long-range-only strategy is a bias toward long range.

Middle panel: the paired seeded-minus-i.i.d. gain in consensus R-precision, per
separation range. If seeding buys anything where the headroom is, the long-range
group is where it has to show.

Right panel: the mechanism check, from `exp254_seed_range.csv` -- seed accuracy
and the R-precision of the rollouts those seeds produced, split by the seed's
own separation range. Long-range seeds are the least accurate; the question is
whether they are nonetheless the most informative.

    uv run python plot_strategies.py --data data --out plots
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

ARMS = ("i.i.d.", "seeded top-100", "seeded long-range", "seeded 1/3 per range")
ARM_COLORS = {
    "i.i.d.": "#0072b2",
    "seeded top-100": "#d55e00",
    "seeded long-range": "#cc79a7",
    "seeded 1/3 per range": "#e69f00",
}
RANGE_ORDER = ("short", "medium", "long")
RANGE_COLORS = {"short": "#cfd8dc", "medium": "#90a4ae", "long": "#37474f"}
BOOTSTRAP_DRAWS = 10_000
SEED = 254


def interval(values: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stamp(path: Path, sources: dict[str, Path], caption: str) -> None:
    meta = {
        "script": Path(sys.argv[0]).name,
        "args": sys.argv[1:],
        "caption": caption,
        "plot": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {
            name: {"path": str(source.resolve().relative_to(REPO)),
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            for name, source in sources.items()
        },
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")


def composition_panel(axis, composition: pd.DataFrame) -> None:
    shares = composition[composition.seed_range != "median_separation"]
    wide = shares.pivot(index="arm", columns="seed_range", values="percent")
    arms = [a for a in ARMS if a in wide.index]
    bottom = np.zeros(len(arms))
    for name in RANGE_ORDER:
        values = wide.loc[arms, name].to_numpy()
        axis.bar(range(len(arms)), values, bottom=bottom, width=0.62,
                 color=RANGE_COLORS[name], label=name,
                 edgecolor="white", linewidth=0.8)
        for position, (value, base) in enumerate(zip(values, bottom)):
            if value >= 6:
                axis.text(position, base + value / 2, f"{value:.0f}", ha="center",
                          va="center", fontsize=9,
                          color="white" if name == "long" else "#33312e")
        bottom = bottom + values
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels([a.replace("seeded ", "") for a in arms], fontsize=8.5)
    axis.set_ylabel("share of the 100 seeds (%)")
    axis.set_ylim(0, 100)
    axis.legend(fontsize=8, title="seed separation", title_fontsize=8,
                frameon=False, loc="lower center", ncols=3,
                bbox_to_anchor=(0.5, -0.32))
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("What each strategy handed the model\n"
                   "'top 100 overall' is already mostly long-range",
                   fontsize=10.5)


def by_range_panel(axis, per_protein: pd.DataFrame) -> None:
    """Paired seeded-minus-i.i.d. consensus R-precision, per separation range.

    Plotted as differences rather than levels on purpose. Per-arm intervals on
    the levels are roughly twice as wide as the paired interval and overlap
    completely, which reads as "no data" when what the data actually says is
    "no difference, measured tightly". The four arms see identical proteins and
    identical document realizations, so the paired difference is the interval
    that belongs on this comparison.
    """
    frame = per_protein[per_protein.cut == "R"]
    wide = frame.pivot_table(index=["stem", "range"], columns="predictor",
                             values="precision").reset_index()
    width = 0.26
    axis.axhspan(-0.005, 0.005, color="#dddad6", alpha=0.55, zorder=0)
    axis.axhline(0, color="#33312e", linewidth=0.9, zorder=1)
    for offset, arm in enumerate(a for a in ARMS if a != "i.i.d."):
        means, los, his = [], [], []
        for name in RANGE_ORDER:
            subset = wide[wide["range"] == name]
            delta = (subset[f"{arm} consensus"]
                     - subset["i.i.d. consensus"]).dropna().to_numpy()
            low, high = interval(delta)
            means.append(delta.mean())
            los.append(delta.mean() - low)
            his.append(high - delta.mean())
        positions = np.arange(len(RANGE_ORDER)) + (offset - 1) * width
        axis.bar(positions, means, width=width, color=ARM_COLORS[arm],
                 label=arm.replace("seeded ", ""), zorder=2)
        axis.errorbar(positions, means, yerr=[los, his], fmt="none",
                      ecolor="#33312e", elinewidth=0.9, capsize=2, zorder=3)
    axis.set_xticks(np.arange(len(RANGE_ORDER)))
    axis.set_xticklabels([f"{n}-range" for n in RANGE_ORDER], fontsize=9)
    axis.set_ylabel("consensus R-precision, seeded - i.i.d.")
    axis.legend(fontsize=8, frameon=False, loc="upper left", title="seeded with",
                title_fontsize=8)
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("Paired gain over unseeded, per separation range\n"
                   "shaded band = the 0.005 tie threshold (#204)", fontsize=10.5)


def mechanism_panel(axis, by_range: pd.DataFrame) -> None:
    frame = by_range[by_range.arm == "seeded 1/3 per range"]
    frame = frame.set_index("seed_range").loc[list(RANGE_ORDER)]
    positions = np.arange(len(RANGE_ORDER))
    axis.bar(positions, frame["seed_accuracy"], width=0.6,
             color=[RANGE_COLORS[n] for n in RANGE_ORDER],
             label="seed is a true contact")
    for position, value in zip(positions, frame["seed_accuracy"]):
        axis.text(position, value + 0.015, f"{value:.2f}", ha="center", fontsize=8.5,
                  color="#33312e")
    axis.plot(positions, frame["rollout_precision"], color="#d55e00", marker="o",
              markersize=6, linewidth=2, label="R-precision of those rollouts")
    axis.axhline(frame["iid_rollout_precision"].iloc[0], color="#0072b2",
                 linewidth=1.6, linestyle="--", label="unseeded rollout")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{n}-range\nseed" for n in RANGE_ORDER], fontsize=9)
    axis.set_ylim(0, 1.0)
    axis.legend(fontsize=8, frameon=False, loc="upper right")
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("Within the equal-thirds arm, by seed range\n"
                   "same proteins, same realizations, different seed",
                   fontsize=10.5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    composition = pd.read_csv(args.data / "exp254_seed_composition.csv")
    per_protein = pd.read_csv(args.data / "exp254_per_protein.csv.gz")
    by_range = pd.read_csv(args.data / "exp254_seed_range.csv")

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    composition_panel(axes[0], composition)
    by_range_panel(axes[1], per_protein)
    mechanism_panel(axes[2], by_range)
    figure.suptitle("Biasing the seeds toward long-range contacts", fontsize=13)
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))

    dest = args.out / "seed_strategy_eval_val.png"
    figure.savefig(dest, dpi=200)
    plt.close(figure)
    stamp(dest, {"exp254_seed_composition": args.data / "exp254_seed_composition.csv",
                 "exp254_per_protein": args.data / "exp254_per_protein.csv.gz",
                 "exp254_seed_range": args.data / "exp254_seed_range.csv"},
          "Seed composition per strategy, consensus R-precision per separation "
          "range per strategy, and seed accuracy against rollout quality by the "
          "seed's own separation range, on eval-val for #232 m2-p06.")
    print(f"[plot] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
