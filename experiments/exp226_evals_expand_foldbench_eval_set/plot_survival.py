# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 — the three figures.

1. ``survival_by_dataset.png`` — what fraction of each slice of the expanded
   eval set clears a <40 % / <30 % identity filter against exp199's training
   set. FoldBench is the dirtiest slice, and the 222 we never used are dirtier
   than the 100 we do.
2. ``natural_gain.png`` — the number that decides whether this was worth doing:
   decontaminated *natural* proteins before and after the expansion, against
   what the issue's extrapolation predicted.
3. ``identity_profile_old_vs_new.png`` — the identity and length distributions
   of the first 100 FoldBench monomers against the other 234, i.e. whether the
   oldest-deposited entries we happened to take are representative.

    uv run python plot_survival.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_survival import DATASET_NEW, GATED, THRESHOLDS, load_rows  # noqa: E402
from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"

#: Read as: the two slices this experiment contrasts get saturated colours, the
#: rest of the eval set stays muted.
COLOR_OLD = "#c0392b"      # the 100 FoldBench monomers we already use
COLOR_NEW = "#2980b9"      # the 222 net-new ones
COLOR_OTHER = "#95a5a6"
COLOR_PREDICTED = "#d68910"

DATASET_LABELS = {
    "foldbench100": "FoldBench-100\n(ours already)",
    DATASET_NEW: "FoldBench rest\n(+222, new)",
    "denovo_pdb": "de novo PDB\n(designed)",
    "cameo_hard": "CAMEO hard",
    "casp_fm": "CASP FM",
}


def plot_survival_by_dataset(by_dataset: pd.DataFrame, out: Path, args: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    order = list(DATASET_LABELS)
    frame = by_dataset.set_index("dataset").reindex(order)
    x = np.arange(len(order))
    width = 0.38

    for offset, threshold, hatch in ((-width / 2, 0.40, None), (width / 2, 0.30, "//")):
        tag = f"{threshold:.0%}".rstrip("%")
        pct = frame[f"survive_{tag}_pct"].to_numpy()
        colors = [COLOR_OLD if d == "foldbench100"
                  else COLOR_NEW if d == DATASET_NEW else COLOR_OTHER for d in order]
        bars = ax.bar(x + offset, pct, width, color=colors, hatch=hatch,
                      edgecolor="white", linewidth=0.6)
        for bar, count, total in zip(bars, frame[f"survive_{tag}"], frame["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{count}/{total}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in order], fontsize=9)
    ax.set_ylabel("% surviving the identity filter")
    ax.set_ylim(0, 100)
    ax.set_title("Homology-free survival against exp199's 70.9 M training sequences\n"
                 "left bar <40 % identity, right (hatched) <30 %", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot_with_meta(
        fig, out, args=args,
        caption="Share of each eval-set slice with no training sequence above the "
                "identity threshold (MMseqs2 -s 7.5, hit counted at E<=1e-3 and "
                "qcov>=0.50). Bar labels are survivors/total. The two FoldBench "
                "slices are the dirtiest in the eval set; the 222 we never used "
                "are dirtier still than the 100 we do.",
    )


def plot_natural_gain(headline: pd.DataFrame, newer: pd.DataFrame,
                      out: Path, args: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(headline))
    width = 0.3

    before = headline["orig554_natural"].to_numpy()
    after = headline["expanded776_natural"].to_numpy()
    # What the issue's extrapolation implied: FoldBench-100's survival rate
    # applied to the 222, all of it assumed natural.
    predicted = before + newer["predicted_from_old_rate"].to_numpy()

    for offset, values, color, label, alpha in (
        (-width, before, COLOR_OTHER, "eval set today (554)", 1.0),
        (0.0, predicted, COLOR_PREDICTED, "predicted by #226's extrapolation", 0.55),
        (width, after, COLOR_NEW, "expanded eval set (776), measured", 1.0),
    ):
        bars = ax.bar(x + offset, values, width, color=color, alpha=alpha,
                      edgecolor="white", linewidth=0.6, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{value:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} identity" for t in headline["threshold"]])
    ax.set_ylabel("decontaminated natural proteins")
    ax.set_title("The number that decides this: natural proteins surviving the filter",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(predicted.max(), after.max()) * 1.25)
    save_plot_with_meta(
        fig, out, args=args,
        caption="Natural (non-designed) eval proteins with no training homolog "
                "above the threshold. 'Natural' excludes both exp65's de novo "
                "set and any FoldBench entity whose RCSB source organism is "
                "synthetic. The expansion delivers its <40 % gain but only "
                "about half the predicted <30 % gain.",
    )


def plot_per_arm(by_arm: pd.DataFrame, complementarity: pd.DataFrame,
                 out: Path, args: list[str]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    # Left: what each training arm alone would have concluded, against the
    # union that exp199 actually trained on.
    net_new = by_arm[by_arm["subset"] == "net_new222"].reset_index(drop=True)
    x = np.arange(len(net_new))
    width = 0.26
    for offset, column, color, label in (
        (-width, "survive_afdb_only", "#8e44ad", "vs AFDB only (4.1 M seqs)"),
        (0.0, "survive_esm_atlas_only", "#16a085", "vs ESM-Atlas only (66.8 M)"),
        (width, "survive_union", COLOR_NEW, "vs both (70.9 M) — exp199's actual training set"),
    ):
        bars = ax_left.bar(x + offset, net_new[column], width, color=color,
                           edgecolor="white", linewidth=0.6, label=label)
        for bar, value in zip(bars, net_new[column]):
            ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         str(value), ha="center", fontsize=9)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{t} identity" for t in net_new["threshold"]])
    ax_left.set_ylabel("survivors of the 222 net-new")
    ax_left.set_title("Checking one arm would have overcounted 3x", fontsize=11)
    ax_left.legend(frameon=False, fontsize=8)
    ax_left.set_ylim(0, net_new["survive_afdb_only"].max() * 1.35)

    # Right: for the proteins that ARE filtered out, which arm was sufficient.
    subsets = ["orig554", "net_new222"]
    frame = complementarity[complementarity["threshold"] == "<40%"].set_index("subset")
    y = np.arange(len(subsets))
    left = np.zeros(len(subsets))
    for column, color, label in (("both_arms", "#34495e", "both arms"),
                                 ("afdb_only", "#8e44ad", "AFDB alone"),
                                 ("esm_atlas_only", "#16a085", "ESM-Atlas alone")):
        values = frame.loc[subsets, column].to_numpy()
        ax_right.barh(y, values, left=left, color=color, label=label,
                      edgecolor="white", linewidth=0.6)
        for yi, (value, start) in enumerate(zip(values, left)):
            if value:
                ax_right.text(start + value / 2, yi, str(value), ha="center",
                              va="center", fontsize=9, color="white")
        left += values
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(["the 554\neval set", "the 222\nnet-new"], fontsize=9)
    ax_right.set_xlabel("proteins removed by a <40 % filter")
    ax_right.set_title("Which arm supplies the disqualifying homolog", fontsize=11)
    ax_right.set_ylim(-0.6, 1.9)  # headroom so the legend clears the top bar
    ax_right.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=3)

    for ax in (ax_left, ax_right):
        ax.spines[["top", "right"]].set_visible(False)
    save_plot_with_meta(
        fig, out, args=args,
        caption="exp199 trained on both corpora, so the union is the filter that "
                "counts — but every prior overlap check (#41, #65, #94) looked at "
                "AFDB only. Left: AFDB alone would have left 76 of the 222 looking "
                "clean at <40 %; the union leaves 23. Right: for the net-new set "
                "the metagenomic ESM-Atlas arm is the *larger* sole contaminator "
                "(53 vs 39), reversing the pattern in the existing 554 (27 vs 60, "
                "which reproduces #213's split exactly).",
    )


def plot_identity_profile(rows: pd.DataFrame, lengths: dict,
                          out: Path, args: list[str]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for dataset, color, label in ((("foldbench100"), COLOR_OLD, "first 100 (ours)"),
                                  ((DATASET_NEW), COLOR_NEW, "other 222 (new)")):
        subset = rows[rows["dataset"] == dataset]
        # No covered hit means no measurable identity; plot it at 0 so the
        # curves stay comparable and the survivors are visible at the left edge.
        identity = subset[GATED].fillna(0.0).to_numpy() * 100
        ordered = np.sort(identity)
        ax_left.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100,
                     color=color, lw=2, label=f"{label}, n={len(ordered)}")
        ax_right.hist(subset["query_len"].to_numpy(), bins=np.logspace(1.6, 3.3, 22),
                      color=color, alpha=0.55, label=label)

    for threshold in THRESHOLDS:
        ax_left.axvline(threshold * 100, color="#7f8c8d", ls=":", lw=1.2)
        ax_left.text(threshold * 100 - 1, 97, f"<{threshold:.0%}", ha="right",
                     fontsize=8, color="#7f8c8d")
    ax_left.set_xlabel("best sequence identity to the training set (%)")
    ax_left.set_ylabel("cumulative % of slice")
    ax_left.set_title("Identity profile: are the newer entries different?", fontsize=11)
    ax_left.legend(frameon=False, fontsize=9, loc="lower right")

    ax_right.set_xscale("log")
    ax_right.set_xlabel("sequence length (residues)")
    ax_right.set_ylabel("proteins")
    ax_right.set_title(
        f"Lengths: median {lengths['foldbench100']['median']:.0f} vs "
        f"{lengths['foldbench_rest']['median']:.0f} aa", fontsize=11)
    ax_right.legend(frameon=False, fontsize=9)
    for ax in (ax_left, ax_right):
        ax.spines[["top", "right"]].set_visible(False)

    save_plot_with_meta(
        fig, out, args=args,
        caption="Left: cumulative distribution of best training-set identity; "
                "proteins with no covered hit are plotted at 0. The new curve "
                "sits to the right of the old one below 40 %, which is the "
                "shortfall against the extrapolation. Right: the length profiles "
                "match, so length is not the explanation.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=DATA / "eval_train_identity_expanded.csv")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    args = ap.parse_args()
    argv = sys.argv[1:]

    rows = pd.DataFrame(load_rows(args.table, args.targets))
    rows[GATED] = pd.to_numeric(rows[GATED], errors="coerce")
    rows["query_len"] = pd.to_numeric(rows["query_len"])
    by_dataset = pd.read_csv(DATA / "survival_by_dataset.csv")
    headline = pd.read_csv(DATA / "survival_headline.csv")
    newer = pd.read_csv(DATA / "newer_vs_older.csv")
    lengths = json.loads((DATA / "survival_summary.json").read_text())["length_profiles"]

    by_arm = pd.read_csv(DATA / "survival_by_arm.csv")
    complementarity = pd.read_csv(DATA / "arm_complementarity.csv")

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_survival_by_dataset(by_dataset, PLOTS / "survival_by_dataset.png", argv)
    plot_natural_gain(headline, newer, PLOTS / "natural_gain.png", argv)
    plot_per_arm(by_arm, complementarity, PLOTS / "per_arm_survival.png", argv)
    plot_identity_profile(rows, lengths, PLOTS / "identity_profile_old_vs_new.png", argv)
    eval2_headline = DATA / "eval2_headline.csv"
    if eval2_headline.exists():
        plot_eval2_scoreboard(pd.read_csv(eval2_headline),
                              pd.read_csv(DATA / "eval2_paired_deltas.csv"),
                              PLOTS / "eval2_scoreboard.png", argv)
    print(f"[plots] -> {PLOTS}", flush=True)
    return 0



def plot_eval2_scoreboard(headline: pd.DataFrame, deltas: pd.DataFrame,
                          out: Path, args: list[str]) -> None:
    """The six-predictor scoreboard, pooled vs natural-only."""
    from analyze_survival import DATASET_NEW  # noqa: F401  (kept for symmetry)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    predictors = ["MarinFold #199 (1.5B, seq only)", "Protenix-v2 single-seq",
                  "ESMFold", "ESMFold2", "Protenix-v2 + MSA", "seq-KNN k=10 (null)"]
    short = ["MarinFold\n#199", "Protenix\nsingle-seq", "ESMFold", "ESMFold2",
             "Protenix\n+MSA", "seq-KNN\n(null)"]
    colors = ["#c0392b", "#2980b9", "#7f8c8d", "#16a085", "#8e44ad", "#d68910"]

    rows = headline[(headline["cut"] == "R") & (headline["range"] == "all")]
    pooled = rows[rows["subset"] == "eval2 (<40% id)"].iloc[0]
    natural = rows[rows["subset"] == "eval2 natural"].iloc[0]
    x = np.arange(len(predictors))
    width = 0.38
    for offset, series, hatch, label in (
        (-width / 2, pooled, None, f"eval2 pooled, n={int(pooled['n'])} (75% designed)"),
        (width / 2, natural, "//", f"eval2 natural, n={int(natural['n'])}"),
    ):
        bars = ax_left.bar(x + offset, [series[p] for p in predictors], width,
                           color=colors, hatch=hatch, edgecolor="white",
                           linewidth=0.6, label=label)
        for bar, p in zip(bars, predictors):
            ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                         f"{series[p]:.3f}", ha="center", fontsize=7.5)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(short, fontsize=8)
    ax_left.set_ylabel("R-precision (all ranges)")
    ax_left.set_ylim(0, 0.95)
    ax_left.set_title("Removing designs, not homologs, is what the ranking turns on",
                      fontsize=10.5)
    ax_left.legend(frameon=False, fontsize=8, loc="upper left")

    # Paired MarinFold-minus-baseline, natural subsets only.
    cut = deltas[deltas["subset"].isin(["eval2 natural", "the 23 net-new"])]
    labels, centres, los, his, cols = [], [], [], [], []
    for subset, colour in (("eval2 natural", "#34495e"), ("the 23 net-new", "#2980b9")):
        for _, row in cut[cut["subset"] == subset].iterrows():
            labels.append(f"{row['baseline'].split(' (')[0][:20]}\n({subset})")
            centres.append(row["delta"])
            los.append(row["delta"] - row["ci_lo"])
            his.append(row["ci_hi"] - row["delta"])
            cols.append(colour)
    y = np.arange(len(labels))
    ax_right.errorbar(centres, y, xerr=[los, his], fmt="o", ms=5,
                      ecolor="#95a5a6", elinewidth=1.4, capsize=3, linestyle="none",
                      markerfacecolor="none", markeredgewidth=0)
    for yi, (c, col) in enumerate(zip(centres, cols)):
        ax_right.plot(c, yi, "o", ms=6, color=col)
    ax_right.axvline(0, color="#c0392b", ls="--", lw=1.2)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(labels, fontsize=7)
    ax_right.set_xlabel("MarinFold #199 − baseline (R-precision), paired 95% CI")
    ax_right.set_title("Parity with Protenix single-seq returns on natural proteins",
                       fontsize=10.5)
    for ax in (ax_left, ax_right):
        ax.spines[["top", "right"]].set_visible(False)
    save_plot_with_meta(
        fig, out, args=args,
        caption="Left: eval2 pooled is 75% de novo design, and every structure "
                "predictor beats MarinFold there. Restricted to the 78 natural "
                "proteins the ordering changes and MarinFold matches Protenix-v2 "
                "single-seq. Right: paired MarinFold-minus-baseline differences "
                "with 10,000-resample bootstrap CIs; intervals crossing the dashed "
                "line are ties.",
    )

if __name__ == "__main__":
    raise SystemExit(main())
