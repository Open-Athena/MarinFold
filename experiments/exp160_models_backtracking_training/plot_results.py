# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two figures for #160: is retraction discriminative, and did it cost accuracy.

Reads what ``score_eval.py`` wrote.

* **retraction.png** — the pass/fail. Enrichment is bounded by 1 / P(FP), so
  the model's 1.13x and the #159 corpus's 5.85x are not on a common scale: their
  FP base rates are 0.80 and 0.17, ceilings of 1.26x and 6.02x. The figure
  compares the fraction of *available* headroom each captured (52% vs 97%), with
  the raw numbers annotated, and shows the precision-vs-base-rate gap behind
  them.
* **accuracy.png** — the paired per-protein difference in R-precision between
  the backtracking model and the exp120 control it was fine-tuned from. Paired,
  not two means: both arms score the same 554 proteins, and the between-protein
  spread of R-precision (~0.3) is an order of magnitude larger than the effect
  we are trying to resolve, so an unpaired comparison could not see it.

    uv run --no-project --with pandas --with matplotlib python plot_results.py \\
        --rows data/exp160_rows.csv.gz --retraction data/exp160_retraction.csv \\
        --out-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The #159 ESM-Atlas corpus the model was trained on (its README's numbers).
# The corpus is an upper reference, not a target: its retractions are placed by
# a posterior trigger with a ground-truth correctness flush behind it, so it
# retracts essentially every surviving false positive. A trained model has no
# flush.
CORPUS = {"enrichment": 5.85, "precision": 0.974, "base_rate": 0.166,
          "mean_distance": 17.9, "median_distance": 9}

# Defaults are #160's arm labels; --trained / --control retarget the same two
# figures at #175's, whose backtracking arm is a *generation-time* mode of one
# checkpoint rather than a separate model.
TRAINED = "exp160-bt50"
CONTROL = "exp120-base"


def plot_retraction(retraction: pd.DataFrame, out: Path, trained: str = TRAINED) -> None:
    """Model vs corpus, normalised by headroom — the only fair comparison.

    Enrichment is bounded by ``1 / P(FP)``, so a model whose rollouts are mostly
    wrong cannot reach a high number however well it discriminates. Plotting the
    model's 1.13x beside the corpus's 5.85x on one axis would say almost nothing
    about discrimination and almost everything about the two FP base rates (0.80
    vs 0.17). The left panel plots the fraction of *available* headroom each one
    captured, with raw enrichment and ceiling annotated; the right panel shows
    the precision-vs-base-rate gap those numbers come from.
    """
    row = retraction[retraction.model == trained]
    if row.empty:
        print(f"[plot] no retraction row for {trained}; skipping")
        return
    r = row.iloc[0]
    enrich = float(r["retract_enrichment"])
    lo, hi = float(r["enrichment_ci_low"]), float(r["enrichment_ci_high"])
    base = float(r["fp_base_rate"])
    ceiling = 1.0 / base if base else np.nan
    corpus_ceiling = 1.0 / CORPUS["base_rate"]

    def headroom(e, c):
        return (e - 1.0) / (c - 1.0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    frac = [headroom(enrich, ceiling), headroom(CORPUS["enrichment"], corpus_ceiling)]
    err = [[frac[0] - headroom(lo, ceiling), 0.0],
           [headroom(hi, ceiling) - frac[0], 0.0]]
    ax.bar([0, 1], frac, width=0.55, color=["#3b6ea5", "#9aa5b1"], yerr=err, capsize=6)
    for k, (f, e, c) in enumerate(zip(frac, [enrich, CORPUS["enrichment"]],
                                      [ceiling, corpus_ceiling])):
        ax.text(k, f + 0.03, f"{f:.0%}", ha="center", fontweight="bold")
        ax.text(k, 0.02, f"{e:.2f}x of {c:.2f}x", ha="center", fontsize=9, color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"trained model\n(own rollouts, P(FP)={base:.2f})",
                        f"#159 corpus\n(P(FP)={CORPUS['base_rate']:.2f})"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of achievable enrichment captured")
    ax.set_title(f"Is retraction discriminative?  {enrich:.2f}x "
                 f"[{lo:.2f}, {hi:.2f}] — CI excludes 1.0")

    labels = ["P(FP | retracted)", "P(FP) base rate", "P(retracted | FP)"]
    vals = [float(r["retract_precision_fp"]), base, float(r["retract_recall_fp"])]
    ax2.barh(range(3), vals, color=["#3b6ea5", "#a0a0a0", "#8fb8de"])
    for k, v in enumerate(vals):
        ax2.text(v + 0.012, k, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1.12)
    ax2.invert_yaxis()
    ax2.set_title(f"{r['mean_retracts_per_doc'] / 100:.1f} retracts/rollout, "
                  f"delay {r['mean_retract_distance']:.1f} statements "
                  f"(corpus {CORPUS['mean_distance']})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def plot_accuracy(rows: pd.DataFrame, out: Path, trained: str = TRAINED,
                  control: str = CONTROL) -> None:
    models = set(rows.model.unique())
    if not {trained, control} <= models:
        print(f"[plot] need both arms, have {sorted(models)}; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rng in zip(axes, ("all", "long")):
        sub = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
        wide = sub.pivot_table(index=["dataset", "stem"], columns="model",
                               values="precision").dropna()
        delta = (wide[trained] - wide[control]).to_numpy()
        n = delta.size
        sem = delta.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ax.hist(delta, bins=41, color="#3b6ea5", alpha=0.85)
        ax.axvline(0, color="#333", lw=1)
        ax.axvline(delta.mean(), color="#b03030", lw=1.6)
        ax.set_xlabel(f"R-precision  {trained} − {control}  ({rng})")
        ax.set_ylabel("proteins")
        ax.set_title(f"{rng}: Δ = {delta.mean():+.4f} ± {1.96 * sem:.4f} (95% CI), "
                     f"n = {n}, wins {100 * (delta > 0).mean():.0f}%")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--retraction", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--trained", default=TRAINED, help="arm label to plot")
    ap.add_argument("--control", default=CONTROL, help="arm it is paired against")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_retraction(pd.read_csv(a.retraction), a.out_dir / "retraction.png", a.trained)
    plot_accuracy(pd.read_csv(a.rows), a.out_dir / "accuracy.png", a.trained, a.control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
