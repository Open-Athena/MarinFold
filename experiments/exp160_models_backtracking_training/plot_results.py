# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Two figures for #160: is retraction discriminative, and did it cost accuracy.

Reads what ``score_eval.py`` wrote.

* **retraction.png** — the pass/fail. Enrichment for the trained model against
  the two reference lines that give it meaning: 1.0 (retraction is noise) and
  the #159 training corpus's 5.85x. The per-protein ceiling (1 / FP base rate)
  is drawn too, because enrichment is bounded by the base rate and a model with
  fewer false positives to catch cannot reach as high a number — comparing raw
  enrichments across corpora without it is misleading.
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

TRAINED = "exp160-bt50"
CONTROL = "exp120-base"


def plot_retraction(retraction: pd.DataFrame, out: Path) -> None:
    row = retraction[retraction.model == TRAINED]
    if row.empty:
        print(f"[plot] no retraction row for {TRAINED}; skipping")
        return
    r = row.iloc[0]
    enrich = float(r["retract_enrichment"])
    lo, hi = float(r["enrichment_ci_low"]), float(r["enrichment_ci_high"])
    ceiling = 1.0 / float(r["fp_base_rate"]) if float(r["fp_base_rate"]) else np.nan

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.bar([0], [enrich], width=0.5, color="#3b6ea5",
           yerr=[[enrich - lo], [hi - enrich]], capsize=6)
    ax.axhline(1.0, color="#b03030", ls="--", lw=1.4)
    ax.text(0.62, 1.0, "1.0 — retraction is noise", color="#b03030", va="bottom", fontsize=9)
    ax.axhline(CORPUS["enrichment"], color="#7a7a7a", ls=":", lw=1.4)
    ax.text(0.62, CORPUS["enrichment"], f"{CORPUS['enrichment']}x — #159 corpus",
            color="#555", va="bottom", fontsize=9)
    if np.isfinite(ceiling):
        ax.axhline(ceiling, color="#2e8b57", ls="-.", lw=1.2)
        ax.text(0.62, ceiling, f"{ceiling:.2f}x — ceiling at this FP rate",
                color="#2e8b57", va="bottom", fontsize=9)
    ax.set_xlim(-0.45, 2.1)
    ax.set_xticks([0])
    ax.set_xticklabels(["trained model\n(own rollouts)"])
    ax.set_ylabel("enrichment  P(FP | retracted) / P(FP)")
    ax.set_title(f"Is retraction discriminative?  {enrich:.2f}x [{lo:.2f}, {hi:.2f}]")

    labels = ["P(FP | retracted)", "P(FP) base rate", "P(retracted | FP)"]
    vals = [float(r["retract_precision_fp"]), float(r["fp_base_rate"]),
            float(r["retract_recall_fp"])]
    ax2.barh(range(3), vals, color=["#3b6ea5", "#a0a0a0", "#8fb8de"])
    for k, v in enumerate(vals):
        ax2.text(v + 0.01, k, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1.12)
    ax2.invert_yaxis()
    ax2.set_title(f"{r['mean_retracts_per_doc']:.1f} retracts/rollout, "
                  f"distance {r['mean_retract_distance']:.1f} statements")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def plot_accuracy(rows: pd.DataFrame, out: Path) -> None:
    models = set(rows.model.unique())
    if not {TRAINED, CONTROL} <= models:
        print(f"[plot] need both arms, have {sorted(models)}; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rng in zip(axes, ("all", "long")):
        sub = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
        wide = sub.pivot_table(index=["dataset", "stem"], columns="model",
                               values="precision").dropna()
        delta = (wide[TRAINED] - wide[CONTROL]).to_numpy()
        n = delta.size
        sem = delta.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ax.hist(delta, bins=41, color="#3b6ea5", alpha=0.85)
        ax.axvline(0, color="#333", lw=1)
        ax.axvline(delta.mean(), color="#b03030", lw=1.6)
        ax.set_xlabel(f"R-precision  {TRAINED} − {CONTROL}  ({rng})")
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
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_retraction(pd.read_csv(a.retraction), a.out_dir / "retraction.png")
    plot_accuracy(pd.read_csv(a.rows), a.out_dir / "accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
