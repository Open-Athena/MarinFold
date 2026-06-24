# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot contact R-precision vs final eval loss across the #75 contacts-v1 1.5B
sweep checkpoints (E1/E2/E4/E8), to see how contact accuracy scales with the
training objective. Single-realization scoring (no ensembling).

Reads ``contact_precision_all.csv`` (after compute_metrics has scored E1/E2/E4
alongside the already-present E8 = ``marinfold-contacts-v1``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (model key in the CSV, epoch label, lr/wd cell, final eval/contacts-v1-val/loss)
SWEEP = [
    ("marinfold-cv1-e1", "E1", "7e-4 / wd0.05", 3.046),
    ("marinfold-cv1-e2", "E2", "7e-4 / wd0.8", 2.942),
    ("marinfold-cv1-e4", "E4", "1e-3 / wd0.05", 2.924),
    ("marinfold-contacts-v1", "E8", "1e-3 / wd0.2", 2.757),
]


def mean_rprec(df, model, rng):
    s = df[(df.model == model) & (df["mode"] == "single_seq") & (df.predictor == "lm")
           & (df.cut == "R") & (df["range"] == rng)]
    return float(s.precision.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision-csv", type=Path, default=Path("data/contact_precision_all.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots/loss_vs_rprecision.png"))
    args = ap.parse_args()
    df = pd.read_csv(args.precision_csv)

    loss = np.array([s[3] for s in SWEEP])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for rng, color, marker in [("all", "#e6550d", "o"), ("long", "#2171b5", "s")]:
        y = np.array([mean_rprec(df, s[0], rng) for s in SWEEP])
        ax.plot(loss, y, marker=marker, color=color, lw=1.8, ms=8, label=f"R-precision ({rng})")
        for (key, ep, cell, lo), yy in zip(SWEEP, y):
            ax.annotate(f"{ep}", (lo, yy), textcoords="offset points", xytext=(6, 6),
                        fontsize=9, color=color, fontweight="bold")
    ax.invert_xaxis()  # lower loss (more trained / better) to the right
    ax.set_xlabel("final eval/contacts-v1-val/loss  (lower = better →)", fontsize=11)
    ax.set_ylabel("contact R-precision  (n=554)", fontsize=11)
    ax.set_title("Contact R-precision vs eval loss — contacts-v1 1.5B (#75 sweep)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.text(0.5, 0.005, "points = per-epoch sweep winners (E1/E2/E4/E8); single-realization scoring, no ensembling",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(
        {"script": "loss_vs_rprecision.py", "caption":
         "Contact R-precision (aggregate + long-range, n=554) vs final contacts-v1-val eval loss "
         "for the #75 sweep per-epoch winners E1/E2/E4/E8. Single-realization, no ensembling."}, indent=2))
    plt.close(fig)
    print("R-precision by checkpoint:")
    for key, ep, cell, lo in SWEEP:
        print(f"  {ep} (loss {lo}, {cell}):  R/all={mean_rprec(df,key,'all'):.3f}  R/long={mean_rprec(df,key,'long'):.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
