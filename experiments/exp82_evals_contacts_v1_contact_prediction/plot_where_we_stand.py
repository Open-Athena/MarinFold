# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Project-README "where we stand" figure — contact R-precision (n=554).

A single aggregate boxplot panel (range = all, sep>=6) putting MarinFold's best
inference — **rollout+resample**, n=100 — next to the structure predictors, for
two checkpoints: the #61/#75 sweep winner the project has been quoting, and the
current best model from Eric's #117 sweep.

Reads exp89's committed per-protein table for the structure predictors and the
rollout rows emitted by ``build_rollout_rows.py`` for the MarinFold bars; both
are scored by exp89's metric implementation, so every bar is comparable.
Style follows exp89 ``plot.py``'s ``plot_single_panel``.

    uv run python plot_where_we_stand.py \
        --exp89-csv <exp89>/data/contact_precision_all.csv \
        --rollout-csv data/where_we_stand_rows.csv.gz \
        --out plots/where_we_stand_rprecision.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Two MarinFold bars (same rollout recipe, different checkpoints) + the structure
# predictors. Oranges for MarinFold so the two read as one group.
CONFIGS = [
    ("marinfold-cv1-exp75-rollout", "single_seq", "lm",
     "MarinFold #61\nn=100 rollouts", "#7f2704"),
    ("marinfold-cv1-exp117-rollout", "single_seq", "lm",
     "MarinFold #117 best\nn=100 rollouts", "#e6550d"),
    ("protenix-v2", "single_seq", "structure", "Protenix-v2 · SS", "#9ecae1"),
    ("protenix-v2", "msa", "structure", "Protenix-v2 · MSA", "#2171b5"),
    ("esmfold", "single_seq", "structure", "ESMFold", "#74c476"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "#238b45"),
]


def _vals(df, model, mode, predictor):
    s = df[(df["model"] == model) & (df["mode"] == mode) & (df["predictor"] == predictor)]
    return s["precision"].to_numpy(dtype=float)


def main(exp89_csv: Path, rollout_csv: Path, out: Path) -> None:
    df = pd.concat([pd.read_csv(exp89_csv), pd.read_csv(rollout_csv)], ignore_index=True)
    sub = df[(df["cut"] == "R") & (df["range"] == "all")]
    labels = [c[3] for c in CONFIGS]
    palette = {c[3]: c[4] for c in CONFIGS}
    rows, means, counts = [], {}, {}
    for model, mode, pred, disp, _ in CONFIGS:
        v = _vals(sub, model, mode, pred)
        v = v[np.isfinite(v)]
        rows += [(disp, x) for x in v]
        means[disp] = float(v.mean()) if v.size else float("nan")
        counts[disp] = int(v.size)
    n_set = sorted(set(counts.values()))
    if len(n_set) != 1:
        print(f"!! uneven protein counts across bars: {counts}")
    bdf = pd.DataFrame(rows, columns=["cfg", "precision"])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.boxplot(data=bdf, x="cfg", y="precision", order=labels, hue="cfg", palette=palette, ax=ax,
                width=0.6, legend=False, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
                flierprops=dict(marker=".", markersize=2, markerfacecolor="0.4", markeredgecolor="none", alpha=0.35),
                medianprops=dict(color="black", linewidth=1.4), boxprops=dict(alpha=0.85), linewidth=1.0)
    for xi, disp in enumerate(labels):
        if not np.isnan(means[disp]):
            ax.text(xi, 1.03, f"{means[disp]:.2f}", ha="center", va="bottom", fontsize=9)
    for t in ax.get_xticklabels():
        t.set_rotation(18); t.set_horizontalalignment("right"); t.set_fontsize(9)
    ax.set_xlabel("")
    ax.set_ylabel("R-precision", fontsize=11)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Contact R-precision  (n={n_set[-1]})", fontsize=13)
    fig.text(0.5, 0.005, "box = median & IQR · whiskers = 1.5×IQR · ◆ = mean · points = outliers",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(
        {"script": "plot_where_we_stand.py",
         "args": ["--exp89-csv", str(exp89_csv), "--rollout-csv", str(rollout_csv)],
         "means": means, "n": counts,
         "caption": ("Contact R-precision (all sep>=6, n=554): MarinFold rollout+resample "
                     "(n=100, temperature 1.0 / top-p 0.95 / no top-k) for the #61/#75 model "
                     "and the current #117 sweep best, vs Protenix-v2 / ESMFold / ESMFold2.")},
        indent=2))
    print(f"wrote {out}  means: " + ", ".join(f"{k.replace(chr(10), ' ')}={v:.3f}"
                                              for k, v in means.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp89-csv", type=Path, required=True)
    ap.add_argument("--rollout-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    main(args.exp89_csv, args.rollout_csv, args.out)
