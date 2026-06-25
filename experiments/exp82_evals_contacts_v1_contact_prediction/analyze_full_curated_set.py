# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate metrics + timing plot for the full curated-set eval.

Reads the resumable per-protein JSONL from ``eval_full_curated_set.py`` and emits:

* the **pairwise vs rollout+resample** metric table (mean over proteins, overall
  and per dataset), and
* the requested **timing plot** — sequence length (x) vs rollout+resample
  wall-time (y), one point per protein, coloured by dataset — plus a timings CSV.

Run::

    uv run python analyze_full_curated_set.py \
        --results _scratch/eval_full_results.jsonl --plots plots --data data
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

COLS = ["long_P@L", "long_P@L2", "long_P@L5", "long_R", "medlong_P@L", "medlong_R", "P@ngt"]
DEV16 = {  # the FoldBench dev proteins used for method selection (flag, don't drop)
    "7qp5_A", "7tjb_A", "7tlh_A", "7uvg_A", "7xcd_A", "7y54_A", "7ykm_A", "7zu3_A",
    "8adc_A", "8arl_A", "8axj_A", "8bau_A", "8bgb_A", "8cpn_A", "8ec3_A", "8gmy_A",
}


def mean_metric(rows, method, col):
    vals = [r[method].get(col) for r in rows
            if isinstance(r[method].get(col), (int, float)) and not np.isnan(r[method].get(col))]
    return float(np.mean(vals)) if vals else float("nan")


def metric_table(rows, label):
    print(f"\n=== {label}  (n={len(rows)}) ===")
    print(f"{'method':<18} " + " ".join(f"{c:>11}" for c in COLS))
    for method in ("pairwise", "resample"):
        name = "rollout+resample" if method == "resample" else method
        print(f"{name:<18} " + " ".join(f"{mean_metric(rows, method, c):>11.3f}" for c in COLS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--plots", type=Path, default=Path("plots"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.results.open()]
    print(f"{len(rows)} proteins in {args.results}")

    # --- metric tables: overall, per dataset, and held-out-from-dev ---
    metric_table(rows, "ALL curated eval proteins")
    for ds in sorted({r["dataset"] for r in rows}):
        metric_table([r for r in rows if r["dataset"] == ds], f"dataset = {ds}")
    metric_table([r for r in rows if r["stem"] not in DEV16],
                 "held out from dev (excludes the 16 selection proteins)")

    # --- timing summary ---
    L = np.array([r["L"] for r in rows])
    t = np.array([r["t_resample_s"] for r in rows])
    order = np.argsort(L)
    print(f"\n=== rollout+resample timing (n_rollouts={rows[0].get('n_rollouts')}, "
          f"{rows[0].get('gpu')}) ===")
    print(f"  mean {t.mean():.1f}s | median {np.median(t):.1f}s | p95 {np.percentile(t, 95):.1f}s "
          f"| max {t.max():.1f}s (L={L[t.argmax()]}) | total {t.sum() / 3600:.2f} h")

    # --- timing CSV ---
    args.data.mkdir(parents=True, exist_ok=True)
    tcsv = args.data / "eval_full_timings.csv"
    with tcsv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "stem", "L", "n_gt", "t_pairwise_s", "t_resample_s"])
        for r in rows:
            w.writerow([r["dataset"], r["stem"], r["L"], r["n_gt"],
                        r["t_pairwise_s"], r["t_resample_s"]])
    print(f"wrote {tcsv}")

    # --- timing plot: x = sequence length, y = rollout+resample time ---
    args.plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in sorted({r["dataset"] for r in rows}):
        xs = [r["L"] for r in rows if r["dataset"] == ds]
        ys = [r["t_resample_s"] for r in rows if r["dataset"] == ds]
        ax.scatter(xs, ys, s=16, alpha=0.6, label=f"{ds} (n={len(xs)})")
    # quadratic trend (generation ~ linear in L; attention adds curvature)
    coef = np.polyfit(L, t, 2)
    xx = np.linspace(L.min(), L.max(), 200)
    ax.plot(xx, np.polyval(coef, xx), "k--", lw=1.3, alpha=0.7, label="quadratic fit")
    ax.set_xlabel("sequence length  L  (residues)")
    ax.set_ylabel("rollout+resample wall-time  (s)")
    ax.set_title(f"rollout+resample cost vs length  "
                 f"(n_rollouts={rows[0].get('n_rollouts')}, {rows[0].get('gpu')}, batch=24)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    save_plot_with_meta(
        fig, args.plots / "eval_full_resample_time_vs_length.png",
        caption=("rollout+resample wall-time per protein vs sequence length (one point per protein, "
                 "554 curated eval proteins, n_rollouts=100, one A5000). Cost grows ~linearly with L "
                 "(generation length) with mild super-linear curvature from attention."),
        script="analyze_full_curated_set.py",
        args=["--results", "_scratch/eval_full_results.jsonl"], dpi=130,
    )
    print("wrote plots/eval_full_resample_time_vs_length.png")

    # --- headline metrics plot: pairwise vs rollout+resample, long-range, per dataset ---
    groups = ["ALL"] + sorted({r["dataset"] for r in rows})
    grp = lambda g: rows if g == "ALL" else [r for r in rows if r["dataset"] == g]  # noqa: E731
    figm, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax2, col, title in ((axes[0], "long_R", "long-range R-precision"),
                            (axes[1], "long_P@L", "long-range contacts@L")):
        pw = [mean_metric(grp(g), "pairwise", col) for g in groups]
        rs = [mean_metric(grp(g), "resample", col) for g in groups]
        x = np.arange(len(groups))
        ax2.bar(x - 0.2, pw, 0.4, label="pairwise", color="#9aa0a6")
        ax2.bar(x + 0.2, rs, 0.4, label="rollout+resample", color="#2a9d63")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{g}\n(n={len(grp(g))})" for g in groups], fontsize=8)
        ax2.set_title(title)
        ax2.grid(True, axis="y", alpha=0.25)
        ax2.legend(fontsize=9)
    figm.suptitle("Inference algorithm — pairwise vs rollout+resample "
                  "(contacts-v1 1.5B, eval loss 2.7566)")
    save_plot_with_meta(
        figm, args.plots / "eval_full_pairwise_vs_resample.png",
        caption=("Long-range (sep>=24) contact metrics, pairwise vs the dev-selected rollout+resample "
                 "recipe, on the 554-protein curated eval set and per dataset. rollout+resample wins "
                 "every metric on every dataset (see the README table for all bands)."),
        script="analyze_full_curated_set.py",
        args=["--results", "_scratch/eval_full_results.jsonl"], dpi=130,
    )
    print("wrote plots/eval_full_pairwise_vs_resample.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
