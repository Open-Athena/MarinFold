# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does the model's own confidence (NLL) predict a rollout's accuracy?

For the train-on-rollouts idea we need to pick high-accuracy rollouts **without**
ground truth. The natural selector is the model's likelihood. This reads a run's
per-rollout metrics (needs `nll` / `nll_per_tok`, captured by the worker's
`logprobs=1` path) and asks, per target:

  * Spearman corr between length-normalized NLL and accuracy (within-target, so
    length is controlled),
  * how good is the **min-NLL** rollout vs the oracle-best and the mean rollout
    (i.e. can confidence-based selection recover the good rollouts?).

    uv run python analyze_nll.py --run gs://.../runs/calib_nll --out-prefix nll_calib
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def spearman(x, y) -> float:
    """Spearman rho without scipy: Pearson on ranks."""
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from build_summary import save_plot_with_meta
except Exception:
    plt = None


def load(run, workers=48):
    fs, _ = fsspec.core.url_to_fs(run)
    paths = [p for p in fs.ls(f"{run.rstrip('/')}/rollout_metrics", detail=False)
             if p.endswith(".parquet")]

    def rd(p):
        with fs.open(p, "rb") as fh:
            m = pq.read_table(fh).to_pandas()
        m["entry_id"] = os.path.basename(p)[: -len(".parquet")]
        return m
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(rd, paths)), ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="all_f1", help="accuracy column to relate to NLL")
    ap.add_argument("--out-prefix", default="nll")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True); os.makedirs(a.plots_dir, exist_ok=True)

    df = load(a.run)
    if "nll_per_tok" not in df or df["nll_per_tok"].isna().all():
        raise SystemExit("no nll_per_tok in this run — re-generate with the logprobs=1 worker")
    print(f"{df.entry_id.nunique()} targets, {len(df)} rollouts", flush=True)

    rows = []
    for e, g in df.groupby("entry_id"):
        g = g.dropna(subset=["nll_per_tok", a.metric])
        if len(g) < 10:
            continue
        rho = spearman(g["nll_per_tok"], g[a.metric])
        # selection strategies (per target)
        oracle = g[a.metric].max()
        mean_ = g[a.metric].mean()
        minnll = g.loc[g["nll_per_tok"].idxmin(), a.metric]          # most confident rollout
        # top-k by confidence, take best within the top-k (a realistic "sample k, keep best")
        top10 = g.nsmallest(max(1, len(g)//100), "nll_per_tok")[a.metric].max()  # best of most-confident 1%
        rows.append(dict(entry_id=e, L=int(g["n_gen_tokens"].median()), n=len(g),
                         rho=rho, oracle=oracle, mean=mean_, min_nll=minnll, best_of_top1pct=top10))
    r = pd.DataFrame(rows)
    r.to_csv(f"{a.data_dir}/{a.out_prefix}_per_target.csv", index=False)

    print(f"\n=== NLL vs {a.metric} (within-target Spearman rho; negative = more "
          f"confident rollouts are more accurate) ===")
    print(f"  mean rho   : {r.rho.mean():+.3f}   median {r.rho.median():+.3f}")
    print(f"  rho<0 (confidence helps) in {100*(r.rho<0).mean():.0f}% of targets")
    print(f"\n=== selection ({a.metric}, mean over {len(r)} targets) ===")
    print(f"  mean rollout (== random pick) : {r['mean'].mean():.3f}")
    print(f"  min-NLL rollout (most confident): {r['min_nll'].mean():.3f}")
    print(f"  best of most-confident 1%      : {r['best_of_top1pct'].mean():.3f}")
    print(f"  oracle (best of all 1000)      : {r['oracle'].mean():.3f}")
    gap = (r['min_nll'].mean() - r['mean'].mean()) / (r['oracle'].mean() - r['mean'].mean() + 1e-9)
    print(f"  -> min-NLL closes {100*gap:.0f}% of the mean->oracle gap")

    if plt is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        # pooled scatter, per-target z-scored so lengths/targets overlay
        d = df.dropna(subset=["nll_per_tok", a.metric]).copy()
        d["nll_z"] = d.groupby("entry_id")["nll_per_tok"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        ax[0].scatter(d["nll_z"], d[a.metric], s=3, alpha=.1)
        ax[0].set_xlabel("per-target z-scored NLL/token  (left = more confident)")
        ax[0].set_ylabel(a.metric); ax[0].set_title("rollout accuracy vs confidence (all targets)")
        ax[1].hist(r.rho, bins=20, color="steelblue", edgecolor="white")
        ax[1].axvline(r.rho.mean(), color="crimson", ls="--", label=f"mean {r.rho.mean():+.2f}")
        ax[1].set_xlabel(f"within-target Spearman(NLL/tok, {a.metric})")
        ax[1].set_ylabel("# targets"); ax[1].legend(); ax[1].set_title("does confidence track accuracy?")
        try:
            save_plot_with_meta(fig, f"{a.plots_dir}/{a.out_prefix}_nll_vs_acc.png",
                                caption=f"Left: rollout {a.metric} vs per-target z-scored NLL/token. "
                                        f"Right: per-target Spearman(NLL/tok, {a.metric}); negative = "
                                        "confident rollouts are more accurate.")
        except Exception:
            fig.savefig(f"{a.plots_dir}/{a.out_prefix}_nll_vs_acc.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {a.data_dir}/{a.out_prefix}_per_target.csv + plot")


if __name__ == "__main__":
    main()
