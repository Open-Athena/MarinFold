# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Consensus across rollouts vs single rollouts: does voting beat picking one?

For each target we have 1000 rollouts, each a predicted contact set (`pred`) with
a per-token NLL. This builds two **ensembles** over the rollouts and scores them
(precision/recall/F1, all sep>=6) against ground truth:

  * unweighted vote   — score[pair] = fraction of rollouts containing it,
  * likelihood-weighted vote — weight rollout r by softmax(-nll_per_tok_r / T),
    score[pair] = sum of weights of rollouts containing it.

A predicted set is formed from the vote score three ways:
  * majority      — score >= 0.5                      (label-free)
  * topK@meansize — top-K, K = mean rollout size      (label-free)
  * topK@oracle   — best F1 over all K                (upper bound; needs GT)

…and compared to the single-rollout baselines (mean / min-NLL / oracle).

    uv run python analyze_ensemble.py --run gs://.../runs/calib_ens \
        --targets data/calib_targets.parquet --out-prefix ens_calib
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from build_summary import save_plot_with_meta
except Exception:
    plt = None


def prf(pred: set, gt: set):
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gt)
    p = tp / len(pred)
    r = tp / len(gt) if gt else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def topk_set(scored, k):
    """Top-k pairs by score (scored: list of (score, pair))."""
    k = max(0, int(k))
    return {pr for _, pr in sorted(scored, key=lambda x: -x[0])[:k]}


def best_f1_over_k(scored, gt):
    """Max F1 over all top-k cutoffs (oracle K)."""
    order = [pr for _, pr in sorted(scored, key=lambda x: -x[0])]
    gtn = len(gt)
    best, tp = 0.0, 0
    for i, pr in enumerate(order, 1):
        tp += pr in gt
        p = tp / i
        r = tp / gtn if gtn else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        best = max(best, f)
    return best


def load_run(run, workers=48):
    fs, _ = fsspec.core.url_to_fs(run)
    paths = [p for p in fs.ls(f"{run.rstrip('/')}/rollout_metrics", detail=False)
             if p.endswith(".parquet")]

    def rd(p):
        with fs.open(p, "rb") as fh:
            m = pq.read_table(fh).to_pandas()
        return os.path.basename(p)[: -len(".parquet")], m
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(rd, paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", required=True, help="parquet with entry_id + gt_contacts")
    ap.add_argument("--temps", default="1.0,0.3", help="softmax temps for weighted vote")
    ap.add_argument("--out-prefix", default="ens")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True); os.makedirs(a.plots_dir, exist_ok=True)
    temps = [float(x) for x in a.temps.split(",")]

    with fsspec.open(a.targets, "rb") as fh:
        tg = pq.read_table(fh).to_pandas()
    GT = {r.entry_id: {tuple(sorted(map(int, p))) for p in r.gt_contacts} for r in tg.itertuples()}

    runs = load_run(a.run)
    print(f"{len(runs)} targets", flush=True)

    rows = []
    for entry, m in runs.items():
        gt = GT[entry]
        N = len(m)
        # parse each rollout's predicted pair set from the flattened `pred`
        preds = []
        for flat in m["pred"]:
            f = list(map(int, flat))
            preds.append({(f[i], f[i + 1]) for i in range(0, len(f), 2)})
        sizes = np.array([len(p) for p in preds])
        nllpt = m["nll_per_tok"].to_numpy()

        # single-rollout baselines
        f1s = m["all_f1"].to_numpy()
        single_mean = float(np.nanmean(f1s))
        single_oracle = float(np.nanmax(f1s))
        single_minnll = float(f1s[int(np.nanargmin(nllpt))])

        def vote(weights):
            sc = {}
            for w, p in zip(weights, preds):
                for pr in p:
                    sc[pr] = sc.get(pr, 0.0) + w
            return [(s, pr) for pr, s in sc.items()]

        rec = dict(entry_id=entry, L=int(tg.set_index("entry_id").loc[entry, "L"]),
                   n_gt=len(gt), mean_size=float(sizes.mean()),
                   single_mean=single_mean, single_minnll=single_minnll,
                   single_oracle=single_oracle)

        # unweighted vote (weights 1/N -> score is fraction)
        sc_u = vote(np.ones(N) / N)
        rec["vote_majority_f1"] = prf(topk_set(sc_u, sum(s >= 0.5 for s, _ in sc_u)), gt)[2]
        rec["vote_meanK_f1"] = prf(topk_set(sc_u, round(sizes.mean())), gt)[2]
        p, r, f = prf(topk_set(sc_u, round(sizes.mean())), gt)
        rec["vote_meanK_prec"], rec["vote_meanK_rec"] = p, r
        rec["vote_oracleK_f1"] = best_f1_over_k(sc_u, gt)

        # likelihood-weighted vote at each temperature
        for T in temps:
            w = np.exp(-(nllpt - np.nanmin(nllpt)) / T)
            w = w / w.sum()
            sc_w = vote(w)
            tag = f"wvote_T{T:g}"
            rec[f"{tag}_meanK_f1"] = prf(topk_set(sc_w, round(sizes.mean())), gt)[2]
            rec[f"{tag}_oracleK_f1"] = best_f1_over_k(sc_w, gt)
        rows.append(rec)

    r = pd.DataFrame(rows)
    r.to_csv(f"{a.data_dir}/{a.out_prefix}_per_target.csv", index=False)

    cols = ["single_mean", "single_minnll", "single_oracle",
            "vote_majority_f1", "vote_meanK_f1", "vote_oracleK_f1"]
    cols += [c for c in r.columns if c.startswith("wvote_")]
    print(f"\n=== F1, mean over {len(r)} targets (all sep>=6) ===")
    for c in cols:
        print(f"  {c:<22} {r[c].mean():.3f}")
    print(f"\n  vote_meanK precision/recall: {r.vote_meanK_prec.mean():.3f} / {r.vote_meanK_rec.mean():.3f}")

    if plt is not None:
        order = ["single_mean", "single_minnll", "single_oracle",
                 "vote_majority_f1", "vote_meanK_f1",
                 f"wvote_T{temps[0]:g}_meanK_f1", "vote_oracleK_f1"]
        order = [c for c in order if c in r.columns]
        vals = [r[c].mean() for c in order]
        labels = ["single\nmean", "single\nmin-NLL", "single\noracle",
                  "vote\nmajority", "vote\ntop-K", f"wvote\ntop-K", "vote\noracle-K"][:len(order)]
        colors = ["#bbb", "#6aa", "#088", "#48c", "#26a", "#a4c", "#222"][:len(order)]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(order)), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("F1 (all sep>=6)")
        ax.set_title("Consensus (vote) vs single-rollout selection — mean over targets")
        try:
            save_plot_with_meta(fig, f"{a.plots_dir}/{a.out_prefix}_strategies.png",
                                caption="Per-target F1 by strategy. single-* pick one rollout; "
                                        "vote-* aggregate contact occurrences across 1000 rollouts "
                                        "(majority / top-K@mean-size = label-free; oracle-K needs GT). "
                                        "wvote weights rollouts by softmax(-NLL/token).")
        except Exception:
            fig.savefig(f"{a.plots_dir}/{a.out_prefix}_strategies.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {a.data_dir}/{a.out_prefix}_per_target.csv + plot")


if __name__ == "__main__":
    main()
