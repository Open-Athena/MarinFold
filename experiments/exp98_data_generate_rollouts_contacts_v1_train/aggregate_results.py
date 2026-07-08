# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage C (local): aggregate exp98 rollout runs into timing + accuracy reports.

Reads one or more run dirs (``timings/``, ``rollout_metrics/``, ``best_rollouts/``
under a GCS or local run root; loaded in parallel) and produces:

  * a throughput summary + projection (tokens/s, s/target, TPU-hours),
  * a consolidated per-target table (``data/<prefix>_per_target.csv`` + a copy
    pushed to GCS for the Colab explorer),
  * accuracy/throughput plots for the README / summary.pdf.

    uv run python aggregate_results.py \
        --runs full=gs://.../runs/full --out-prefix full \
        --gcs-summary gs://.../runs/full/per_target_summary.parquet
"""
from __future__ import annotations

import argparse
import json
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


def _ls(fs, path):
    try:
        return [p for p in fs.ls(path, detail=False)]
    except FileNotFoundError:
        return []


def _read_metrics(fs, p):
    entry = os.path.basename(p)[: -len(".parquet")]
    with fs.open(p, "rb") as fh:
        m = pq.read_table(fh).to_pandas()
    return dict(
        entry_id=entry, n_rollouts=len(m),
        mean_prec=m["all_prec"].mean(), mean_rec=m["all_rec"].mean(),
        mean_f1=m["all_f1"].mean(),
        max_rec=m["all_rec"].max(), max_f1=m["all_f1"].max(),
        mean_long_rec=m["long_rec"].mean(skipna=True),
        mean_gen_tokens=m["n_gen_tokens"].mean(),
        frac_finished=m["finished"].mean(),
    )


def _read_best(fs, p):
    with fs.open(p, "r") as fh:
        d = json.load(fh)
    return dict(
        entry_id=d["entry_id"], L=d["L"], n_gt=d["n_gt"],
        best_f1=d["best_f1"]["f1"], best_f1_prec=d["best_f1"]["precision"],
        best_f1_rec=d["best_f1"]["recall"],
        best_recall=d["best_recall"]["recall"],
        best_recall_prec=d["best_recall"]["precision"],
        best_recall_f1=d["best_recall"]["f1"],
    )


def _read_csv(fs, p):
    with fs.open(p, "r") as fh:
        return pd.read_csv(fh)


def load_run(run_root: str, workers: int = 48) -> dict:
    fs, _ = fsspec.core.url_to_fs(run_root)
    root = run_root.rstrip("/")
    tpaths = [p for p in _ls(fs, f"{root}/timings") if p.endswith(".csv")]
    mpaths = [p for p in _ls(fs, f"{root}/rollout_metrics") if p.endswith(".parquet")]
    bpaths = [p for p in _ls(fs, f"{root}/best_rollouts") if p.endswith(".json")]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        timings = pd.concat(list(ex.map(lambda p: _read_csv(fs, p), tpaths)),
                            ignore_index=True) if tpaths else pd.DataFrame()
        per_target = pd.DataFrame(list(ex.map(lambda p: _read_metrics(fs, p), mpaths)))
        best = pd.DataFrame(list(ex.map(lambda p: _read_best(fs, p), bpaths)))
    return dict(timings=timings, per_target=per_target, best=best)


def throughput_summary(name, timings, full_targets=1000, full_rollouts=1000):
    if timings.empty:
        return {}
    tot_tok = timings["total_gen_tokens"].sum()
    tot_gen_s = timings["gen_seconds"].sum()
    n_roll = timings["n_rollouts"].sum()
    tps = tot_tok / tot_gen_s
    mean_tok_per_roll = tot_tok / n_roll
    full_count = full_targets * full_rollouts
    proj_tokens = mean_tok_per_roll * full_count
    proj_gen_s_1tpu = proj_tokens / tps
    return dict(
        run=name, n_targets=len(timings), n_rollouts=int(n_roll),
        total_gen_tokens=int(tot_tok), sum_gen_seconds=round(tot_gen_s, 1),
        tokens_per_s=round(tps, 1), mean_tok_per_rollout=round(mean_tok_per_roll, 1),
        proj_1tpu_hours=round(proj_gen_s_1tpu / 3600, 2),
        proj_8tpu_hours=round(proj_gen_s_1tpu / 3600 / 8, 2),
    )


def consolidate(r, name):
    t = r["timings"].copy()
    if not r["best"].empty:
        # timings already carries L / n_gt; drop the duplicates from best.
        best = r["best"].drop(columns=[c for c in ("L", "n_gt") if c in r["best"]])
        t = t.merge(best, on="entry_id", how="left")
    acc = ["entry_id", "mean_prec", "mean_rec", "mean_f1", "max_rec", "max_f1", "mean_long_rec"]
    if not r["per_target"].empty:
        t = t.merge(r["per_target"][acc], on="entry_id", how="left")
    t.insert(0, "run", name)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_root pairs")
    ap.add_argument("--out-prefix", default="full")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--gcs-summary", default=None,
                    help="also write the consolidated per-target table here (for Colab)")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    runs = {}
    for spec in args.runs:
        name, root = spec.split("=", 1)
        print(f"loading {name} from {root} ...", flush=True)
        runs[name] = load_run(root)

    summ = pd.DataFrame([throughput_summary(n, r["timings"]) for n, r in runs.items()])
    print("\n=== THROUGHPUT ===")
    print(summ.to_string(index=False))
    summ.to_csv(f"{args.data_dir}/{args.out_prefix}_throughput.csv", index=False)

    consolidated = {}
    for name, r in runs.items():
        t = consolidate(r, name)
        consolidated[name] = t
        t.to_csv(f"{args.data_dir}/{args.out_prefix}_{name}_per_target.csv", index=False)
        # accuracy headline over all targets
        print(f"\n=== {name}: accuracy over {len(t)} targets (all sep>=6) ===")
        for col in ["best_recall", "best_f1", "max_rec", "mean_rec"]:
            if col in t:
                s = t[col].describe()[["mean", "50%", "max"]]
                print(f"  {col:<12} mean={s['mean']:.3f} median={s['50%']:.3f} max={s['max']:.3f}")
        if args.gcs_summary and name == list(runs)[0]:
            with fsspec.open(args.gcs_summary, "wb") as fh:
                import pyarrow as pa
                pq.write_table(pa.Table.from_pandas(t, preserve_index=False), fh)
            print(f"  wrote consolidated table -> {args.gcs_summary}")

    if plt is not None:
        main_name = list(runs)[0]
        t = consolidated[main_name]
        # 1. histograms of best-of-N accuracy
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, title in ((axes[0], "best_recall", "best recall"),
                               (axes[1], "best_f1", "best F1")):
            ax.hist(t[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            ax.axvline(t[col].mean(), color="crimson", ls="--", label=f"mean {t[col].mean():.2f}")
            ax.set_xlabel(f"{title} (best of 1000 rollouts)"); ax.set_ylabel("# targets"); ax.legend()
        fig.suptitle(f"Best-of-1000 accuracy across {len(t)} targets (all sep>=6)")
        _save(fig, f"{args.plots_dir}/{args.out_prefix}_best_hist.png",
              "Distribution of per-target best-of-1000 recall and F1 (all contacts, sep>=6).")
        # 2. best accuracy vs L
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t["L"], t["best_f1"], s=10, alpha=0.4, label="best F1")
        ax.scatter(t["L"], t["best_recall"], s=10, alpha=0.4, label="best recall")
        ax.set_xlabel("L (residues)"); ax.set_ylabel("best of 1000 (all sep>=6)")
        ax.set_title("Best rollout accuracy vs length"); ax.legend()
        _save(fig, f"{args.plots_dir}/{args.out_prefix}_bestacc_vs_L.png",
              "Per-target best-of-1000 recall/F1 vs protein length.")
        # 3. best F1 vs n_gt
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(t["n_gt"], t["best_f1"], c=t["L"], s=10, alpha=0.5, cmap="viridis")
        fig.colorbar(sc, label="L"); ax.set_xlabel("# ground-truth contacts")
        ax.set_ylabel("best F1 (of 1000)"); ax.set_title("Best F1 vs contact count")
        _save(fig, f"{args.plots_dir}/{args.out_prefix}_bestf1_vs_ngt.png",
              "Per-target best-of-1000 F1 vs number of ground-truth contacts (colored by L).")
        # 4. throughput vs L
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t["L"], t["tokens_per_s"], s=10, alpha=0.4, color="darkorange")
        ax.set_xlabel("L (residues)"); ax.set_ylabel("generation tokens/s")
        ax.set_title("Per-target throughput vs length (tp=4 v5p-8)")
        _save(fig, f"{args.plots_dir}/{args.out_prefix}_throughput_vs_L.png",
              "Per-target generation throughput (tokens/s) vs length.")
    print(f"\nwrote {args.data_dir}/{args.out_prefix}_*.csv + {args.plots_dir}/{args.out_prefix}_*.png")


def _save(fig, path, caption):
    try:
        save_plot_with_meta(fig, path, caption=caption)
    except Exception:
        fig.savefig(path, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
