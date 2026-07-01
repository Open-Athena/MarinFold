# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage C (local): aggregate an exp100 only-correct run into timing + NLL +
correctness reports.

Reads a run dir (``timings/``, ``nll/``, ``documents/`` under a GCS or local run
root; loaded in parallel) and produces:

  * a throughput summary (gen + score tok/s, s/target, TPU-hours) and an
    overall **correctness check** (every constrained document should be 100%
    precision + full recall),
  * a consolidated per-target table (``data/<prefix>_per_target.csv``),
  * NLL-spread plots (best-of-N vs mean NLL, NLL vs length).

    uv run python aggregate_results.py \
        --run gs://.../exp100_only_correct_contacts_v1_train/runs/full --out-prefix full
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _ls(fs, path):
    try:
        return list(fs.ls(path, detail=False))
    except FileNotFoundError:
        return []


def _read_timing(fs, p):
    with fs.open(p, "rb") as fh:
        return pd.read_csv(fh)


def _read_nll(fs, p):
    entry = os.path.basename(p)[: -len(".parquet")]
    with fs.open(p, "rb") as fh:
        m = pq.read_table(fh).to_pandas()
    ok = ((m["all_prec"] == 1.0) & (m["all_rec"] == 1.0))
    best = m.loc[m["struct_nll"].idxmin()]
    return dict(
        entry_id=entry, n_rollouts=len(m), n_correct=int(ok.sum()),
        n_contacts=int(m["n_contacts"].max()),
        struct_nll_min=float(m["struct_nll"].min()),
        struct_nll_mean=float(m["struct_nll"].mean()),
        struct_nll_max=float(m["struct_nll"].max()),
        struct_nll_per_tok_min=float(m["struct_nll_per_tok"].min()),
        selected_r=int(best["r"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GCS or local run dir")
    ap.add_argument("--out-prefix", default="full")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(a.run)
    run = a.run.rstrip("/")

    tpaths = [p for p in _ls(fs, f"{run}/timings") if p.endswith(".csv")]
    npaths = [p for p in _ls(fs, f"{run}/nll") if p.endswith(".parquet")]
    print(f"{len(tpaths)} timings, {len(npaths)} nll", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        tim = pd.concat(list(ex.map(lambda p: _read_timing(fs, p), tpaths)), ignore_index=True)
        nll = pd.DataFrame(list(ex.map(lambda p: _read_nll(fs, p), npaths)))

    df = tim.merge(nll, on="entry_id", suffixes=("", "_n"))
    os.makedirs(a.data_dir, exist_ok=True)
    csv_path = f"{a.data_dir}/{a.out_prefix}_per_target.csv"
    df.sort_values("entry_id").to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)", flush=True)

    # ---- throughput + correctness summary ----
    gen_tok = tim["total_gen_tokens"].sum()
    gen_s = tim["gen_seconds"].sum()
    score_s = tim["score_seconds"].sum() if "score_seconds" in tim else float("nan")
    tp = int(tim["tensor_parallel"].iloc[0]) if "tensor_parallel" in tim else 1
    n_correct = int(nll["n_correct"].sum())
    n_rollouts = int(nll["n_rollouts"].sum())
    print("\n==== exp100 summary ====")
    print(f"targets:            {len(nll)}")
    print(f"rollouts:           {n_rollouts}")
    print(f"100%-correct:       {n_correct}/{n_rollouts} "
          f"({100*n_correct/max(n_rollouts,1):.2f}%)  <-- must be 100%")
    print(f"gen tokens:         {gen_tok:,}")
    print(f"gen tok/s (tp={tp}):  {gen_tok/gen_s:,.0f}" if gen_s else "")
    print(f"gen TPU-hours:      {gen_s/3600:.2f}  (+score {score_s/3600:.2f})")
    print(f"struct NLL best-of-N: mean {nll['struct_nll_min'].mean():.2f}  "
          f"median {nll['struct_nll_min'].median():.2f}")
    print(f"struct NLL spread (mean-min): {(nll['struct_nll_mean']-nll['struct_nll_min']).mean():.2f}")

    if plt is None:
        print("matplotlib unavailable; skipping plots", flush=True)
        return 0
    os.makedirs(a.plots_dir, exist_ok=True)

    # best-of-N vs mean structure NLL per contact (length-normalized for fairness)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["L"], df["struct_nll_min"] / df["n_contacts"].clip(lower=1) / 3,
               s=8, alpha=0.4, label="best-of-N")
    ax.scatter(df["L"], df["struct_nll_mean"] / df["n_contacts"].clip(lower=1) / 3,
               s=8, alpha=0.4, label="mean")
    ax.set_xlabel("L (residues)"); ax.set_ylabel("structure NLL / contact-token")
    ax.set_title("exp100 only-correct NLL vs length"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{a.plots_dir}/{a.out_prefix}_nll_vs_L.png", dpi=130)
    print(f"wrote {a.plots_dir}/{a.out_prefix}_nll_vs_L.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
