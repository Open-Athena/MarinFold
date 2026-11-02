# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""The honest draft-quality table, over the COMPLETE rollout set.

Two provisional numbers preceded this and both were biased, in opposite
directions: a 6-protein local smoke read per-rollout precision 0.411, and the
first parts off the cluster read 0.297.  Neither is quotable. Each shard walks
its targets in **ascending length**, so early parts are the shortest proteins in
the pool -- where there are few valid pairs and the model over-generates -- while
the smoke was simply too small to mean anything (#163: a 40-protein probe read
+0.048 where the full 553 read +0.065).

So this reports by **length band and by arm**, not just a pooled mean, because
the pooled mean is exactly the statistic the sampling order distorts.

It also reports the **best-of-K spread** per protein -- how much better a
protein's best draft is than its average one.  That spread, not the mean, is what
the multi-draft format is built to exploit and what a best-of-N reward would be
paid out of.

    uv run python summarize_rollouts.py --rollouts gs://.../rollouts --out data/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load(uri: str, limit_files: int | None = None) -> pd.DataFrame:
    import fsspec

    fs = fsspec.core.url_to_fs(uri)[0]
    scheme = uri.split("://", 1)[0] if "://" in uri else None
    paths = sorted(fs.glob(f"{uri.rstrip('/')}/*.parquet"))
    if limit_files:
        paths = paths[:limit_files]
    if not paths:
        raise SystemExit(f"no rollout parquets under {uri}")
    frames = []
    for p in paths:
        full = p if (scheme is None or "://" in p) else f"{scheme}://{p}"
        with fs.open(full, "rb") as fh:
            frames.append(pq.read_table(
                fh, columns=["target_id", "arm", "r", "L", "n_gt", "n_pred", "tp",
                             "precision", "recall", "f1", "finished", "n_gen_tokens",
                             "pred"]).to_pandas())
    print(f"[rollouts] {len(paths)} parts", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--limit-files", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    d = load(a.rollouts, a.limit_files)
    print(f"[rollouts] {len(d):,} rollouts over {d.target_id.nunique():,} proteins")

    overall = {
        "n_rollouts": int(len(d)),
        "n_proteins": int(d.target_id.nunique()),
        "precision": float(d.precision.mean()),
        "recall": float(d.recall.mean()),
        "f1": float(d.f1.mean()),
        "finished": float(d.finished.mean()),
        "pred_over_gt": float((d.n_pred / d.n_gt.clip(lower=1)).mean()),
        "L_median": float(d.L.median()),
    }
    print("\n== overall ==")
    for k, v in overall.items():
        print(f"  {k:16s} {v:,.4f}" if isinstance(v, float) else f"  {k:16s} {v:,}")

    band = pd.cut(d.L, [0, 100, 150, 200, 300, 400, 512], include_lowest=True)
    by_band = (d.groupby(band, observed=True)
                 .agg(n=("r", "size"), proteins=("target_id", "nunique"),
                      precision=("precision", "mean"), recall=("recall", "mean"),
                      f1=("f1", "mean"), finished=("finished", "mean"),
                      pred_over_gt=("n_pred", "mean"), tokens=("n_gen_tokens", "mean"))
                 .round(4).reset_index().rename(columns={"L": "L_band"}))
    by_arm = (d.groupby("arm")
                .agg(n=("r", "size"), proteins=("target_id", "nunique"),
                     L_median=("L", "median"), precision=("precision", "mean"),
                     recall=("recall", "mean"), f1=("f1", "mean"),
                     finished=("finished", "mean"))
                .round(4).reset_index())
    print("\n== by length band ==\n" + by_band.to_string(index=False))
    print("\n== by arm ==\n" + by_arm.to_string(index=False))

    # Per-protein best-of-K: the spread the multi-draft format is meant to exploit.
    per_prot = d.groupby("target_id").agg(best_f1=("f1", "max"), mean_f1=("f1", "mean"),
                                          k=("r", "size"))
    overall["best_of_k_f1"] = float(per_prot.best_f1.mean())
    overall["mean_f1_per_protein"] = float(per_prot.mean_f1.mean())
    overall["best_minus_mean"] = overall["best_of_k_f1"] - overall["mean_f1_per_protein"]
    print(f"\n== spread ==\n  best-of-K F1 {overall['best_of_k_f1']:.4f} vs mean "
          f"{overall['mean_f1_per_protein']:.4f}  (+{overall['best_minus_mean']:.4f})")
    print("  #163's E8 drafts: oracle best-of-16 nearly doubled a single rollout.")

    by_band.to_csv(a.out / "rollout_quality_by_band.csv", index=False)
    by_arm.to_csv(a.out / "rollout_quality_by_arm.csv", index=False)
    (a.out / "rollout_quality.json").write_text(json.dumps(overall, indent=2))
    print(f"\n[rollouts] wrote {a.out}/rollout_quality*.{{csv,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
