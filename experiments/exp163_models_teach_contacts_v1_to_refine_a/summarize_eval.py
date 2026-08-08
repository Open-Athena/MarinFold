# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the exp163 Phase-3 eval shards into the headline tables (issue #163).

Reads ``eval554/scores/<model>/shard-*.parquet`` (written by
``eval_refiner_worker.py``) and reports, per model and per band, the
candidate-context sweep:

* ``K0``            — no candidates (the one-shot readout, and the control every
  other arm is compared against)
* ``K1 … K16``      — raw candidate blocks, with the realized contact count so an
  out-of-distribution context is visible rather than silent
* ``consensus``     — one high-precision voted block

and the paired base-vs-refiner deltas that the success criteria are stated in.

Everything is **paired**: only proteins present for both models and all arms are
used, so a delta is never an artifact of differing protein sets.

    uv run python summarize_eval.py --scores s3://…/eval554/scores --out data/
"""
from __future__ import annotations

import argparse
import os
import re

import fsspec
import numpy as np
import pandas as pd

S3_KW = {"s3": {"addressing_style": "virtual"}}


def _fs(url: str):
    if not url.startswith("s3://"):
        return fsspec.filesystem("file")
    return fsspec.filesystem("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                             config_kwargs=S3_KW)


def _strip(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


def load(scores_url: str) -> dict[str, pd.DataFrame]:
    fs = _fs(scores_url)
    out: dict[str, pd.DataFrame] = {}
    for d in fs.ls(_strip(scores_url).rstrip("/"), detail=False):
        model = d.rstrip("/").split("/")[-1]
        parts = sorted(fs.glob(f"{d.rstrip('/')}/*.parquet"))
        if not parts:
            continue
        frames = []
        for p in parts:
            with fs.open(p, "rb") as fh:
                frames.append(pd.read_parquet(fh))
        out[model] = pd.concat(frames, ignore_index=True)
        print(f"  {model}: {len(out[model])} proteins from {len(parts)} shard(s)", flush=True)
    return out


def ks_in(df: pd.DataFrame) -> list[int]:
    return sorted(int(m.group(1)) for c in df.columns
                  if (m := re.fullmatch(r"Rraw(\d+)_all", c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default=None, help="directory for the per-protein CSVs")
    a = ap.parse_args()

    print(f"[exp163] reading {a.scores}", flush=True)
    models = load(a.scores)
    if not models:
        raise SystemExit("no score shards found")

    # pair on the protein sets shared by every model
    common = set.intersection(*(set(d.entry_id) for d in models.values()))
    print(f"\n[exp163] {len(common)} proteins common to all {len(models)} model(s)", flush=True)
    models = {m: d[d.entry_id.isin(common)].sort_values("entry_id").reset_index(drop=True)
              for m, d in models.items()}

    Ks = ks_in(next(iter(models.values())))
    for band in ("all", "long"):
        print(f"\n=== {band} band — R-precision (paired, n={len(common)}) ===")
        hdr = f"{'model':>9} {'K0':>8} " + " ".join(f"{'K'+str(k):>8}" for k in Ks) + f" {'consensus':>10}"
        print(hdr)
        for m, d in models.items():
            cells = " ".join(f"{d[f'Rraw{k}_{band}'].mean():8.4f}" for k in Ks)
            print(f"{m:>9} {d[f'R0_{band}'].mean():8.4f} {cells} {d[f'Rcons_{band}'].mean():10.4f}")
        # realized candidate size, so an out-of-distribution context is visible
        d0 = next(iter(models.values()))
        sizes = " ".join(f"{d0[f'ncon{k}'].mean():8.0f}" for k in Ks)
        print(f"{'contacts':>9} {0:8d} {sizes} {d0['cons_n'].mean():10.0f}")

    print("\n=== deltas vs each model's own K0 (all band) ===")
    for m, d in models.items():
        best_k = max(Ks, key=lambda k: d[f"Rraw{k}_all"].mean())
        dk = d[f"Rraw{best_k}_all"].mean() - d["R0_all"].mean()
        dc = d["Rcons_all"].mean() - d["R0_all"].mean()
        win_c = float((d["Rcons_all"] > d["R0_all"]).mean())
        print(f"  {m:>9}: best raw K={best_k:<3} d={dk:+.4f} | consensus d={dc:+.4f} "
              f"(beats own K0 on {win_c:.0%} of proteins)")

    if "base" in models and "refiner" in models:
        b, r = models["base"], models["refiner"]
        print("\n=== refiner vs base, paired (all band) ===")
        for arm, col in [("K0", "R0_all"), ("consensus", "Rcons_all")] + \
                        [(f"K{k}", f"Rraw{k}_all") for k in Ks]:
            diff = r[col].values - b[col].values
            print(f"  {arm:>10}: refiner {r[col].mean():.4f}  base {b[col].mean():.4f}  "
                  f"d={diff.mean():+.4f}  refiner wins {np.mean(diff > 0):.0%}")
        print(f"\n  consensus-block precision: {r['cons_prec'].mean():.3f} "
              f"(mean {r['cons_n'].mean():.0f} contacts)")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for m, d in models.items():
            p = os.path.join(a.out, f"eval554_{m}.csv")
            d.to_csv(p, index=False)
            print(f"\n[exp163] wrote {p}")


if __name__ == "__main__":
    main()
