# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired per-protein comparison of two scored checkpoints — issue #237.

`score_agg_modes.py` reports means. A mean is not enough here: the differences
that decide this experiment are on the order of #204's **0.0023** noise floor,
and the arms are scored on the *same* proteins, so the paired difference has far
less variance than either mean. #230 used a paired per-protein bootstrap for
exactly this reason, and this reproduces it against the per-rollout parquets both
labels already write.

Reported per (cut, aggregation mode):

* Δ = arm − reference, averaged over proteins after averaging over that
  protein's rollouts, so a protein with more rollouts does not get more weight;
* a 95 % CI from 10,000 paired bootstrap resamples **over proteins**;
* the win/loss split, because #230 found a case where the magnitude sat at the
  noise floor while the *direction* was consistent enough to be significant —
  a small consistent cost is a different thing from noise, and only the split
  separates them.

    python compare_arms.py --arm data/agg_modes_m_c_step72_per_rollout.parquet \\
        --ref data/agg_modes_exp230_step1988_per_rollout.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CUTS = {
    "legacy554": lambda d: d["in_legacy554"],
    "eval2": lambda d: d["in_eval2"],
    "eval2_natural": lambda d: d["in_eval2"] & ~d["designed_any"],
    "eval2_lt30": lambda d: d["in_eval2"] & d["passes_30"],
}
MODES = ("consensus", "best", "last", "second_last")
N_BOOT = 10_000
SEED = 237


def per_protein(df: pd.DataFrame, mode: str, col: str) -> pd.Series:
    d = df[df["mode"] == mode]
    return d.groupby(["dataset", "stem"])[col].mean()


def paired(arm: pd.Series, ref: pd.Series, rng: np.random.Generator) -> dict:
    idx = arm.index.intersection(ref.index)
    a, r = arm.loc[idx].to_numpy(), ref.loc[idx].to_numpy()
    d = a - r
    if not len(d):
        return {}
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=int(len(d)), arm=float(a.mean()), ref=float(r.mean()),
                delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                ties=int((d == 0).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="per-rollout parquet for the arm")
    ap.add_argument("--ref", required=True, help="per-rollout parquet for the reference")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    arm_df, ref_df = pd.read_parquet(a.arm), pd.read_parquet(a.ref)
    label = a.label or Path(a.arm).stem.replace("agg_modes_", "").replace("_per_rollout", "")
    rng = np.random.default_rng(SEED)

    report = {}
    hdr = (f"{'cut':<15}{'mode':<14}{'n':>5}{'arm':>9}{'ref':>9}{'delta':>10}"
           f"{'95% CI':>22}{'win/loss':>12}")
    print(f"\n=== {label} vs reference: R-precision (all), paired per protein ===")
    print(hdr)
    print("-" * len(hdr))
    for cut, mask in CUTS.items():
        for mode in MODES:
            try:
                arm_p = per_protein(arm_df[mask(arm_df)], mode, "r_prec")
                ref_p = per_protein(ref_df[mask(ref_df)], mode, "r_prec")
            except KeyError:
                continue
            res = paired(arm_p, ref_p, rng)
            if not res:
                continue
            report[f"{cut}/{mode}"] = res
            ci = f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]"
            sig = "" if res["ci_lo"] <= 0 <= res["ci_hi"] else "  *"
            tag = f"{mode}{' *ORACLE' if mode == 'best' else ''}"
            print(f"{cut:<15}{tag:<14}{res['n']:>5}{res['arm']:>9.4f}{res['ref']:>9.4f}"
                  f"{res['delta']:>+10.4f}{ci:>22}{res['wins']:>6}/{res['losses']:<6}{sig}")
    print("\n* = the 95 % paired bootstrap CI excludes zero.")
    print("`best` is an ORACLE (selects with ground truth); it bounds a selector, "
          "it is not deployable.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"label": label, "results": report}, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
