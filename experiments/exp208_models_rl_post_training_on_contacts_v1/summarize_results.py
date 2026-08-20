# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired per-protein comparison of exp208 arms against the baseline — issue #208.

WHY PAIRED, AND AGAINST WHICH BASELINE. Between-protein spread of R-precision is
~0.3, so an unpaired mean over 554 proteins has an SEM of ~0.013 — wider than any
effect this experiment could plausibly produce. Every arm is scored on the *same*
554 proteins, so the paired difference is the right statistic and its standard
error is an order of magnitude smaller (#169 makes the same argument).

The baseline is **exp208's own Phase 1 parity run**, not the committed #180 rows
for exp199. Those disagree by +0.0226 on the unchanged checkpoint — the metric is
identical (n_true / n_candidate / n_top match on 100% of rows) and so is the
recipe, but #199 was scored on CoreWeave H100 and exp208 on v5p, and rollout
R-precision is not comparable across accelerators at that level. Comparing an arm
to 0.5873 would manufacture a +0.023 improvement out of hardware.

    uv run python summarize_results.py \\
        --baseline label=/scratch/scores/exp199_parity \\
        --arm armS=/scratch/scores/armS --arm armB=/scratch/scores/armB \\
        --gt ~/git/MarinFold/experiments/exp89_.../data/gt_universe.jsonl \\
        --out-paired data/exp208_paired.csv --out-summary data/exp208_summary.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

EXP82 = Path(__file__).resolve().parent.parent / "exp82_evals_contacts_v1_contact_prediction"
# #180: four evaluations of one unchanged checkpoint span this much. Any claimed
# effect has to clear it before anything else is worth saying about it.
REPEAT_SPAN = 0.0023


def build_rows(gt: Path, models: dict[str, Path], out: Path) -> pd.DataFrame:
    """Score every model dir through exp82's row builder (exp89 metrics verbatim)."""
    args = [sys.executable, str(EXP82 / "build_rollout_rows.py"), "--gt", str(gt)]
    for label, directory in models.items():
        args += ["--model", f"{label}={directory}"]
    with tempfile.TemporaryDirectory() as tmp:
        summary = Path(tmp) / "summary.csv"
        args += ["--out", str(out), "--summary", str(summary)]
        subprocess.run(args, check=True)
    return pd.read_csv(out)


def paired(rows: pd.DataFrame, baseline: str, arm: str, *, cut="R", rng="all") -> dict:
    """Mean paired delta, its SE, a 95% CI and the win rate."""
    sel = rows[(rows["cut"] == cut) & (rows["range"] == rng)]
    a = sel[sel["model"] == arm].set_index(["dataset", "stem"])["precision"]
    b = sel[sel["model"] == baseline].set_index(["dataset", "stem"])["precision"]
    joined = pd.concat([a.rename("arm"), b.rename("base")], axis=1).dropna()
    d = joined["arm"] - joined["base"]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {
        "arm": arm, "range": rng, "cut": cut, "n": n,
        "baseline_mean": float(joined["base"].mean()),
        "arm_mean": float(joined["arm"].mean()),
        "delta": float(d.mean()),
        "se": float(se),
        "sigma": float(d.mean() / se) if se and se > 0 else np.nan,
        "ci_lo": float(d.mean() - 1.96 * se),
        "ci_hi": float(d.mean() + 1.96 * se),
        "win_rate": float((d > 0).mean()),
        "tie_rate": float((d == 0).mean()),
        "clears_repeat_span": bool(abs(d.mean()) > REPEAT_SPAN),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--baseline", required=True, metavar="LABEL=DIR")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--out-rows", type=Path, default=Path("data/exp208_rows.csv.gz"))
    ap.add_argument("--out-paired", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()

    def split(spec):
        label, _, directory = spec.partition("=")
        if not directory:
            ap.error(f"expected LABEL=DIR, got {spec!r}")
        return label, Path(directory)

    base_label, base_dir = split(a.baseline)
    models = {base_label: base_dir}
    models.update(dict(split(s) for s in a.arm))

    a.out_rows.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(a.gt, models, a.out_rows)

    records = []
    for label in models:
        if label == base_label:
            continue
        for rng in ("all", "long"):
            records.append(paired(rows, base_label, label, rng=rng))
    out = pd.DataFrame(records)
    out.to_csv(a.out_paired, index=False)

    summary = (rows[rows["cut"].isin(["R", "L", "AUC"])]
               .groupby(["model", "range", "cut"])["precision"].mean().reset_index()
               .rename(columns={"precision": "mean"}))
    summary.to_csv(a.out_summary, index=False)

    print(f"\nbaseline: {base_label}  (exp208's own parity run — NOT the committed #180 rows;\n"
          f"          those differ by +0.0226 on the unchanged checkpoint, across accelerators)\n")
    show = out[["arm", "range", "baseline_mean", "arm_mean", "delta", "se", "sigma",
                "win_rate", "clears_repeat_span"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nPre-registered: primary >= +0.010 at >=3 sigma; signal floor >= +0.005 at >=3 sigma;\n"
          f"hard floor |delta| > {REPEAT_SPAN} (#180's four-repeat span).")
    print(f"wrote {a.out_paired} / {a.out_summary} / {a.out_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
