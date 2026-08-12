# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 (issue #211) — the statistics and the figures.

Reads ``data/arm_scores.csv`` (from ``score_arms.py``) and answers the two
questions the experiment was filed for.

**Primary.** Is a rollout's own contact set more 3D-realizable than a
marginal-matched chimera drawn from the same protein's pooled rollout votes?
Both arms share the model, the protein, the per-pair marginals and the set size,
so any gap is joint structure the model put there while generating. Tested
paired at the protein level (each protein contributes the mean over its
replicates, so no protein counts more than once) with a Wilcoxon signed-rank
test and a bootstrap CI over proteins.

**Secondary.** Does the score rank a protein's rollouts by accuracy *without
ground truth*? If yes it is a best-of-N selector and a candidate RL reward for
#200/#208 — usable on sequences with no known structure, which is the entire
ESM-Atlas half of #199's training mixture. Measured as the per-protein Spearman
rho between consistency and precision, and as the precision gained by picking
the most-consistent rollout instead of a random one.

Analysis is restricted to **L >= 100 and chain-break-free** proteins by default:
the GT gate found the metric near-blind below L~100 (a short chain embeds almost
anything) and 15% of the eval set is not embeddable as the continuous 3.8 A
chain a contacts-v1 document asserts. Both subsets are reported, not hidden.

    uv run python analyze.py --scores data/arm_scores.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = "contact_excess_per_contact"
PAIRED_ARMS = ("rollout", "chimera_marginal", "chimera_splice")
REFERENCE_ARMS = ("gt", "gt_subsampled", "random", "decoy")


def boot_ci(x, n=10000, seed=0, alpha=0.05):
    """Bootstrap CI of the mean, resampling proteins."""
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    means = x[r.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def paired_table(df):
    """Per-protein mean of each arm — the unit of analysis for the primary test."""
    return df.pivot_table(index="record_id", columns="arm", values=METRIC,
                          aggfunc="mean")


def primary(df, label, out):
    from scipy.stats import wilcoxon

    w = paired_table(df)
    attrs = df.groupby("record_id")[["L", "n_gt_contacts"]].first()
    w = w.join(attrs)
    lines = [f"\n=== PRIMARY — {label} (n={len(w)} proteins) ==="]

    lines.append(f"\n  per-contact excess by arm (lower = more 3D-consistent):")
    for arm in list(PAIRED_ARMS) + list(REFERENCE_ARMS):
        if arm in w:
            lines.append(f"    {arm:20s} mean {w[arm].mean():.4f}   "
                         f"median {w[arm].median():.4f}")

    for null in ("chimera_marginal", "chimera_splice"):
        if null not in w or "rollout" not in w:
            continue
        d = (w[null] - w["rollout"]).dropna()   # >0 means the rollout is better
        lo, hi = boot_ci(d)
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float("nan")
        lines.append(
            f"\n  rollout vs {null}:"
            f"\n    mean delta (null - rollout) {d.mean():+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]"
            f"\n    rollout better on {100 * (d > 0).mean():.1f}% of proteins   "
            f"Wilcoxon p = {p:.3g}"
            f"\n    VERDICT: {'rollout MORE consistent' if (lo > 0) else ('rollout LESS consistent' if hi < 0 else 'NO DIFFERENCE (CI spans 0)')}"
        )
        out[f"{label}__rollout_vs_{null}"] = dict(
            n=int(len(d)), mean_delta=float(d.mean()), ci_lo=float(lo),
            ci_hi=float(hi), frac_rollout_better=float((d > 0).mean()),
            wilcoxon_p=float(p))
    return "\n".join(lines), w


def secondary(df, label, out):
    """Is consistency a reference-free predictor of rollout accuracy?"""
    from scipy.stats import spearmanr

    r = df[df["arm"] == "rollout"].dropna(subset=[METRIC, "precision"])
    rows, gains = [], []
    for rid, g in r.groupby("record_id"):
        if len(g) < 8 or g[METRIC].nunique() < 3:
            continue
        rho = spearmanr(g[METRIC], g["precision"]).statistic
        best = g.loc[g[METRIC].idxmin(), "precision"]   # most consistent rollout
        rows.append(dict(record_id=rid, rho=rho, L=g["L"].iloc[0]))
        gains.append(dict(record_id=rid, selected=best, mean=g["precision"].mean(),
                          oracle=g["precision"].max(), worst=g["precision"].min()))
    rho_df, gain_df = pd.DataFrame(rows), pd.DataFrame(gains)
    lines = [f"\n=== SECONDARY — reference-free selection, {label} "
             f"(n={len(rho_df)} proteins) ==="]
    if len(rho_df):
        neg = float((rho_df["rho"] < 0).mean())   # negative rho = lower excess -> higher precision
        lines.append(f"  Spearman rho(excess, precision): mean {rho_df['rho'].mean():+.4f}   "
                     f"negative (i.e. useful) on {100 * neg:.1f}% of proteins")
        d = gain_df["selected"] - gain_df["mean"]
        lo, hi = boot_ci(d)
        lines.append(f"  precision of most-consistent rollout vs mean rollout: "
                     f"{d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"  oracle-best rollout vs mean: "
                     f"{(gain_df['oracle'] - gain_df['mean']).mean():+.4f}  "
                     f"(the headroom any selector is competing for)")
        out[f"{label}__selection"] = dict(
            n=int(len(rho_df)), mean_rho=float(rho_df["rho"].mean()),
            frac_useful=float(neg), selection_gain=float(d.mean()),
            gain_ci=[float(lo), float(hi)],
            oracle_gain=float((gain_df["oracle"] - gain_df["mean"]).mean()))
    return "\n".join(lines), rho_df, gain_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("data/arm_scores.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("data/results.json"))
    ap.add_argument("--out-txt", type=Path, default=Path("data/results.txt"))
    ap.add_argument("--min-length", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    print(f"[analyze] {len(df):,} rows, {df['record_id'].nunique()} proteins, "
          f"arms {sorted(df['arm'].unique())}")

    main_set = df[(df["L"] >= args.min_length) & (~df["has_chain_break"])]
    short_set = df[df["L"] < args.min_length]
    broken_set = df[df["has_chain_break"]]

    out: dict = {"n_rows": int(len(df)),
                 "n_proteins": int(df["record_id"].nunique()),
                 "min_length": args.min_length}
    chunks = []
    txt, wide = primary(main_set, "main", out)
    chunks.append(txt)
    txt2, rho_df, gain_df = secondary(main_set, "main", out)
    chunks.append(txt2)

    # Length stratification — #180 predicts the effect grows with L.
    chunks.append("\n=== by length ===")
    for lo, hi in ((100, 200), (200, 350), (350, 10**9)):
        s = main_set[(main_set["L"] >= lo) & (main_set["L"] < hi)]
        if s["record_id"].nunique() < 10:
            continue
        w = paired_table(s)
        if "rollout" not in w or "chimera_marginal" not in w:
            continue
        d = (w["chimera_marginal"] - w["rollout"]).dropna()
        lo_ci, hi_ci = boot_ci(d)
        chunks.append(f"  L {lo:4d}-{min(hi, 761):4d}  n={len(d):3d}  "
                      f"delta {d.mean():+.4f}  CI [{lo_ci:+.4f}, {hi_ci:+.4f}]  "
                      f"rollout better on {100 * (d > 0).mean():.1f}%")

    # The excluded subsets, reported rather than hidden.
    for name, s in (("L<100", short_set), ("chain-break", broken_set)):
        if s["record_id"].nunique() >= 10:
            w = paired_table(s)
            if "rollout" in w and "chimera_marginal" in w:
                d = (w["chimera_marginal"] - w["rollout"]).dropna()
                chunks.append(f"\n  [{name} subset, excluded from the headline] "
                              f"n={len(d)}  delta {d.mean():+.4f}  "
                              f"rollout better on {100 * (d > 0).mean():.1f}%")

    report = "\n".join(chunks)
    print(report)
    args.out_txt.write_text(report + "\n")
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    wide.to_csv(args.out_json.with_name("per_protein_arms.csv"))
    if len(rho_df):
        rho_df.to_csv(args.out_json.with_name("per_protein_rho.csv"), index=False)
        gain_df.to_csv(args.out_json.with_name("per_protein_selection.csv"), index=False)
    print(f"\n[analyze] wrote {args.out_txt}, {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
