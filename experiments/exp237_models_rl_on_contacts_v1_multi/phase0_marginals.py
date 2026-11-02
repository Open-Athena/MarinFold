# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — is the WITHIN-ROLLOUT section marginal a measurable signal? — issue #237.

#208's lesson, stated in its own words: *a null result at a learning rate that
does not move the policy is not a result*.  The cheaper version of that lesson is
this script.  Arm M-C's reward is

    A_k  =  (m_k - mean_g(m)) / std_g(m),      m_k = C(all sections) - C(all \\ {k})

where ``C`` is the consensus R-precision of ONE rollout's own sections.  Two
things can make that identically useless before a single GPU-hour is spent:

1. **``m_k`` is discrete.**  ``C`` is R-precision over integer vote counts with a
   stable positional tie-break, so removing one section out of ~22 very often
   changes nothing at all.  If ``m_k == 0`` for every section of a rollout then
   ``std_g(m) == 0`` and that whole prompt contributes **zero** advantage.  The
   fraction of rollouts in that state is a hard upper bound on how much of the
   training batch can carry signal.
2. **``m_k`` might be a restatement of section F1.**  If the marginal simply
   ranks sections the way their own F1 does, then arm M-C is arm M-B with extra
   steps and #230's oracle-best number already bounds it.  #208 asked the same
   question of its group-level marginal and called it "gate 2".

Both are answered offline, on generations that already exist: #230's
``eval/agg_sections`` parquets — 577 proteins x 8 multi rollouts x ~22 sections,
produced by the very checkpoint this experiment warm-starts from.

    python phase0_marginals.py --sections ~/exp230_data/eval/agg_sections \\
        --targets ~/exp230_data/eval577_targets.parquet --out data/
"""

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "skyrl"))

import consensus as cs  # noqa: E402


def f1(pred: set, gt: set) -> float:
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    p, r = tp / len(pred), tp / len(gt)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def load_sections(root: str) -> dict:
    """-> {(dataset, stem): {r: [set_of_pairs, ...]}} in section order."""
    parts = sorted(glob.glob(f"{root.rstrip('/')}/**/*.parquet", recursive=True))
    if not parts:
        raise SystemExit(f"no section parquets under {root}")
    by = defaultdict(lambda: defaultdict(dict))
    for p in parts:
        for row in pq.read_table(p).to_pylist():
            key = (row["dataset"], row["stem"])
            if row["sec_idx"] < 0:
                by[key].setdefault(row["r"], {})
                continue
            by[key][row["r"]][row["sec_idx"]] = {(int(i), int(j)) for i, j in row["contacts"]}
    return {k: {r: [s[i] for i in sorted(s)] for r, s in v.items()} for k, v in by.items()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without a scipy dependency."""
    if len(a) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else math.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None, help="proteins, for a smoke run")
    a = ap.parse_args()

    tgt = {(r["dataset"], r["stem"]): r for r in pq.read_table(a.targets).to_pylist()}
    sections = load_sections(a.sections)
    keys = sorted(sections)
    if a.limit:
        keys = keys[: a.limit]
    print(f"[phase0] {len(keys)} proteins", flush=True)

    per_rollout: list[dict] = []
    per_section: list[dict] = []
    for n, key in enumerate(keys, 1):
        rec = tgt.get(key)
        if rec is None:
            continue
        L = int(rec["L"])
        gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        if not gt:
            continue
        pairs, position = cs.candidate_index(L)
        is_true = cs.truth_mask(pairs, gt)
        n_true = int(is_true.sum())
        if n_true <= 0:
            continue
        for r, secs in sections[key].items():
            if not secs:
                continue
            votes = cs.vote_counts(secs, position, len(pairs))
            C, marg = cs.loo_marginals(votes, is_true, n_true)
            if math.isnan(C):
                continue
            marg = np.nan_to_num(marg, nan=0.0)
            f1s = np.array([f1(s, gt) for s in secs], dtype=np.float64)
            sizes = np.array([len(s) for s in secs], dtype=np.float64)
            # How many siblings also emitted each of this section's pairs -- the
            # redundancy the marginal is supposed to price.
            counts: dict = defaultdict(int)
            for s in secs:
                for p in s:
                    counts[p] += 1
            novelty = np.array(
                [np.mean([1.0 - (counts[p] - 1) / max(len(secs) - 1, 1) for p in s]) if s else 0.0
                 for s in secs], dtype=np.float64)
            js = [jaccard(secs[i], secs[j])
                  for i in range(len(secs)) for j in range(i + 1, len(secs))]
            union = set().union(*secs) if secs else set()
            per_rollout.append(dict(
                dataset=key[0], stem=key[1], r=int(r), L=L, n_sections=len(secs),
                n_gt=n_true, consensus=float(C),
                best_f1=float(f1s.max()), last_f1=float(f1s[-1]), mean_f1=float(f1s.mean()),
                marg_mean=float(marg.mean()), marg_std=float(marg.std()),
                marg_absmax=float(np.abs(marg).max()),
                frac_zero=float(np.mean(marg == 0.0)),
                dead=bool(marg.std() == 0.0),
                union_pairs=int(len(union)),
                total_votes=int(sum(len(s) for s in secs)),
                mean_jaccard=float(np.mean(js)) if js else math.nan,
                rho_marg_f1=spearman(marg, f1s),
                rho_marg_novelty=spearman(marg, novelty),
                rho_marg_size=spearman(marg, sizes),
                in_legacy554=bool(rec["in_legacy554"]),
            ))
            for k in range(len(secs)):
                per_section.append(dict(
                    dataset=key[0], stem=key[1], r=int(r), k=k, marg=float(marg[k]),
                    f1=float(f1s[k]), size=int(sizes[k]), novelty=float(novelty[k])))
        if n % 100 == 0:
            print(f"[phase0] {n}/{len(keys)}", flush=True)

    import pandas as pd

    dr = pd.DataFrame(per_rollout)
    ds = pd.DataFrame(per_section)
    a.out.mkdir(parents=True, exist_ok=True)
    dr.to_csv(a.out / "phase0_per_rollout.csv.gz", index=False)
    ds.to_parquet(a.out / "phase0_per_section.parquet", index=False)

    # GATE 1 -- does the reward exist at all?  A rollout whose marginals are all
    # equal contributes exactly zero advantage after group centring.
    dead = float(dr["dead"].mean())
    # GATE 2 -- does it say anything section F1 does not?
    rho_f1 = float(dr["rho_marg_f1"].dropna().mean())
    rho_nov = float(dr["rho_marg_novelty"].dropna().mean())
    summary = dict(
        n_rollouts=int(len(dr)), n_proteins=int(dr[["dataset", "stem"]].drop_duplicates().shape[0]),
        mean_sections=float(dr["n_sections"].mean()),
        mean_consensus=float(dr["consensus"].mean()),
        mean_best_f1=float(dr["best_f1"].mean()), mean_last_f1=float(dr["last_f1"].mean()),
        mean_union_pairs=float(dr["union_pairs"].mean()),
        mean_total_votes=float(dr["total_votes"].mean()),
        votes_per_pair=float(dr["total_votes"].sum() / max(dr["union_pairs"].sum(), 1)),
        mean_jaccard=float(dr["mean_jaccard"].mean()),
        frac_dead_rollouts=dead,
        mean_frac_zero_marginals=float(dr["frac_zero"].mean()),
        marg_std_within_rollout=float(dr["marg_std"].mean()),
        marg_sd_pooled=float(ds["marg"].std()),
        rho_marginal_vs_f1=rho_f1,
        rho_marginal_vs_novelty=rho_nov,
        rho_marginal_vs_size=float(dr["rho_marg_size"].dropna().mean()),
    )
    (a.out / "phase0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Phase 0: within-rollout section marginals ===")
    for k, v in summary.items():
        print(f"  {k:<32} {v:.4f}" if isinstance(v, float) else f"  {k:<32} {v}")
    print()
    print(f"GATE 1 (signal exists): {100 * (1 - dead):.1f}% of rollouts have a non-degenerate "
          f"marginal spread. Below ~50% and arm M-C is mostly training on zeros.")
    print(f"GATE 2 (not a restatement of F1): rho(marginal, section F1) = {rho_f1:.4f}. "
          f"Near 1.0 and M-C is M-B with extra steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
