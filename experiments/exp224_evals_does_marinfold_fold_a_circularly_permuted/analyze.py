#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step C — metrics and the CP-vs-WT contrast for exp224.

Three levels of analysis, narrowest to sharpest:

1. **Standard metrics per unit** — exp89's ``compute_metrics`` functions,
   imported verbatim so the numbers are on the same scale as every other
   MarinFold contact number: precision @ {L, L/2, L/5, R} and AUC, over the
   all / short / medium / long separation ranges, restricted to the
   resolved-residue candidate universe.

2. **The permutation contrast** — the thing the experiment is for. Every WT
   residue pair is classified by whether the permutation *moved* it:

   * ``within-segment``: both residues fall in the same permutation segment, so
     the pair sits at the **same** sequence separation in CP as in WT.
   * ``cross-segment``: the residues straddle the cut, so the pair's separation
     changes as ``CP_sep = (L_cp - n_linker - n_tail) - WT_sep``.

   Scoring the two classes separately, on the same model and the same 3D
   contacts, isolates the effect of the re-ordering from every other difference
   between the two molecules.

3. **Seed spread** — every metric is recomputed per seed, so the CP-WT gap is
   reported against the seed-to-seed standard deviation rather than asserted
   from one draw. The WT crystal replicates (1DSB, 1A2J) give the second, wider
   error bar: how much the *ground truth* alone moves the number.

    uv run python analyze.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# exp89 owns the metric definition; import it rather than fork it.
sys.path.insert(0, str(HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import (  # noqa: E402
    MIN_DEG, MIN_SEP, RANGES, metric_rows, true_matrix,
)

DATA = HERE / "data"


def load_units() -> dict:
    return json.loads((DATA / "units.json").read_text())


def seed_dirs(scores_root: Path) -> list[Path]:
    return sorted((p for p in scores_root.glob("seed*") if p.is_dir()),
                  key=lambda p: int(p.name[4:]))


def per_unit_metrics(units: dict, scores_root: Path) -> pd.DataFrame:
    """exp89 metrics for every (unit, seed)."""
    rows = []
    for sd in seed_dirs(scores_root):
        seed = int(sd.name[4:])
        for unit, rec in units.items():
            npz = sd / f"{unit}.npz"
            if not npz.exists():
                continue
            L = rec["L"]
            score = np.load(npz)["score"].astype(np.float64)
            assert score.shape == (L, L), f"{unit}: {score.shape} != ({L},{L})"
            tmat = true_matrix(L, rec["contacts"])
            resolved = np.array(rec["resolved_positions"])
            a, b = np.triu_indices(len(resolved), k=1)
            pi, pj = resolved[a], resolved[b]
            psep = pj - pi
            for r in metric_rows(score, tmat, pi, pj, psep, L, with_precision=True):
                rows.append(dict(unit=unit, pdb=rec["pdb"], role=rec["role"],
                                 label=rec["label"], L=L, seed=seed, **r))
    return pd.DataFrame(rows)


def permutation_contrast(units: dict, cpmap: dict, scores_root: Path) -> pd.DataFrame:
    """Score CP and WT on the SAME pairs, split by whether the permutation moved them.

    Both arms are evaluated in **WT coordinates** over the pairs whose residues
    are resolved in both structures, using each structure's own ground truth.
    ``cross-segment`` pairs are the ones the permutation re-ordered.
    """
    from sklearn.metrics import roc_auc_score

    cp, wt = units["cp_1un2"], units["wt_1fvk"]
    c2w = cpmap["cp_to_wt"]
    w2c = {w: c for c, w in enumerate(c2w) if w is not None}
    seg_a, seg_b = cpmap["seg_a"], cpmap["seg_b"]

    def seg_of_wt(w: int) -> str:
        return "A" if seg_a["wt_start"] <= w < seg_a["wt_end"] else "B"

    # Candidate universe: residues resolved in BOTH structures.
    cp_res_wt = {c2w[p] for p in cp["resolved_positions"] if c2w[p] is not None}
    both = sorted(cp_res_wt & set(wt["resolved_positions"]))
    idx = np.array(both)
    a, b = np.triu_indices(len(idx), k=1)
    wi, wj = idx[a], idx[b]
    keep = (wj - wi) >= MIN_SEP
    wi, wj = wi[keep], wj[keep]
    moved = np.array([seg_of_wt(int(i)) != seg_of_wt(int(j)) for i, j in zip(wi, wj)])

    # Ground truth per arm, in WT coordinates.
    gt = {}
    gt["wt"] = true_matrix(wt["L"], wt["contacts"])[wi, wj].astype(int)
    cp_t = true_matrix(cp["L"], cp["contacts"])
    ci = np.array([w2c[int(w)] for w in wi])
    cj = np.array([w2c[int(w)] for w in wj])
    lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
    gt["cp"] = cp_t[lo, hi].astype(int)

    rows = []
    for sd in seed_dirs(scores_root):
        seed = int(sd.name[4:])
        sc = {}
        s_wt = np.load(sd / "wt_1fvk.npz")["score"].astype(np.float64)
        sc["wt"] = s_wt[wi, wj]
        s_cp = np.load(sd / "cp_1un2.npz")["score"].astype(np.float64)
        sc["cp"] = s_cp[lo, hi]
        for arm in ("cp", "wt"):
            for cls, mask in (("all", np.ones_like(moved)),
                              ("within-segment", ~moved),
                              ("cross-segment", moved)):
                s, g = sc[arm][mask], gt[arm][mask]
                n_true = int(g.sum())
                if n_true == 0 or n_true == len(g):
                    continue
                order = np.argsort(-s, kind="mergesort")
                gs = g[order]
                rows.append(dict(
                    arm=arm, pair_class=cls, seed=seed,
                    n_candidate=len(g), n_true=n_true,
                    r_precision=float(gs[:n_true].sum()) / n_true,
                    precision_at_L=float(gs[:min(len(g), wt["L"])].sum())
                    / min(len(g), wt["L"]),
                    auc=float(roc_auc_score(g, s)),
                ))
    return pd.DataFrame(rows)


def gt_agreement(units: dict, cpmap: dict) -> dict:
    """How similar are the CP and WT folds? Bounds what the model could achieve."""
    c2w = cpmap["cp_to_wt"]
    cp, wt = units["cp_1un2"], units["wt_1fvk"]

    def true_set(rec):
        L = rec["L"]
        return {(i, j) for i, j, d in rec["contacts"]
                if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L}

    cp_in_wt = set()
    for i, j in true_set(cp):
        a, b = c2w[i], c2w[j]
        if a is not None and b is not None:
            cp_in_wt.add((min(a, b), max(a, b)))
    cp_res_wt = {c2w[p] for p in cp["resolved_positions"] if c2w[p] is not None}
    both = cp_res_wt & set(wt["resolved_positions"])

    def restrict(S):
        return {(i, j) for i, j in S if i in both and j in both}

    A, B = restrict(cp_in_wt), restrict(true_set(wt))
    out = dict(n_residues_both_resolved=len(both), n_cp_contacts=len(A),
               n_wt_contacts=len(B), n_shared=len(A & B),
               jaccard=len(A & B) / len(A | B),
               wt_recovered_in_cp=len(A & B) / len(B))
    for r in ("wt_1dsb", "wt_1a2j"):
        o = restrict(true_set(units[r]))
        out[f"{r}_jaccard_vs_1fvk"] = len(o & B) / len(o | B)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=HERE / "_scratch" / "scores")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    units = load_units()
    cpmap = json.loads((DATA / "cp_wt_map.json").read_text())

    per_unit = per_unit_metrics(units, args.scores)
    per_unit.to_csv(args.out_dir / "per_unit_metrics.csv.gz", index=False)
    contrast = permutation_contrast(units, cpmap, args.scores)
    contrast.to_csv(args.out_dir / "permutation_contrast.csv", index=False)
    agree = gt_agreement(units, cpmap)
    (args.out_dir / "gt_agreement.json").write_text(json.dumps(agree, indent=1))

    n_seeds = per_unit.seed.nunique()
    print(f"=== exp224: {n_seeds} seeds x 100 rollouts, model = MODELS.yaml default\n")

    print("Ground-truth fold agreement (bounds what any model could score):")
    print(f"  CP vs WT(1FVK): Jaccard {agree['jaccard']:.3f}, "
          f"{agree['wt_recovered_in_cp']:.1%} of WT contacts present in CP")
    print(f"  crystal replicates vs 1FVK: 1DSB {agree['wt_1dsb_jaccard_vs_1fvk']:.3f}, "
          f"1A2J {agree['wt_1a2j_jaccard_vs_1fvk']:.3f}  <- GT noise floor\n")

    print("Standard exp89 metrics (mean +/- sd over seeds):")
    hdr = f"  {'unit':10s} {'L':>4s}  " + "  ".join(f"{r:>14s}" for r in
                                                    ("R-prec all", "R-prec long", "AUC all"))
    print(hdr)
    for unit in ["cp_1un2", "ctrl_identity", "wt_1fvk", "wt_1dsb", "wt_1a2j"]:
        d = per_unit[per_unit.unit == unit]
        cells = []
        for rng, cut in (("all", "R"), ("long", "R"), ("all", "AUC")):
            v = d[(d.range == rng) & (d.cut == cut)].precision
            cells.append(f"{v.mean():.4f}+-{v.std():.4f}")
        print(f"  {unit:10s} {d.L.iloc[0]:4d}  " + "  ".join(f"{c:>14s}" for c in cells))

    print("\nPermutation contrast (same pairs, same 3D contacts, WT coordinates):")
    print(f"  {'pair class':16s} {'arm':4s} {'n_pairs':>8s} {'n_true':>7s} "
          f"{'R-precision':>16s} {'AUC':>16s}")
    for cls in ("all", "within-segment", "cross-segment"):
        for arm in ("wt", "cp"):
            d = contrast[(contrast.pair_class == cls) & (contrast.arm == arm)]
            if d.empty:
                continue
            print(f"  {cls:16s} {arm:4s} {d.n_candidate.iloc[0]:8d} {d.n_true.iloc[0]:7d} "
                  f"{d.r_precision.mean():.4f}+-{d.r_precision.std():.4f}   "
                  f"{d.auc.mean():.4f}+-{d.auc.std():.4f}")
    for cls in ("within-segment", "cross-segment"):
        w = contrast[(contrast.pair_class == cls) & (contrast.arm == "wt")]
        c = contrast[(contrast.pair_class == cls) & (contrast.arm == "cp")]
        d_r = c.r_precision.mean() - w.r_precision.mean()
        d_a = c.auc.mean() - w.auc.mean()
        print(f"  -> {cls:16s} CP-WT: R-precision {d_r:+.4f}, AUC {d_a:+.4f}")

    print(f"\nwrote {args.out_dir}/per_unit_metrics.csv.gz, permutation_contrast.csv, "
          f"gt_agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
