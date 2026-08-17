# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Score the five aggregation modes, on every eval cut.

| # | mode | what it is |
|---|---|---|
| 1 | `plain` | plain `<contacts-v1>`, exp82 rollout+resample — from Gate A |
| 2 | `consensus` | vote across the sections of ONE rollout |
| 3 | `best` | the highest-F1 section of one rollout — **ORACLE** |
| 4 | `last` | the final section |
| 5 | `second_last` | the section before the final one |

**Every mode is scored by Gate A's ``metrics_for``, imported, not copied.** Each
one is turned into an ``[L, L]`` score matrix and handed to the same function
that produced the Gate A number, so the five are comparable with each other AND
with #180's frontier. Re-implementing the metric per mode is the obvious way to
get five numbers that cannot be compared.

**How a single section becomes a ranking.** R-precision cuts a *ranking* at
R = n_true, but a section is an unordered SET. Scored as a 0/1 matrix, exp89's
stable ``mergesort`` puts every predicted pair first (in index order) and the
rest after, so the top-R is: all n predicted pairs when n <= R, else an
index-ordered R-subset of them. The consequence is worth stating plainly — a mode
that emits fewer than R contacts is **capped at n/R**, so ``n_pred`` is reported
next to every score rather than left implicit.

**Mode 3 is a ceiling, not a result.** It picks the section using the ground
truth. It bounds what a perfect selector could reach from one rollout; nothing
can be deployed at that number. It is labelled ORACLE everywhere it appears.

    python score_agg_modes.py --sections <dir> --targets <parquet> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from score_gate_a import metrics_for      # the SAME metric Gate A reports

MODES = ("consensus", "best", "last", "second_last")
#: The budget-matched plain baseline. `last`/`second_last` have no meaning
#: across INDEPENDENT rollouts -- there is no ordering to be last in -- so the
#: grouped path reports only the three rules that do.
PLAIN_MODES = ("single", "best", "consensus")


def load_sections(root: str) -> dict:
    """-> {(dataset, stem): {r: [set_of_pairs, ...]}} in section order."""
    parts = sorted(glob.glob(f"{root.rstrip('/')}/**/*.parquet", recursive=True))
    if not parts:
        raise SystemExit(f"no section parquets under {root}")
    by = defaultdict(lambda: defaultdict(dict))
    for p in parts:
        t = pq.read_table(p).to_pylist()
        for row in t:
            key = (row["dataset"], row["stem"])
            if row["sec_idx"] < 0:
                by[key].setdefault(row["r"], {})
                continue
            by[key][row["r"]][row["sec_idx"]] = {
                (int(i), int(j)) for i, j in row["contacts"]}
    out = {}
    for key, rolls in by.items():
        out[key] = {r: [secs[i] for i in sorted(secs)] for r, secs in rolls.items()}
    return out


def f1(pred: set, gt: set) -> float:
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    p, rc = tp / len(pred), tp / len(gt)
    return 2 * p * rc / (p + rc) if (p + rc) else 0.0


def score_matrix(mode: str, secs: list[set], gt: set, L: int):
    """(score matrix, n_pred) for one rollout under one aggregation rule."""
    M = np.zeros((L, L), np.float32)
    if not secs:
        return M, 0
    if mode == "consensus":
        # Vote count across the sections of THIS rollout. This is the only mode
        # that yields a real ranking rather than a 0/1 set.
        for s in secs:
            for i, j in s:
                if 0 <= i < L and 0 <= j < L:
                    M[i, j] += 1.0
                    M[j, i] += 1.0
        n_pred = int((np.triu(M, 1) > 0).sum())
        return M, n_pred
    if mode == "best":
        chosen = max(secs, key=lambda s: f1(s, gt))     # ORACLE: uses gt
    elif mode == "last":
        chosen = secs[-1]
    elif mode == "second_last":
        chosen = secs[-2] if len(secs) >= 2 else secs[-1]
    else:
        raise ValueError(mode)
    for i, j in chosen:
        if 0 <= i < L and 0 <= j < L:
            M[i, j] = 1.0
            M[j, i] = 1.0
    return M, len(chosen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="finetune")
    ap.add_argument("--group-rollouts", action="store_true",
                    help="plain baseline: pool each protein's ROLLOUTS into one "
                         "candidate list, so N independent rollouts are aggregated "
                         "exactly as N sections of one multi rollout are")
    a = ap.parse_args()

    tgt = {(r["dataset"], r["stem"]): r for r in pq.read_table(a.targets).to_pylist()}
    sections = load_sections(a.sections)
    print(f"[agg] {len(sections)} proteins with sections")

    rows = []
    for key, rolls in sections.items():
        rec = tgt.get(key)
        if rec is None:
            continue
        L = int(rec["L"])
        gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}

        if a.group_rollouts:
            # Each plain rollout contributes exactly one contact set; pool them
            # so the SAME rules that combined 22 sections now combine 22 rollouts.
            cands = [secs[0] for secs in rolls.values() if secs]
            if not cands:
                continue
            common = dict(dataset=key[0], stem=key[1], n_sections=len(cands),
                          n_gt=len(gt), in_legacy554=bool(rec["in_legacy554"]),
                          in_eval2=bool(rec["in_eval2"]),
                          designed_any=bool(rec["designed_any"]),
                          passes_30=bool(rec["passes_30"]))
            # `single` is the mean over the individual candidates, NOT one of
            # them: quoting a single arbitrary rollout would carry the sampling
            # noise of that draw into the comparison.
            for ci, c in enumerate(cands):
                M = np.zeros((L, L), np.float32)
                for i, j in c:
                    if 0 <= i < L and 0 <= j < L:
                        M[i, j] = M[j, i] = 1.0
                m = metrics_for(M.astype(np.float16), gt, L)
                rows.append(dict(r=ci, mode="single", n_pred=len(c),
                                 r_prec=m.get("all:R", float("nan")),
                                 r_prec_long=m.get("long:R", float("nan")),
                                 prec_L=m.get("all:L", float("nan")), **common))
            for mode in ("best", "consensus"):
                M, n_pred = score_matrix(mode, cands, gt, L)
                m = metrics_for(M.astype(np.float16), gt, L)
                rows.append(dict(r=0, mode=mode, n_pred=n_pred,
                                 r_prec=m.get("all:R", float("nan")),
                                 r_prec_long=m.get("long:R", float("nan")),
                                 prec_L=m.get("all:L", float("nan")), **common))
            continue

        for r, secs in rolls.items():
            for mode in MODES:
                if mode == "second_last" and len(secs) < 2:
                    # Honest: with one section there IS no second-to-last. Skipping
                    # (rather than falling back to the only section) keeps the mode
                    # from silently becoming `last` wherever the model emitted one.
                    continue
                M, n_pred = score_matrix(mode, secs, gt, L)
                m = metrics_for(M.astype(np.float16), gt, L)
                rows.append(dict(
                    dataset=key[0], stem=key[1], r=int(r), mode=mode,
                    n_sections=len(secs), n_pred=n_pred, n_gt=len(gt),
                    r_prec=m.get("all:R", float("nan")),
                    r_prec_long=m.get("long:R", float("nan")),
                    prec_L=m.get("all:L", float("nan")),
                    in_legacy554=bool(rec["in_legacy554"]),
                    in_eval2=bool(rec["in_eval2"]),
                    designed_any=bool(rec["designed_any"]),
                    passes_30=bool(rec["passes_30"]),
                ))
    if not rows:
        raise SystemExit("no rows scored")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = a.out or Path("data")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"agg_modes_{a.label}_per_rollout.parquet", index=False)

    CUTS = {
        "legacy554": df["in_legacy554"],
        "eval2": df["in_eval2"],
        "eval2_natural": df["in_eval2"] & ~df["designed_any"],
        "eval2_lt30": df["in_eval2"] & df["passes_30"],
    }
    report = {}
    print(f"\n{'cut':<15}{'mode':<14}{'n_prot':>7}{'R-prec':>9}{'long':>9}"
          f"{'n_pred':>8}{'n_gt':>7}{'sections':>10}")
    modes = PLAIN_MODES if a.group_rollouts else MODES
    for cut, mask in CUTS.items():
        for mode in modes:
            d = df[mask & (df["mode"] == mode)]
            if d.empty:
                continue
            # Mean over rollouts WITHIN a protein first, then over proteins, so a
            # protein with more rollouts does not get more weight.
            per = d.groupby(["dataset", "stem"]).agg(
                r_prec=("r_prec", "mean"), r_prec_long=("r_prec_long", "mean"),
                n_pred=("n_pred", "mean"), n_gt=("n_gt", "first"),
                n_sections=("n_sections", "mean"))
            tag = f"{mode}{' *ORACLE' if mode == 'best' else ''}"
            report[f"{cut}/{mode}"] = dict(
                n_proteins=int(len(per)), r_prec=float(per["r_prec"].mean()),
                r_prec_long=float(per["r_prec_long"].mean()),
                n_pred=float(per["n_pred"].mean()), n_gt=float(per["n_gt"].mean()),
                n_sections=float(per["n_sections"].mean()), oracle=(mode == "best"))
            print(f"{cut:<15}{tag:<14}{len(per):>7}{per['r_prec'].mean():>9.4f}"
                  f"{per['r_prec_long'].mean():>9.4f}{per['n_pred'].mean():>8.1f}"
                  f"{per['n_gt'].mean():>7.1f}{per['n_sections'].mean():>10.2f}")
    (out / f"agg_modes_{a.label}.json").write_text(json.dumps(report, indent=2))
    print(f"\n* ORACLE selects the section using ground truth -- a ceiling, not a result.")
    print(f"wrote {out}/agg_modes_{a.label}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
