# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute exp120's downstream contact-prediction metrics from score matrices.

Reuses exp89's metric core (``metric_rows`` / ``load_gt`` / ``true_matrix`` /
``resolved_pairs``) over the shared GT universe, and reports the mean over the 554
eval-set proteins of R-precision, AUC, and contacts@{L,L/2,L/5} per sequence-
separation range — the "did you find the contacts" comparison from #89. Point it
at one or more ``score/`` dirs (one per checkpoint) produced by exp89's
``score_eval_set.py``.

    uv run python downstream_metrics.py \
        --gt ../exp89_.../data/gt_universe.jsonl \
        --scores orig-lr3e-4=<dir> regen-lr3e-4=<dir> [...]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP89 = Path(__file__).resolve().parent.parent / "exp89_evals_contacts_v1_model_on_eval_set"
sys.path.insert(0, str(EXP89))
from compute_metrics import load_gt, metric_rows, resolved_pairs, true_matrix  # noqa: E402


def score_model(score_dir: Path, gt: list[dict]) -> dict:
    """Mean metrics over proteins for one checkpoint's score dir."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    n = 0
    for rec in gt:
        npz = score_dir / f"{rec['dataset']}__{rec['stem']}.npz"
        if not npz.exists():
            continue
        score = np.load(npz)["score"].astype(np.float32)
        L = rec["L"]
        tmat = true_matrix(L, rec["contacts"])
        resolved = np.asarray(rec["resolved"], dtype=int)
        pi, pj, psep = resolved_pairs(resolved)
        for r in metric_rows(score, tmat, pi, pj, psep, L, with_precision=True):
            v = r["precision"]
            if not np.isnan(v):
                acc[(r["range"], r["cut"])].append(v)
        n += 1
    out = {"n_proteins": n}
    for (rng, cut), vals in acc.items():
        out[f"{rng}/{cut}"] = round(float(np.mean(vals)), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores", nargs="+", required=True,
                    help="name=score_dir pairs, one per checkpoint")
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    gt = load_gt(a.gt)
    print(f"GT universe: {len(gt)} proteins\n")
    keys = ["long/R", "long/AUC", "long/L", "all/R", "all/AUC", "medium/R", "short/R"]
    hdr = f"{'model':22s} {'n':>4s} " + " ".join(f"{k:>10s}" for k in keys)
    print(hdr); print("-" * len(hdr))
    all_rows = []
    for spec in a.scores:
        name, d = spec.split("=", 1)
        m = score_model(Path(d), gt)
        print(f"{name:22s} {m['n_proteins']:>4d} " +
              " ".join(f"{m.get(k, float('nan')):>10.4f}" for k in keys))
        all_rows.append({"model": name, **m})
    if a.csv:
        import csv
        cols = sorted({k for r in all_rows for k in r})
        with a.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model"] + [c for c in cols if c != "model"])
            w.writeheader(); w.writerows(all_rows)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
