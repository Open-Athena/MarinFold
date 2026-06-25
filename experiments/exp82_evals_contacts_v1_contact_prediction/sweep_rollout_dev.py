# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout sampling-hyperparameter sweep on the FoldBench dev set (exp82).

Rollout (sampled-completion frequency voting) was the dev-set-winning inference
method for the strong contacts-v1 model (eval loss 2.7566) — it beat pairwise on
every metric while iterative hurt. Rollout is variance reduction, so *sharper*
sampling should vote more decisively. This sweeps the sampling knobs to see how
much further it goes:

* **temperature** — 1.0 / 0.7 / 0.5 (sharpen the per-step distribution).
* **top_p** — nucleus cutoff.
* **top_k** — incl. a domain-aware ``k = L/5`` (restrict each step to the ~L/5
  likeliest next position tokens — there are only ~L plausible contact partners).

All configs run on the **same 16 FoldBench dev proteins** — we tune on dev, never
the held-out contacts-v1 test split. pairwise is reprinted as the reference; the
first rollout config (T1.0/p0.95/k50) reproduces the main run's 0.158.

Run (needs marinfold on the path)::

    PYTHONPATH=<repo>/marinfold uv run python sweep_rollout_dev.py \
        --model /home/bizon/exp89_export/hf_step35679 --n-rollouts 100
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from eval_contact_prediction import Scorer, aggregate, metrics, rank_pairwise, rank_rollout
from eval_search_foldbench_dev import DEV_JSONL, load_dev_proteins

# (name, temperature, top_p, top_k); top_k="L/5" => per-protein max(1, L//5).
CONFIGS = [
    ("T1.0 p0.95 k50", 1.0, 0.95, 50),    # baseline — reproduces the main run (~0.158)
    ("T0.7 p0.95 k50", 0.7, 0.95, 50),    # lower temperature
    ("T0.5 p0.95 k50", 0.5, 0.95, 50),    # lower still (watch for mode collapse)
    ("T0.7 p0.90 k50", 0.7, 0.90, 50),    # lower temp + tighter nucleus
    ("T0.7 p1.0 kL/5", 0.7, 1.0, "L/5"),  # domain-aware top-k + low temp
    ("T1.0 p1.0 kL/5", 1.0, 1.0, "L/5"),  # top-k alone (isolate its effect)
]
COLS = ["long_P@L", "long_P@L2", "long_P@L5", "long_R", "medlong_P@L", "medlong_R", "P@ngt"]


def agg_row(name: str, rows: list[dict]) -> str:
    a = aggregate(rows)
    return f"{name:<16} " + " ".join(f"{a.get(c, float('nan')):>11.3f}" for c in COLS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    proteins = load_dev_proteins(DEV_JSONL)
    scorer = Scorer(args.model)
    print(f"{len(proteins)} FoldBench dev proteins; n_rollouts={args.n_rollouts} "
          f"batch={args.batch}\n", flush=True)
    print(f"{'config':<16} " + " ".join(f"{c:>11}" for c in COLS), flush=True)
    print("(long = sep>=24, medlong = >=12; R = R-precision = precision at top-#GT in band)",
          flush=True)

    # pairwise reference (deterministic) — re-run so the table is self-contained.
    pw = [dict(metrics(rank_pairwise(scorer, p), p.gt, p.L)) for p in proteins]
    print(agg_row("pairwise (ref)", pw), flush=True)

    for name, temp, top_p, top_k in CONFIGS:
        rows = []
        for p in proteins:
            torch.manual_seed(args.seed)  # same RNG start per protein => controlled config comparison
            k = max(1, p.L // 5) if top_k == "L/5" else top_k
            rk = rank_rollout(scorer, p, args.n_rollouts, temp, top_p, top_k=k, batch=args.batch)
            rows.append(dict(metrics(rk, p.gt, p.L)))
        print(agg_row(name, rows), flush=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
