# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B2 (issue #211) — is the metric sensitive *at the model's operating point*?

The GT gate (``run_gt_gate.py``) established that the metric separates a true
contact set from a separation-matched random one by 5.6x. That is a large, easy
contrast. The experiment's actual comparison is far subtler: a rollout against a
chimera built from the *same* rollouts with the *same* per-pair marginals. Both
are ~60/40 true/false mixtures of the same contacts. If the score cannot resolve
differences in that neighbourhood the experiment cannot work no matter how many
rollouts are generated, and it is much cheaper to learn that here than after a
16-H100 fan-out.

So: sweep the corruption fraction across the band the model actually lives in
(#199 scores R-precision ~0.59, so ~40% of an emitted set is wrong) and ask
whether neighbouring levels are *paired-separable* across proteins.

    uv run python power_check.py --gt-dir _scratch/gt --n-proteins 60

Corruption replaces a fraction of the true contacts with separation-matched
random ones, holding the contact count and the ``|i - j|`` profile fixed — so the
only thing that varies is how much joint geometry survives.

**This is an upper bound on the real effect, not a prediction of it.** Corrupting
a true set destroys joint structure completely at the corrupted pairs, whereas
the marginal-matched chimera keeps every pair the model actually proposed and
only breaks their co-occurrence. If the sweep is flat here it is certainly flat
there; if it is steep, the real effect still has to be measured.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arms import separation_matched_random
from calibrate_bounds import load_bundle
from consistency import contact_matrix, embed_residual
from run_gt_gate import bounds_from_json, chain_break_count, gt_pairs

# The band #199 actually occupies, bracketed. R-precision ~0.59 puts a rollout
# near 0.40; the neighbouring levels are what a within-vs-chimera gap would look
# like if it were worth a few points of effective precision.
LEVELS = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0)


def corrupt(pairs, length, frac, rng):
    """Replace ``frac`` of the contacts with separation-matched random ones."""
    src = sorted(set(pairs))
    k = int(round(frac * len(src)))
    if k == 0:
        return src
    keep = [src[i] for i in rng.permutation(len(src))[k:]]
    swap = [src[i] for i in rng.permutation(len(src))[:k]]
    return sorted(set(keep) | set(separation_matched_random(swap, length, rng)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=Path("_scratch/gt"))
    ap.add_argument("--bounds", type=Path, default=Path("data/bounds.json"))
    ap.add_argument("--out", type=Path, default=Path("data/power_check.csv"))
    ap.add_argument("--n-proteins", type=int, default=60)
    ap.add_argument("--min-length", type=int, default=100,
                    help="the gate found L<100 uninformative (nearly everything "
                         "embeds), so the power check runs where the metric works")
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    bounds = bounds_from_json(args.bounds)

    proteins = []
    for record_id, meta, xyz, raw in load_bundle(args.gt_dir):
        length = int(meta["L"])
        if length < args.min_length or chain_break_count(xyz) > 0:
            continue
        gt = gt_pairs(raw)
        if len(gt) >= 20:
            proteins.append((record_id, meta["dataset"], length, gt))
    idx = np.random.default_rng(0).permutation(len(proteins))[: args.n_proteins]
    proteins = [proteins[i] for i in sorted(idx)]
    print(f"[power] {len(proteins)} proteins, L>={args.min_length}, no chain breaks",
          flush=True)

    rows, t0 = [], time.time()
    for k, (record_id, dataset, length, gt) in enumerate(proteins):
        rng = np.random.default_rng(1000 + k)
        sets = [corrupt(gt, length, f, rng) for f in LEVELS]
        masks = np.stack([contact_matrix(s, length) for s in sets])
        emb = embed_residual(masks, bounds, n_restarts=args.n_restarts,
                             iters=args.iters, seed=k)
        for f, s, e in zip(LEVELS, sets, emb):
            rows.append({"record_id": record_id, "dataset": dataset, "L": length,
                         "frac_corrupt": f, "n_contacts": len(s), **e})
        if (k + 1) % 20 == 0:
            print(f"[power] {k + 1}/{len(proteins)}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    w = df.pivot_table(index="record_id", columns="frac_corrupt",
                       values="contact_excess_per_contact")
    print(f"\n=== dose-response at the operating point (n={len(w)} proteins) ===")
    print(f"{'frac':>6} {'median':>9} {'mean':>9}")
    for f in LEVELS:
        print(f"{f:>6.2f} {w[f].median():>9.4f} {w[f].mean():>9.4f}")

    # The question that decides the experiment: are *neighbouring* levels
    # separable pairwise? A 0.05 step in corruption is the scale a
    # within-vs-chimera gap would plausibly live at.
    from scipy.stats import wilcoxon

    print(f"\n{'contrast':>16} {'higher on':>11} {'median delta':>14} {'Wilcoxon p':>12}")
    for a, b in [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
                 (0.35, 0.45), (0.30, 0.50)]:
        d = w[b] - w[a]
        try:
            p = wilcoxon(w[a], w[b]).pvalue
        except ValueError:
            p = float("nan")
        print(f"{a:.2f} -> {b:.2f}  {100 * (d > 0).mean():>10.1f}% "
              f"{d.median():>14.4f} {p:>12.2e}")

    print(f"\nwrote {args.out}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
