# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Where is the headroom in oracle-best-section F1? — issue #237 follow-up.

Arm M-B's target is ``max_k F1(section k)``: the ceiling a perfect selector could
reach from one rollout. Two arms have moved it (M-B step-36 to 0.5574, M-C
step-18 to 0.5578, both from #230's 0.5342) and the obvious question is where the
next increment comes from. ``max`` of a sample has exactly two levers — **how
many draws** and **the distribution they come from** — so both are priced here,
offline, on #230's own generations.

**Lever 1, more candidates.** Not saturating: E[max F1] rises ~+0.022 per
doubling and is still climbing at 22, which is where the 8,192-token context
runs out.

**Lever 2, a better distribution — tested and refuted.** Per-section F1 against
section size is strongly non-monotone: it peaks at ~80 contacts (mean 0.55) and
collapses for the small sections #230's power-law size draw produces (36.6 % of
sections are under 73 contacts, mean F1 0.32). The tempting conclusion is that
uniform ~80-contact sections would raise the ceiling. Paired on the same
proteins, they do the opposite for anything but a single draw:

    candidates   all sections   in-band only     gain
             1         0.4233         0.4382   +0.0149
             4         0.4996         0.4987   -0.0009
             8         0.5237         0.5201   -0.0036
            22         0.5527         0.5442   -0.0084

Restricting to the good size band raises the **mean** section F1 from 0.432 to
0.532 and *lowers* E[max of 22]. The variance that makes the average candidate
worse is exactly what best-of-N feeds on — the same trade this whole experiment
keeps finding, now on the corpus's section-size law rather than on a reward.

Caveat, stated because it bounds the claim: this is a re-sampling of sections
that were *generated* under the power law, not a model retrained to emit uniform
sections. It shows that selecting for size does not help; it does not prove a
size-uniform model would fail.

    python analyze_oracle_headroom.py --data data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BAND = (73, 105)      # the peak-F1 size band, from the octile table below
SEED = 237
NS = [1, 2, 4, 8, 16, 22]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    d = pd.read_parquet(a.data / "phase0_per_section.parquet")
    rng = np.random.default_rng(SEED)
    out = {}

    # --- lever 1: how much is another candidate worth? -----------------------
    pools = [np.array(v) for v in d.groupby(["dataset", "stem", "r"])["f1"].apply(list)
             if len(v) >= max(NS)]
    curve = {}
    for n in NS:
        curve[n] = float(np.mean([
            np.mean([np.max(rng.choice(v, n, replace=False)) for _ in range(a.reps)])
            for v in pools]))
    out["e_max_by_n"] = curve
    print(f"[headroom] {len(pools)} rollouts with >= {max(NS)} sections\n")
    print(f"{'candidates':>11}{'E[max F1]':>12}{'gain':>9}")
    prev = None
    for n in NS:
        print(f"{n:>11}{curve[n]:>12.4f}" + (f"{curve[n] - prev:>+9.4f}" if prev else " " * 9))
        prev = curve[n]

    # --- the size / F1 relationship ------------------------------------------
    q = pd.qcut(d["size"], 8, duplicates="drop")
    tab = d.groupby(q, observed=True).agg(n=("f1", "size"), size_med=("size", "median"),
                                          f1_mean=("f1", "mean"))
    print("\n[headroom] per-section F1 by size octile (non-monotone, peaks ~80 contacts)")
    print(tab.round(3).to_string())
    out["f1_by_size_octile"] = [
        dict(size_median=float(r.size_med), n=int(r.n), f1_mean=float(r.f1_mean))
        for r in tab.itertuples()]

    # --- lever 2, paired: does restricting to the good band help? ------------
    inb = d[(d["size"] >= BAND[0]) & (d["size"] <= BAND[1])]
    keep = {k for k, v in inb.groupby(["dataset", "stem"])["f1"].apply(list).items()
            if len(v) >= 8}
    idx_all = d.set_index(["dataset", "stem"]).index.isin(keep)
    idx_inb = inb.set_index(["dataset", "stem"]).index.isin(keep)
    allpool = {k: v.values for k, v in d[idx_all].groupby(["dataset", "stem"])["f1"]}
    bandpool = {k: v.values for k, v in inb[idx_inb].groupby(["dataset", "stem"])["f1"]}

    def emax(pool, n):
        return float(np.mean([np.mean([np.max(rng.choice(pool[k], n, replace=True))
                                       for _ in range(a.reps)]) for k in pool]))

    print(f"\n[headroom] paired on {len(keep)} proteins: all sections vs the "
          f"{BAND[0]}-{BAND[1]}-contact band only")
    print(f"{'candidates':>11}{'all':>12}{'in-band':>12}{'gain':>9}")
    paired = []
    for n in NS:
        x, y = emax(allpool, n), emax(bandpool, n)
        paired.append(dict(n=n, all_sections=x, in_band=y, gain=y - x))
        print(f"{n:>11}{x:>12.4f}{y:>12.4f}{y - x:>+9.4f}")
    out["size_band_paired"] = paired
    out["mean_f1_all"] = float(d[idx_all].f1.mean())
    out["mean_f1_in_band"] = float(inb[idx_inb].f1.mean())
    print(f"\nmean section F1: all {out['mean_f1_all']:.4f} -> in-band "
          f"{out['mean_f1_in_band']:.4f}, and E[max of 22] FALLS. The spread is the resource.")

    (a.data / "oracle_headroom.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.data}/oracle_headroom.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
