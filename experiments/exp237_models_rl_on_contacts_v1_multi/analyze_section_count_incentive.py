# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does arm M-C's reward pay a rollout for emitting FEWER sections? — issue #237.

**The hypothesis, and it is not the one this experiment first wrote down.** Arm
M-C pays each section its leave-one-out contribution to its own rollout's
consensus, ``m_k = C(all) − C(all \\ {k})``. That quantity is not scale-free in
the number of sections: with 22 sections, removing one barely moves a vote count
and ``m_k ≈ 0``; with two, removing one halves the vote and ``m_k`` is enormous;
with one, ``C(all \\ {k})`` is the consensus of *nothing*. So **a rollout can
raise every one of its sections' rewards simply by emitting fewer of them** — by
making its own consensus worse, so that each surviving section is more
load-bearing.

Group centring does not remove this. The group is the prompt's rollouts, and a
rollout that emits fewer sections than its siblings scores above the group mean
on *every* section, gets a positive advantage on all of them, and is reinforced.
The pressure is within-group and first-order.

**Why the observational test is underpowered, and what this does instead.**
Correlating section count against marginal across #230's own generations gives
rho = −0.04: the base model fills the context every time, so 95 % of rollouts sit
within ±5 % of their group's median section count and there is almost no
variation to correlate. The incentive is *latent* — invisible in the base
distribution, and exactly what gradient ascent goes looking for. So this measures
it directly instead, by **truncating real rollouts** to a controlled number of
sections and re-running the reward on them:

* Part 1 — the mechanical effect. For a real rollout, mean ``m_k`` as a function
  of how many of its sections are kept.
* Part 2 — the policy-gradient effect. Build synthetic groups whose rollouts
  differ *only* in section count, centre exactly as `centred_section_advantages`
  does, and report mean advantage by section count. This is what the optimiser
  sees.

    python analyze_section_count_incentive.py --sections <#230 agg_sections> \\
        --targets <eval577_targets.parquet> --out data/
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent / "skyrl"))

import consensus as cs  # noqa: E402

#: Section counts to truncate to. 22 is #230's measured mean.
KEEP = [1, 2, 3, 4, 6, 8, 12, 16, 22]
SEED = 237


def load_sections(root: str) -> dict:
    by = defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(f"{root.rstrip('/')}/**/*.parquet", recursive=True)):
        for row in pq.read_table(p).to_pylist():
            if row["sec_idx"] < 0:
                continue
            by[(row["dataset"], row["stem"])][row["r"]][row["sec_idx"]] = {
                (int(i), int(j)) for i, j in row["contacts"]}
    return {k: {r: [s[i] for i in sorted(s)] for r, s in v.items()} for k, v in by.items()}


def marginals(sections, is_true, position, n_pairs, n_true):
    votes = cs.vote_counts(sections, position, n_pairs)
    c, m = cs.loo_marginals(votes, is_true, n_true)
    return c, np.nan_to_num(m, nan=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--n-proteins", type=int, default=250)
    a = ap.parse_args()

    tgt = {(r["dataset"], r["stem"]): r for r in pq.read_table(a.targets).to_pylist()}
    sections = load_sections(a.sections)
    rng = np.random.default_rng(SEED)
    keys = sorted(sections)
    rng.shuffle(keys)
    keys = keys[: a.n_proteins]

    part1 = defaultdict(list)     # keep -> [mean marginal]
    cons = defaultdict(list)      # keep -> [consensus]
    part2 = defaultdict(list)     # keep -> [mean group-centred advantage]

    for key in keys:
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
        rolls = [s for s in sections[key].values() if len(s) >= max(KEEP)]
        if len(rolls) < 4:
            continue

        # Part 1 -- one rollout, truncated to each size.
        for secs in rolls[:4]:
            for k in KEEP:
                c, m = marginals(secs[:k], is_true, position, len(pairs), n_true)
                if not np.isnan(c):
                    part1[k].append(float(m.mean()))
                    cons[k].append(float(c))

        # Part 2 -- a synthetic group whose members differ ONLY in section count,
        # centred exactly as the reward does it. Each rollout is truncated to a
        # different size; every other property is held fixed by construction.
        sizes = [1, 2, 4, 8, 16, 22]
        marg_by_size, order = {}, []
        for secs, k in zip(rolls, sizes):
            c, m = marginals(secs[:k], is_true, position, len(pairs), n_true)
            if np.isnan(c):
                break
            marg_by_size[k] = m
            order.append(k)
        if len(order) < len(sizes):
            continue
        pooled = np.concatenate([marg_by_size[k] for k in order])
        mu, sd = pooled.mean(), pooled.std()
        if sd <= 0:
            continue
        for k in order:
            part2[k].append(float(((marg_by_size[k] - mu) / sd).mean()))

    print(f"[incentive] {len(cons[max(KEEP)])} truncated rollouts, "
          f"{len(part2[1])} synthetic groups\n")
    print("PART 1 -- the reward per section, against how many sections exist")
    print(f"{'sections kept':>14}{'mean marginal':>16}{'x vs 22':>10}{'consensus':>12}")
    base = float(np.mean(part1[max(KEEP)]))
    rows1 = []
    for k in KEEP:
        mm, cc = float(np.mean(part1[k])), float(np.mean(cons[k]))
        rows1.append(dict(keep=k, mean_marginal=mm, ratio_vs_22=mm / base if base else float("nan"),
                          consensus=cc))
        print(f"{k:>14}{mm:>16.5f}{mm / base if base else float('nan'):>10.1f}{cc:>12.4f}")

    print("\nPART 2 -- mean group-centred ADVANTAGE, groups differing only in section count")
    print(f"{'sections':>14}{'mean advantage':>17}")
    rows2 = []
    for k in sorted(part2):
        v = float(np.mean(part2[k]))
        rows2.append(dict(keep=k, mean_advantage=v))
        print(f"{k:>14}{v:>+17.4f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "section_count_incentive.json").write_text(json.dumps(
        {"per_section_reward": rows1, "group_centred_advantage": rows2,
         "n_groups": len(part2[1])}, indent=2) + "\n")
    print(f"\nwrote {a.out}/section_count_incentive.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
