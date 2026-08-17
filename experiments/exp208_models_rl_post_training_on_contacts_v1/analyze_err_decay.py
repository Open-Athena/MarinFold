# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does `err_decay` break the p̄-centred baseline? — issue #208.

The stepwise reward pays `+(1-p̄)` for a correct contact and `-p̄·δ^k` for the
k-th wrong one in a section. At `δ=1` that is a calibrated baseline: a contact is
worth emitting exactly when its correctness probability beats p̄, since

    E[r] = p(1-p̄) - (1-p)p̄ = p - p̄

At `δ<1` the k-th error is discounted, so the marginal contact is worth
`p - p̄·δ^k`, which turns positive for ANY contact once a few errors are in. The
question this script answers is not whether that is true — it is arithmetic — but
whether it MATTERS at the error counts real rollouts produce.

It matters enormously if rollouts make many errors, because the total penalty is a
geometric series bounded at `p̄/(1-δ)` no matter how many errors follow, while the
positive term grows without bound. Real contacts-v1 rollouts emit ~160 contacts at
~0.27 precision — about 117 errors each.

Scored on the 10,000 Phase 0 rollouts. The penalty depends only on the error
COUNT, not on which contacts they are: the k-th error costs `p̄·δ^k` wherever it
falls, so summing over a rollout gives `p̄·(1-δ^n_wrong)/(1-δ)` regardless of how
errors interleave with correct contacts. That closed form is checked against the
production `dense_rewards` on synthetic token streams before it is used.

    python analyze_err_decay.py
"""

import numpy as np
import pandas as pd

import contact_rewards as cr

P_BAR_OBSERVED = 0.2547   # what the EMA settled to on the training pool
P_BAR_INITIAL = 0.45      # INITIAL_PRECISION, the run's opening value


def closed_form(n_hit, n_wrong, p_bar, decay):
    """Total stepwise reward for a single-section rollout."""
    positive = n_hit * (1.0 - p_bar)
    if decay == 1.0:
        penalty = p_bar * n_wrong
    else:
        penalty = p_bar * (1.0 - decay ** n_wrong) / (1.0 - decay)
    return positive - penalty


def _synthetic_response(order, p0=cr.P0_ID):
    """A token stream with contacts in `order` (True = correct), plus its truth set.

    Positions are chosen so that every pair clears MIN_SEP and correct/incorrect
    is controlled purely by membership in the returned ground-truth set.
    """
    ids, gt, pos_to_seq = [cr.BEGIN_STATEMENTS_ID], set(), {}
    for n in range(2 * len(order) + 2):
        pos_to_seq[n] = n
    for k, correct in enumerate(order):
        i, j = 0, 10 + k          # separation >= 6 for all k
        ids += [cr.CONTACT_ID, p0 + i, p0 + j]
        if correct:
            gt.add((i, j))
    ids.append(cr.END_ID)
    return ids, gt, pos_to_seq


def verify_closed_form():
    """The closed form must reproduce the production reward exactly."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(1, 40))
        order = rng.random(n) < rng.random()      # random precision per trial
        ids, gt, pos_to_seq = _synthetic_response(list(order), )
        for p_bar in (0.2547, 0.45):
            for decay in (0.0, 0.5, 1.0):
                out = cr.dense_rewards(ids, pos_to_seq, gt, mode="plain",
                                       precision_baseline=p_bar, err_decay=decay)
                actual = float(np.sum(out.token_rewards))
                n_hit = int(out.diagnostics.get("n_contacts_correct", 0))
                n_scored = int(out.diagnostics.get("n_contacts_scored", 0))
                expect = closed_form(n_hit, n_scored - n_hit, p_bar, decay)
                worst = max(worst, abs(actual - expect))
    assert worst < 1e-4, f"closed form disagrees with dense_rewards by {worst}"
    print(f"closed form verified against dense_rewards on 200 rollouts "
          f"x 2 p_bar x 3 decay  (max |diff| = {worst:.2e})\n")


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    verify_closed_form()

    d = pd.read_csv("data/phase0_per_rollout.csv.gz")
    d = d[d["n_pred"] > 0].copy()
    d["n_wrong"] = d["n_pred"] - d["n_hit"]
    print(f"{len(d)} Phase 0 rollouts, {d['dataset'].nunique()} datasets")
    print(f"median contacts/rollout {d['n_pred'].median():.0f}, "
          f"median errors/rollout {d['n_wrong'].median():.0f}, "
          f"median precision {d['precision'].median():.3f}\n")

    for p_bar in (P_BAR_OBSERVED, P_BAR_INITIAL):
        print(f"=== p_bar = {p_bar} " + "=" * 46)
        rows = []
        for decay in (0.5, 1.0):
            r = closed_form(d["n_hit"].to_numpy(), d["n_wrong"].to_numpy(), p_bar, decay)
            pos = d["n_hit"].to_numpy() * (1.0 - p_bar)
            pen = pos - r
            rows.append({
                "decay": decay,
                "mean reward": r.mean(),
                "penalty % of positive": 100.0 * pen.sum() / max(pos.sum(), 1e-9),
                "rho(reward, f1)": spearman(r, d["f1"]),
                "rho(reward, precision)": spearman(r, d["precision"]),
                "rho(reward, recall)": spearman(r, d["recall"]),
                "rho(reward, n_pred)": spearman(r, d["n_pred"]),
            })
        out = pd.DataFrame(rows).set_index("decay")
        print(out.round(4).to_string(), "\n")

        # The direct question an RL policy asks: what does one more WRONG contact cost?
        n_wrong = d["n_wrong"].to_numpy()
        for decay in (0.5, 1.0):
            marginal = p_bar * (decay ** n_wrong)
            free = 100.0 * (marginal < 0.01 * (1.0 - p_bar)).mean()
            print(f"  decay={decay}: cost of the NEXT wrong contact -- "
                  f"median {np.median(marginal):.6f}, "
                  f"mean {marginal.mean():.6f}  "
                  f"(vs +{1 - p_bar:.3f} for a correct one; "
                  f"{free:.1f}% of rollouts have it below 1% of that)")
        print()


def sweep(d):
    """Is any intermediate decay defensible, or is 1.0 simply best?"""
    print("=== decay sweep at p_bar = 0.2547 " + "=" * 30)
    rows = []
    for decay in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        r = closed_form(d["n_hit"].to_numpy(), d["n_wrong"].to_numpy(), P_BAR_OBSERVED, decay)
        pos = d["n_hit"].to_numpy() * (1.0 - P_BAR_OBSERVED)
        rows.append({
            "decay": decay,
            "penalty % of positive": 100.0 * (pos - r).sum() / max(pos.sum(), 1e-9),
            "rho(reward, f1)": spearman(r, d["f1"]),
            "rho(reward, precision)": spearman(r, d["precision"]),
            "next error costs": P_BAR_OBSERVED * decay ** np.median(d["n_wrong"]),
        })
    print(pd.DataFrame(rows).set_index("decay").round(4).to_string(), "\n")


def silence_risk(d):
    """At decay=1 the baseline bites: does it push the policy to emit nothing?

    `E[r] = p - p̄` is the point of the design, but it means a policy whose
    precision sits BELOW p̄ is paid to stay silent. p̄ is an EMA of observed
    precision so it converges to the policy, but its opening value is a free
    parameter, and INITIAL_PRECISION is 0.45 while the RL training pool measured
    0.267. That gap is a silence pressure applied to every rollout in the opening
    steps -- exactly the collapse the decay was introduced to avoid.
    """
    print("=== decay=1.0: share of rollouts paid to have spoken " + "=" * 12)
    for p_bar in (0.20, 0.2547, 0.30, 0.45, 0.50):
        r = closed_form(d["n_hit"].to_numpy(), d["n_wrong"].to_numpy(), p_bar, 1.0)
        print(f"  p_bar={p_bar:.4f}: {100.0 * (r > 0).mean():5.1f}% of rollouts have "
              f"reward > 0   (mean {r.mean():+.2f})")
    print("\n  p_bar must track the pool's true precision. Above it, decay=1 pays for\n"
          "  silence; below it, every contact is worth emitting regardless of quality.")


if __name__ == "__main__":
    main()
    _d = pd.read_csv("data/phase0_per_rollout.csv.gz")
    _d = _d[_d["n_pred"] > 0].copy()
    _d["n_wrong"] = _d["n_pred"] - _d["n_hit"]
    sweep(_d)
    silence_risk(_d)
