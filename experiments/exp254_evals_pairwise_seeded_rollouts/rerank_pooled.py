# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Can a better score over the pooled candidate pairs beat a raw vote count?

exp254's coverage diagnostic says the sample is not the problem: the 100
rollouts collectively propose **92 % of the true contacts** using only ~16x R
distinct pairs, while ranking them by vote count recovers 52 % at the R cut. The
gap between 0.52 and 0.92 is ranking loss, and it is an order of magnitude
larger than anything the seeding arms moved.

So this fits a score over three per-pair features that cost no extra inference:

``log(votes + 1)``
    the exp82 recipe's entire ranking signal.
``log P(contact)``
    the pairwise readout from phase 1. exp82 used this only as a *tie-break*
    inside equal-vote groups, worth +0.0007; here it gets a weight and can move
    pairs across vote boundaries.
``mean emission rank``
    where in a rollout the pair tended to be written, normalised by that
    rollout's length. A pair a rollout commits to first is not the same evidence
    as one it appends last, and the vote count throws that away.
``log(quality-weighted votes + 1)``
    a vote count in which each rollout is weighted by its own self-consistency
    -- the mean plain-vote count of the pairs it emitted, which needs no ground
    truth. A pair asserted by rollouts that otherwise agree with the ensemble is
    better evidence than one asserted by outliers. This is the only feature here
    carrying *joint* information: the other three score a pair in isolation.

Features are standardised **within a protein** -- proteins differ by an order of
magnitude in L and in vote scale, so a globally-fitted weight on raw features
would mostly be fitting protein size.

**Everything is 5-fold cross-validated over proteins.** eval-val is 97 proteins
and this fits 1-4 parameters on them; an in-sample number here would be a
guarantee of nothing. The in-sample fit is reported next to the cross-validated
one precisely so the gap is visible.

    uv run python rerank_pooled.py --run /data/exp_contactseed/run --out data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from build_metrics import (
    MIN_SEP,
    load_detail,
    metric_rows,
    resolved_pairs,
    true_matrix,
)
from common import EXPECTED_UNITS, load_ground_truth, load_targets

FEATURES = ("log_votes", "log_pairwise", "mean_emission_rank", "log_quality_votes")


def protein_features(detail: pd.DataFrame, pairwise: np.ndarray, L: int):
    """Dense ``[L, L]`` feature planes for one protein.

    ``mean_emission_rank`` is 1.0 for a pair no rollout emitted, which is the
    "latest possible" value -- unvoted pairs are ranked by the pairwise term
    alone and this keeps the feature from inventing a preference among them.
    """
    votes = np.zeros((L, L))
    rank_sum = np.zeros((L, L))
    ii = detail["i"].to_numpy()
    jj = detail["j"].to_numpy()
    rollouts = detail["rollout"].to_numpy()
    # Normalise each rollout's emission rank by its own length, so a long
    # rollout's rank-20 and a short one's rank-20 are not treated as equal.
    lengths = detail.groupby("rollout")["rank"].transform("max").to_numpy() + 1
    normalised = detail["rank"].to_numpy() / np.maximum(lengths, 1)
    np.add.at(votes, (ii, jj), 1.0)
    np.add.at(rank_sum, (ii, jj), normalised)
    votes = votes + votes.T
    rank_sum = rank_sum + rank_sum.T
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rank = np.where(votes > 0, rank_sum / np.maximum(votes, 1), 1.0)
    # Rollout self-consistency: the mean plain-vote count of the pairs a
    # rollout emitted, normalised by the rollout count. High means "this rollout
    # agrees with the ensemble". Derived from `votes`, so it uses no ground truth.
    n_rollouts = max(len(np.unique(rollouts)), 1)
    quality = pd.Series(votes[ii, jj]).groupby(rollouts).mean() / n_rollouts
    quality_votes = np.zeros((L, L))
    np.add.at(quality_votes, (ii, jj), quality.reindex(rollouts).to_numpy())
    quality_votes = quality_votes + quality_votes.T

    return {
        "log_votes": np.log1p(votes),
        "log_pairwise": np.log(np.clip(pairwise, 1e-30, None)),
        "mean_emission_rank": mean_rank,
        "log_quality_votes": np.log1p(quality_votes),
    }


def standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance within this protein's candidate set."""
    spread = values.std()
    return (values - values.mean()) / (spread if spread > 1e-12 else 1.0)


def build(run: Path, arm: str):
    """Per-protein candidate tables: standardised features, labels, geometry."""
    gt = load_ground_truth()
    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS
    detail = load_detail(run / arm)

    proteins = []
    for target in targets:
        record = gt[(target.dataset, target.stem)]
        L = record["L"]
        pairwise = np.load(run / "pairwise" / f"{target.key}.npz")["score"].astype(float)
        assert pairwise.shape == (L, L)
        mine = detail[(detail["dataset"] == target.dataset)
                      & (detail["stem"] == target.stem)]
        planes = protein_features(mine, pairwise, L)
        truth = true_matrix(L, record["contacts"])
        pi, pj, psep = resolved_pairs(np.asarray(record["resolved"], dtype=np.int64))
        keep = psep >= MIN_SEP
        pi, pj = pi[keep], pj[keep]
        design = np.column_stack([standardise(planes[f][pi, pj]) for f in FEATURES])
        proteins.append(dict(
            voted=planes["log_votes"][pi, pj] > 0,
            target=target, L=L, planes=planes, truth=truth,
            design=design, label=truth[pi, pj].astype(int),
            pi=pi, pj=pj,
            resolved=np.asarray(record["resolved"], dtype=np.int64),
        ))
    return proteins


def score_matrix(protein: dict, weights: np.ndarray) -> np.ndarray:
    """Dense ``[L, L]`` combined score, standardised on the same candidate set."""
    L = protein["L"]
    pi, pj = protein["pi"], protein["pj"]
    total = np.zeros((L, L))
    for weight, name in zip(weights, FEATURES):
        plane = protein["planes"][name]
        reference = plane[pi, pj]
        spread = reference.std()
        total += weight * (plane - reference.mean()) / (spread if spread > 1e-12 else 1.0)
    return total


def evaluate(proteins, weights_for) -> pd.DataFrame:
    """R-precision / AUC per protein under a (possibly per-fold) weight vector."""
    rows = []
    for index, protein in enumerate(proteins):
        weights = weights_for(index)
        score = score_matrix(protein, weights)
        record_target = protein["target"]
        pi, pj, psep = resolved_pairs(protein["resolved"])
        for row in metric_rows(score, protein["truth"], pi, pj, psep, protein["L"],
                               with_precision=True):
            rows.append(dict(stem=record_target.stem, **row))
    return pd.DataFrame(rows)


def fit_weights(proteins, indices) -> np.ndarray:
    """Logistic regression over the pooled candidates of the given proteins."""
    design = np.concatenate([proteins[i]["design"] for i in indices])
    label = np.concatenate([proteins[i]["label"] for i in indices])
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(design, label)
    return model.coef_[0]


def headline(frame: pd.DataFrame, label: str) -> dict:
    row = {"model": label}
    for rng in ("all", "long"):
        sel = frame[(frame["range"] == rng)]
        row[f"R_{rng}"] = float(sel[sel["cut"] == "R"]["precision"].mean())
        row[f"AUC_{rng}"] = float(sel[sel["cut"] == "AUC"]["precision"].mean())
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="iid",
                    help="which arm's rollouts to re-rank (default the control)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fit-on", choices=("all", "voted"), default="all",
                    help="'all' fits over every candidate pair, which is 99.9%% "
                         "never-proposed negatives and therefore optimises AUC "
                         "rather than the top of the ranking; 'voted' fits only "
                         "over pairs at least one rollout proposed, which is the "
                         "pool that actually competes for the top R")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    proteins = build(args.run, args.arm)
    n = len(proteins)
    print(f"[combine] built candidates for {n} proteins "
          f"({sum(len(p['label']) for p in proteins):,} pairs)")

    results, detail_frames = [], {}

    # --- baselines, no fitting -------------------------------------------
    for label, weights in (
        ("votes only (exp82 recipe)", np.array([1.0, 0.0, 0.0, 0.0])),
        ("pairwise only", np.array([0.0, 1.0, 0.0, 0.0])),
        ("emission rank only", np.array([0.0, 0.0, -1.0, 0.0])),
        ("quality-weighted votes only (unfitted)", np.array([0.0, 0.0, 0.0, 1.0])),
    ):
        frame = evaluate(proteins, lambda _i, w=weights: w)
        results.append(headline(frame, label))
        detail_frames[label] = frame

    # --- one-parameter blend, in sample, to show the shape ----------------
    alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)
    curve = []
    for alpha in alphas:
        weights = np.array([1.0 - alpha, alpha, 0.0, 0.0])
        frame = evaluate(proteins, lambda _i, w=weights: w)
        row = headline(frame, f"blend alpha={alpha:.2f}")
        row["alpha"] = float(alpha)
        curve.append(row)
    curve_frame = pd.DataFrame(curve)
    curve_frame.to_csv(args.out / "exp254_blend_curve.csv", index=False)
    best = curve_frame.loc[curve_frame["R_all"].idxmax()]
    print(f"\n[combine] best in-sample blend: alpha={best['alpha']:.2f} "
          f"R_all={best['R_all']:.4f} (votes-only {curve_frame.iloc[0]['R_all']:.4f})")

    # --- cross-validated fits --------------------------------------------
    order = np.random.default_rng(254).permutation(n)
    folds = np.array_split(order, args.folds)
    for label, columns in (("fitted: votes + pairwise", (0, 1)),
                           ("fitted: votes + pairwise + rank", (0, 1, 2)),
                           ("fitted: + quality-weighted votes", (0, 1, 2, 3)),
                           ("fitted: quality-weighted votes only", (3,))):
        weights_by_protein = np.zeros((n, len(FEATURES)))
        for fold in folds:
            train = np.setdiff1d(order, fold)
            # Fit on the selected feature columns only, then place the
            # coefficients back into the full-width weight vector.
            masked = np.zeros(len(FEATURES))
            def rows_of(i):
                mask = proteins[i]["voted"] if args.fit_on == "voted" else slice(None)
                return (proteins[i]["design"][mask][:, list(columns)],
                        proteins[i]["label"][mask])

            design = np.concatenate([rows_of(i)[0] for i in train])
            label_vector = np.concatenate([rows_of(i)[1] for i in train])
            model = LogisticRegression(max_iter=2000, C=1.0)
            model.fit(design, label_vector)
            masked[list(columns)] = model.coef_[0]
            weights_by_protein[fold] = masked
        frame = evaluate(proteins, lambda i: weights_by_protein[i])
        results.append(headline(frame, label + " (5-fold CV)"))
        detail_frames[label] = frame
        mean_weights = weights_by_protein.mean(axis=0)
        print(f"[combine] {label}: mean CV weights "
              + ", ".join(f"{f}={w:+.3f}" for f, w in zip(FEATURES, mean_weights)))


    summary = pd.DataFrame(results)
    summary["fit_on"] = args.fit_on
    summary.to_csv(args.out / f"exp254_rerank_summary_{args.fit_on}.csv", index=False)
    print("\n[combine] R-precision / AUC on eval-val (n=97):")
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
