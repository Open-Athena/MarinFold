# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — is the consensus marginal worth rewarding? — issue #208.

#208's document-level reward pays a rollout for its **leave-one-out marginal
contribution to the group's consensus** rather than for its own F1. That choice
is the experiment's main design bet, and it rests on two claims that were
assumptions when the plan was written:

1. **The marginal is not just precision in disguise.** If a rollout's marginal
   contribution is perfectly predicted by how often it is right, then the
   document term adds nothing the stepwise per-contact term does not already
   provide, and #208 collapses to a one-arm experiment.
2. **The marginal is estimable at the group size we can afford.** The reported
   metric is a consensus over 100 rollouts; RL groups will hold 8-32. A rollout's
   influence on a 16-rollout vote is a much cruder thing than its influence on a
   100-rollout vote, and if the two are uncorrelated the reward optimizes a
   quantity unrelated to what we report.

Both are measurable *before* spending any training compute, on rollouts the
baseline eval has to generate anyway. That is what this script does.

It also reports two things the plan got slightly wrong and that only reading the
eval path end-to-end reveals:

* there is **no pairwise tie-break** on the rollout path — ``fetch_cw_scores``
  writes bare vote counts, so ties are settled by the stable sort's index order.
  The right question is therefore not "what does our tie-break approximation
  cost" but "how much of top-R is decided by an arbitrary index order at all",
  which ``tie_fraction`` reports;
* **union coverage** (`|union of predictions| / R`) is a hard ceiling on
  R-precision — below 1.0 the metric pads top-R with zero-vote pairs that are
  almost never true. This is vote collapse in its most quantitative form.

    uv run python phase0_marginal_analysis.py \\
        --scores gs://marin-us-central1/protein-structure/MarinFold/exp208/phase0/scores/exp199 \\
        --gt ~/git/MarinFold/experiments/exp89_.../data/gt_universe.jsonl \\
        --out-per-rollout data/phase0_per_rollout.csv.gz \\
        --out-per-protein data/phase0_per_protein.csv \\
        --out-summary data/phase0_summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import consensus as C  # noqa: E402

GROUP_SIZES = (8, 16, 32)
N_ROUNDS = 5


def read_parquet_dir(prefix: str, pattern: str) -> pd.DataFrame:
    """Concatenate every parquet matching ``pattern`` under ``prefix``."""
    import fsspec
    import pyarrow.parquet as pq

    fs, root = fsspec.core.url_to_fs(prefix)
    paths = sorted(fs.glob(f"{root.rstrip('/')}/{pattern}"))
    if not paths:
        raise SystemExit(f"no parquet matching {pattern!r} under {prefix}")
    frames = []
    for path in paths:
        with fs.open(path, "rb") as fh:
            frames.append(pq.read_table(fh).to_pandas())
    return pd.concat(frames, ignore_index=True)


def verify_dump_matches_votes(dump: pd.DataFrame, votes: pd.DataFrame) -> int:
    """Hard cross-check: the per-rollout dump must sum to the vote matrix.

    The dump and the votes are written by the same loop but through different
    accumulators, so this is a real check on the dump rather than a tautology —
    and if it ever fails, every number downstream is measuring a different
    quantity from the one the eval reports. Fail loudly.
    """
    rebuilt = (dump.groupby(["dataset", "stem", "i", "j"]).size()
               .rename("votes_rebuilt").reset_index())
    units = dump[["dataset", "stem"]].drop_duplicates()
    subset = votes.merge(units, on=["dataset", "stem"], how="inner")
    merged = subset.merge(rebuilt, on=["dataset", "stem", "i", "j"], how="outer")
    bad = merged[merged["votes"].fillna(-1) != merged["votes_rebuilt"].fillna(-1)]
    if len(bad):
        raise SystemExit(
            f"!! dump does not reconstruct the vote matrix: {len(bad)} disagreeing pairs, "
            f"e.g.\n{bad.head(5)}"
        )
    return len(units)


def pair_sets_for(rows: pd.DataFrame, n_rollouts: int) -> list[set[tuple[int, int]]]:
    """One pair set per rollout index, including empty ones."""
    out: list[set[tuple[int, int]]] = [set() for _ in range(n_rollouts)]
    highest = int(rows["rollout"].max()) if len(rows) else -1
    if highest >= n_rollouts:
        raise SystemExit(
            f"dump holds rollout index {highest} but --n-rollouts is {n_rollouts}; "
            "the dump and the eval were generated with different group sizes, and "
            "every marginal below would be computed over a truncated group"
        )
    for k, i, j in zip(rows["rollout"].to_numpy(), rows["i"].to_numpy(), rows["j"].to_numpy()):
        out[int(k)].add((int(i), int(j)))
    return out


def tie_fraction(total: np.ndarray, n_true: int) -> float:
    """Fraction of the top-R selection settled by the stable sort's index order.

    Pairs tied on vote count with the last selected pair could equally well have
    been chosen; that share of the metric is arbitrary. High values mean the
    consensus is being read off a coarse, heavily-tied ranking.
    """
    if n_true <= 0 or total.size == 0:
        return math.nan
    top = min(int(n_true), int(total.size))
    order = np.argsort(-total, kind="mergesort")
    boundary = total[order[top - 1]]
    n_tied_total = int(np.count_nonzero(total == boundary))
    n_tied_in_top = int(np.count_nonzero(total[order[:top]] == boundary))
    if n_tied_total <= n_tied_in_top:
        return 0.0          # the whole tied block fits inside top-R; nothing arbitrary
    return n_tied_in_top / top


def subsample_marginals(
    votes: np.ndarray, is_true: np.ndarray, n_true: int, group: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Marginals as seen at group size ``group``, over ``N_ROUNDS`` shuffled rounds.

    Each round shuffles all rollouts and cuts them into DISJOINT groups of
    ``group``, so every rollout is measured once per round. Sampling independent
    random subsets instead would leave a long tail of rollouts in no subset at
    all (at ``group`` = 8 with 20 draws, roughly a fifth of them), and the
    estimator would quietly be uneven rather than loudly wrong.

    Returns:
        ``(first_draw, mean_over_rounds)``, both ``[n_rollouts]``. The first-draw
        array is what one training step actually observes; the mean is the
        estimator's ceiling if we could afford many draws. Reporting both is what
        separates "this quantity is unmeasurable at G" from "one draw of it is
        noisy".
    """
    n_rollouts = len(votes)
    first = np.full(n_rollouts, math.nan)
    total = np.zeros(n_rollouts)
    count = np.zeros(n_rollouts)
    if group > n_rollouts:
        return first, np.full(n_rollouts, math.nan)

    n_groups = n_rollouts // group
    for _ in range(N_ROUNDS):
        order = rng.permutation(n_rollouts)
        for g in range(n_groups):
            idx = order[g * group : (g + 1) * group]
            _, marg = C.loo_marginals(votes[idx], is_true, n_true)
            for slot, member in enumerate(idx):
                if math.isnan(first[member]):
                    first[member] = marg[slot]
                total[member] += marg[slot]
                count[member] += 1
    mean = np.where(count > 0, total / np.maximum(count, 1), math.nan)
    return first, mean


def within_protein_corr(frame: pd.DataFrame, a: str, b: str, method: str = "pearson") -> pd.Series:
    """Per-protein correlation of two per-rollout columns.

    Within-protein is the right unit: the RL group is one protein, so a pooled
    correlation would be dominated by between-protein differences in difficulty
    rather than by the thing the reward actually discriminates.
    """
    def one(g):
        x, y = g[a], g[b]
        if x.notna().sum() < 3 or x.nunique() < 2 or y.nunique() < 2:
            return math.nan
        return x.corr(y, method=method)

    # Subset to the two columns before apply: passing the grouping columns through
    # is deprecated in pandas 2.2+ and the warning is noise on every call.
    return frame.groupby(["dataset", "stem"])[[a, b]].apply(one)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="prefix holding shard-*-part-*.parquet and rollouts/dump-*.parquet")
    ap.add_argument("--gt", type=Path, required=True, help="exp89 gt_universe.jsonl")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-per-rollout", type=Path, required=True)
    ap.add_argument("--out-per-protein", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    a = ap.parse_args()

    votes_df = read_parquet_dir(a.scores, "shard-*-part-*.parquet")
    dump_df = read_parquet_dir(a.scores, "rollouts/dump-*.parquet")
    n_units = verify_dump_matches_votes(dump_df, votes_df)
    print(f"[phase0] dump reconstructs the vote matrix exactly for {n_units} protein(s)")

    gt_by_unit = {}
    for line in a.gt.open():
        rec = json.loads(line)
        gt_by_unit[(rec["dataset"], rec["stem"])] = rec

    rng = np.random.default_rng(a.seed)
    per_rollout: list[dict] = []
    per_protein: list[dict] = []

    for (dataset, stem), rows in dump_df.groupby(["dataset", "stem"], sort=True):
        rec = gt_by_unit.get((dataset, stem))
        if rec is None:
            print(f"  {dataset}/{stem}: not in the GT universe; skipping")
            continue
        length = int(rec["L"])
        pairs, position = C.candidate_index(length, resolved=rec["resolved"])
        gt = {
            (int(i), int(j)) for i, j, deg in rec["contacts"]
            if deg >= 0.001 and (int(j) - int(i)) >= C.MIN_SEP
        }
        is_true = C.truth_mask(pairs, gt)
        n_true = int(is_true.sum())
        if n_true <= 0 or len(pairs) == 0:
            continue

        sets = pair_sets_for(rows, a.n_rollouts)
        votes = C.vote_counts(sets, position, len(pairs))
        total = votes.sum(axis=0)

        consensus, marginals = C.loo_marginals(votes, is_true, n_true)
        diag = C.group_diagnostics(votes, is_true, n_true)
        diag.update(
            dataset=dataset, stem=stem, L=length,
            tie_fraction=tie_fraction(total, n_true),
            consensus_n100=consensus,
        )

        subs = {g: subsample_marginals(votes, is_true, n_true, g, rng) for g in GROUP_SIZES}
        for g, (first, mean) in subs.items():
            diag[f"frac_zero_marginal_g{g}"] = float(np.mean(first[~np.isnan(first)] == 0.0))
        diag["frac_zero_marginal_n100"] = float(np.mean(marginals == 0.0))
        per_protein.append(diag)

        for k in range(a.n_rollouts):
            row = {
                "dataset": dataset, "stem": stem, "rollout": k,
                "marginal_n100": float(marginals[k]),
                **C.rollout_precision_recall(votes[k], is_true, n_true),
            }
            for g, (first, mean) in subs.items():
                row[f"marginal_g{g}_first"] = float(first[k])
                row[f"marginal_g{g}_mean"] = float(mean[k])
            per_rollout.append(row)

    if not per_protein:
        raise SystemExit("no proteins scored — check --scores and --gt line up")

    roll = pd.DataFrame(per_rollout)
    prot = pd.DataFrame(per_protein)

    summary: list[dict] = []

    def record(name, series, note=""):
        s = pd.Series(series).dropna()
        summary.append(dict(metric=name, mean=float(s.mean()), median=float(s.median()),
                            p10=float(s.quantile(0.10)), p90=float(s.quantile(0.90)),
                            n=int(s.size), note=note))

    # Claim 1 — is the marginal just precision?
    record("corr_precision_vs_marginal_n100",
           within_protein_corr(roll, "precision", "marginal_n100"),
           "within-protein Pearson; near 1.0 means the doc term is redundant")
    record("spearman_precision_vs_marginal_n100",
           within_protein_corr(roll, "precision", "marginal_n100", method="spearman"))
    record("corr_recall_vs_marginal_n100",
           within_protein_corr(roll, "recall", "marginal_n100"))
    record("corr_npred_vs_marginal_n100",
           within_protein_corr(roll, "n_pred", "marginal_n100"))

    # Claim 2 — is it estimable at an affordable group size?
    for g in GROUP_SIZES:
        record(f"corr_marginal_g{g}_first_vs_n100",
               within_protein_corr(roll, f"marginal_g{g}_first", "marginal_n100"),
               "what one training step actually observes")
        record(f"corr_marginal_g{g}_mean_vs_n100",
               within_protein_corr(roll, f"marginal_g{g}_mean", "marginal_n100"),
               f"mean over {N_ROUNDS} rounds — the estimator's ceiling")
        record(f"frac_zero_marginal_g{g}", prot[f"frac_zero_marginal_g{g}"],
               "degenerate-signal check")
    record("frac_zero_marginal_n100", prot["frac_zero_marginal_n100"])

    # Metric structure, for the baseline the arms will be compared against.
    for col, note in [
        ("consensus_n100", "baseline consensus R-precision on the dumped subset"),
        ("union_over_r", "< 1.0 means top-R is padded with zero-vote pairs"),
        ("tie_fraction", "share of top-R decided by the stable sort's index order"),
        ("mean_jaccard", "inter-rollout agreement — the collapse detector"),
        ("vote_entropy", ""),
        ("mean_vote_top_r", ""),
        ("mean_pairs_per_rollout", ""),
    ]:
        record(col, prot[col], note)
    record("rollout_precision", roll["precision"], "single-rollout, for the #200 comparison")
    record("rollout_recall", roll["recall"])
    record("rollout_f1", roll["f1"])

    out = pd.DataFrame(summary)
    for path, frame in ((a.out_per_rollout, roll), (a.out_per_protein, prot), (a.out_summary, out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"wrote {len(frame)} rows -> {path}")

    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    gate_redundant = out.loc[out["metric"] == "corr_precision_vs_marginal_n100", "mean"].iloc[0]
    print(f"[phase0] GATE 1  corr(precision, marginal) = {gate_redundant:.3f} "
          f"-> {'REDUNDANT: drop the consensus arms' if gate_redundant > 0.85 else 'the doc term carries independent signal'}")
    for g in GROUP_SIZES:
        c = out.loc[out["metric"] == f"corr_marginal_g{g}_first_vs_n100", "mean"].iloc[0]
        print(f"[phase0] GATE 2  G={g:>2}: corr(single-draw marginal, n=100 marginal) = {c:.3f} "
              f"-> {'usable' if c > 0.5 else 'too noisy at this group size'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
