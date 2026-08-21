# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Can #211's geometric self-consistency pick the best cluster consensus?

`cluster_rollouts.py` established the prize: clustering the 100 rollouts and
taking a consensus per cluster gives K candidate contact maps whose best is worth
**+0.0158** R-precision over the single pooled consensus at K=10 — but every
selector needing no ground truth (largest cluster, blind average) *loses* to just
taking the single consensus. This asks whether #211's reference-free embeddability
residual is the selector that works.

**The prior is negative and it is quantified.** #211 measured exactly this on
individual rollouts: Spearman rho(excess, precision) within a protein averaged
**-0.0175**, useful on 51.8 % of proteins, and selecting the most-consistent of 30
rollouts captured **8 %** of the available oracle headroom. Its diagnosis was that
the score is **sequence-blind** -- a decoy protein's true contact map scores as
well as the real one -- so it cannot tell a coherent wrong fold from a coherent
right one.

Two reasons to run it anyway. A cluster consensus is a different object from a
single rollout: it is an aggregate, and clusters whose members agree more should
produce more coherent consensuses, which is the mechanism #211 never had a chance
to test. And the answer is cheap and gates a much more expensive question -- if
this cannot rank K candidate maps, neither a folding model's confidence head nor
any other downstream selector is likely to, and the whole cluster-and-fold idea
can be dropped for the price of an afternoon rather than K x the folding compute.

Every set for one protein is scored in **one** `embed_residual` call. #211's
batching is deliberate about this: a batch shares one RNG stream, so every
candidate faces the same draw of optimization landscapes, which is what makes the
within-protein comparison paired.

All sets are cut at the same R, so `contact_excess` is comparable across them
without normalisation and no candidate is favoured by being shorter.

    uv run python cluster_consistency.py --run /data/exp_contactseed/run --out data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr

from build_metrics import RANGES, load_detail, resolved_pairs, true_matrix
from cluster_rollouts import consensus_matrix, jaccard_distance, partition, r_precision
from common import EXPECTED_UNITS, load_ground_truth, load_targets

# exp211's scorer, on its SECOND use case. Per experiments/AGENTS.md that is the
# trigger to promote it into an `evals/` kind library; doing so means creating
# that library and rewriting a merged experiment's imports, so it is flagged in
# this experiment's README rather than done unilaterally here.
EXP211 = (Path(__file__).resolve().parents[1]
          / "exp211_evals_contact_set_3d_self_consistency")
sys.path.insert(0, str(EXP211))

from consistency import contact_matrix, embed_residual  # noqa: E402
from run_gt_gate import bounds_from_json  # noqa: E402

K_VALUES = (5, 10)
METHOD = "kmeans"
#: #211 found the metric blind below this length ( +0.0029 at L < 100 ).
MIN_LENGTH_FOR_HEADLINE = 100


def top_pairs(matrix: np.ndarray, n: int, resolved_mask: np.ndarray, min_sep: int):
    """The ``n`` highest-voted pairs, restricted to resolved residues."""
    L = matrix.shape[0]
    ii, jj = np.triu_indices(L, k=min_sep)
    keep = resolved_mask[ii] & resolved_mask[jj]
    ii, jj = ii[keep], jj[keep]
    order = np.argsort(-matrix[ii, jj], kind="mergesort")[:n]
    return list(zip(ii[order].tolist(), jj[order].tolist()))


def chunk_by_pairs(n_sets: int, length: int, n_restarts: int, max_pairs: int) -> int:
    """#211's memory rule for how many sets fit in one embed call."""
    per_row = max(length * length // 2, 1)
    rows = max(int(max_pairs // per_row), 1)
    return max(rows // max(n_restarts, 1), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="iid")
    ap.add_argument("--bounds", type=Path, default=EXP211 / "data/bounds.json")
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max-pairs", type=int, default=40_000_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="re-summarise the saved per-set scores instead of "
                         "re-running the embedder, which is the expensive part")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scores_path = args.out / "exp254_cluster_consistency.csv.gz"
    if args.reuse:
        summarise(pd.read_csv(scores_path), args.out)
        return 0
    bounds = bounds_from_json(args.bounds)
    gt = load_ground_truth()
    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS
    detail = load_detail(args.run / args.arm)
    if args.limit:
        targets = targets[: args.limit]

    lo, _ = RANGES["all"]
    rows = []
    for n, target in enumerate(targets):
        record = gt[(target.dataset, target.stem)]
        L = record["L"]
        resolved = np.asarray(record["resolved"], dtype=np.int64)
        resolved_mask = np.zeros(L, dtype=bool)
        resolved_mask[resolved] = True
        tmat = true_matrix(L, record["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        n_true = int(tmat[pi[psep >= lo], pj[psep >= lo]].sum())
        if n_true <= 0:
            continue
        # A sequence-gap proxy for #211's geometric chain-break gate: we have no
        # coordinates here, only which residues resolved.
        has_gap = bool(np.any(np.diff(resolved) > 1))

        mine = detail[(detail["dataset"] == target.dataset)
                      & (detail["stem"] == target.stem)]
        pairs_by_rollout = {
            int(r): list(zip(group["i"].astype(int), group["j"].astype(int)))
            for r, group in mine.groupby("rollout")
        }
        rollouts = sorted(pairs_by_rollout)
        sets = [set(pairs_by_rollout[r]) for r in rollouts]
        tree = linkage(jaccard_distance(sets), method="average")
        rng = np.random.default_rng(254)

        named = [("single", 0, 1,
                  consensus_matrix(rollouts, pairs_by_rollout, L), len(rollouts))]
        for k in K_VALUES:
            labels = partition(METHOD, k, tree, sets, rng)
            for index, label in enumerate(np.unique(labels)):
                members = [rollouts[m] for m in np.flatnonzero(labels == label)]
                named.append((f"kmeans{k}", k, index,
                              consensus_matrix(members, pairs_by_rollout, L),
                              len(members)))

        candidates = []
        for kind, k, index, matrix, size in named:
            pairs = top_pairs(matrix, n_true, resolved_mask, lo)
            if len(pairs) < 3:          # #211's embedder needs a real set
                continue
            candidates.append(dict(
                kind=kind, k=k, index=index, size=size, pairs=pairs,
                precision=r_precision(matrix, record, tmat, pi, pj, psep)))

        step = chunk_by_pairs(len(candidates), L, args.n_restarts, args.max_pairs)
        for start in range(0, len(candidates), step):
            block = candidates[start:start + step]
            masks = np.stack([contact_matrix(c["pairs"], L) for c in block])
            scored = embed_residual(masks, bounds, n_restarts=args.n_restarts,
                                    iters=args.iters, seed=n * 977 + start,
                                    device=args.device)
            for candidate, result in zip(block, scored):
                rows.append(dict(
                    dataset=target.dataset, stem=target.stem, L=L,
                    n_true=n_true, has_gap=has_gap,
                    kind=candidate["kind"], k=candidate["k"],
                    cluster=candidate["index"], cluster_size=candidate["size"],
                    n_pairs=len(candidate["pairs"]),
                    precision=candidate["precision"], **result))
        print(f"[consistency] [{n + 1}/{len(targets)}] {target.stem} L={L} "
              f"{len(candidates)} sets", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(scores_path, index=False)
    summarise(frame, args.out)
    return 0


def summarise(frame: pd.DataFrame, out: Path) -> None:
    """Compare the selectors against the two baselines that matter.

    ``single`` is the pooled consensus over all 100 rollouts — the thing already
    in production, and the bar any cluster-and-select scheme has to clear.
    ``blind`` is the mean cluster consensus, i.e. what picking one at random
    gets. A selector that beats ``blind`` is doing real work; only one that
    beats ``single`` is worth deploying, and the two questions have different
    answers here.

    Sign convention: ``contact_excess`` is a violation, so **lower is more
    self-consistent**. If consistency predicted accuracy, rho(excess, precision)
    would be *negative*, which is what ``rho_predictive_frac`` counts.
    """
    summary = []
    for scope, subset in (("all proteins", frame),
                          (f"L>={MIN_LENGTH_FOR_HEADLINE}, no gap",
                           frame[(frame.L >= MIN_LENGTH_FOR_HEADLINE) & ~frame.has_gap])):
        single = subset[subset.kind == "single"].set_index("stem")["precision"]
        for k in K_VALUES:
            clusters = subset[subset.kind == f"kmeans{k}"]
            picked, oracle, largest, blind, rhos = [], [], [], [], []
            for stem, group in clusters.groupby("stem"):
                if len(group) < 2:
                    continue
                best = group.loc[group["contact_excess"].idxmin()]
                picked.append(best["precision"])
                oracle.append(group["precision"].max())
                largest.append(group.loc[group["cluster_size"].idxmax(), "precision"])
                blind.append(group["precision"].mean())
                rho = spearmanr(group["contact_excess"], group["precision"]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
            stems = [s for s, g in clusters.groupby("stem") if len(g) >= 2]
            if not stems:
                # An empty scope is a real outcome (no protein qualified), but a
                # row of NaNs reads like a failed computation. Say it instead.
                print(f"[consistency] scope {scope!r} K={k}: no qualifying proteins")
                continue
            base = single.reindex(stems).to_numpy()
            summary.append(dict(
                scope=scope, K=k, n_proteins=len(stems),
                single=float(base.mean()),
                blind_cluster=float(np.mean(blind)),
                largest_cluster=float(np.mean(largest)),
                most_consistent=float(np.mean(picked)),
                oracle_cluster=float(np.mean(oracle)),
                vs_single=float(np.mean(picked) - base.mean()),
                vs_blind=float(np.mean(picked) - np.mean(blind)),
                oracle_vs_single=float(np.mean(oracle) - base.mean()),
                mean_rho=float(np.mean(rhos)),
                rho_predictive_frac=float(np.mean([r < 0 for r in rhos])),
            ))
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(out / "exp254_cluster_consistency_summary.csv", index=False)
    print("\n[consistency] can #211's residual pick the best cluster consensus?")
    print(summary_frame.round(4).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
