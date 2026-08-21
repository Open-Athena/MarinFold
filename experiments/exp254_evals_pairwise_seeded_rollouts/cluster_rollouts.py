# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Do the 100 rollouts hold more than one distinct contact map?

The single consensus averages all 100 rollouts into one ranking. If the rollouts
carry genuinely different hypotheses about the fold -- not just noise around one
-- then clustering them by contact-set similarity and taking a consensus *per
cluster* yields K candidate maps, and the best of those K could be much better
than the one map the pooled vote produces.

This measures the ceiling of that idea at the contact level, which is the cheap
half and the half that gates the rest: if oracle-best-of-K over cluster
consensuses is not much better than the single consensus, then no downstream
selector -- a folding model's confidence head, the LM's own likelihood -- can
make it pay, because the candidates do not differ in quality.

Three numbers per K, all all-range R-precision, all on the same 97 proteins:

``single``      the pooled consensus over all 100 rollouts (exp82's recipe)
``oracle@K``    the best of the K cluster consensuses, chosen with ground truth
``largest``     the consensus of the biggest cluster -- a selector that needs no
                ground truth, and the obvious thing to try first
``mean@K``      the average cluster consensus, i.e. what picking blind gets you

For reference, exp254 measured oracle best over the 100 *individual* rollouts at
0.5341 against a 0.5217 single consensus. A cluster consensus averages within
its cluster, so it should beat an individual rollout; the question is by how
much, and whether the spread across clusters is real.

Three partitioning methods, because a negative result here would otherwise be
one clustering choice away from meaningless:

``average``   average-linkage agglomerative on Jaccard distance -- the natural
              choice, free to make clusters as lopsided as the data is
``kmeans``    k-means on binary pair-membership vectors, which tends to split
              more evenly and so cannot hide a real mode inside one giant blob
``random``    **the control.** An equal-sized random partition of the 100
              rollouts, which by construction finds no structure at all. If a
              real clustering does not beat this, the K candidate maps are
              subsamples of one distribution rather than distinct hypotheses,
              and no selector downstream can turn them into an improvement.

The metric implementation is exp89's, imported from ``build_metrics`` rather
than re-derived.

    uv run python cluster_rollouts.py --run /data/exp_contactseed/run --out data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from build_metrics import (
    RANGES,
    load_detail,
    metric_rows,
    resolved_pairs,
    true_matrix,
)
from common import EXPECTED_UNITS, load_ground_truth, load_targets

#: Cluster counts to sweep. K=1 is the pooled consensus and is scored separately.
K_VALUES = (2, 3, 5, 10, 20)
METHODS = ("average", "kmeans", "random")


def jaccard_distance(sets: list[set]) -> np.ndarray:
    """Condensed pairwise Jaccard distance between rollout contact sets."""
    n = len(sets)
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            union = len(sets[a] | sets[b])
            similarity = len(sets[a] & sets[b]) / union if union else 0.0
            out[a, b] = out[b, a] = 1.0 - similarity
    return squareform(out, checks=False)


def partition(method: str, k: int, tree, sets: list[set], rng) -> np.ndarray:
    """Labels in ``[0, k)`` for each rollout under one partitioning method."""
    n = len(sets)
    if method == "average":
        return fcluster(tree, t=k, criterion="maxclust")
    if method == "random":
        labels = np.arange(n) % k
        rng.shuffle(labels)
        return labels
    if method != "kmeans":
        raise ValueError(f"unknown method {method!r}")
    vocabulary = sorted(set().union(*sets)) if sets else []
    if len(vocabulary) == 0:
        return np.zeros(n, dtype=int)
    index = {pair: c for c, pair in enumerate(vocabulary)}
    design = np.zeros((n, len(vocabulary)), dtype=np.float32)
    for row, members in enumerate(sets):
        for pair in members:
            design[row, index[pair]] = 1.0
    return KMeans(n_clusters=min(k, n), n_init=4, random_state=254).fit_predict(design)


def consensus_matrix(members, pairs_by_rollout, L: int) -> np.ndarray:
    """Vote matrix over a subset of the rollouts."""
    matrix = np.zeros((L, L))
    for rollout in members:
        for i, j in pairs_by_rollout[rollout]:
            matrix[i, j] += 1.0
    return matrix + matrix.T


def r_precision(score: np.ndarray, record: dict, tmat, pi, pj, psep) -> float:
    """All-range R-precision under exp89's metric."""
    for row in metric_rows(score, tmat, pi, pj, psep, record["L"], with_precision=True):
        if row["range"] == "all" and row["cut"] == "R":
            return row["precision"]
    raise AssertionError("metric_rows returned no all/R row")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="iid",
                    help="which arm's rollouts to cluster (default the control)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()
    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS
    detail = load_detail(args.run / args.arm)

    rows = []
    for target in targets:
        record = gt[(target.dataset, target.stem)]
        L = record["L"]
        mine = detail[(detail["dataset"] == target.dataset)
                      & (detail["stem"] == target.stem)]
        pairs_by_rollout = {
            int(r): list(zip(group["i"].astype(int), group["j"].astype(int)))
            for r, group in mine.groupby("rollout")
        }
        rollouts = sorted(pairs_by_rollout)
        sets = [set(pairs_by_rollout[r]) for r in rollouts]

        tmat = true_matrix(L, record["contacts"])
        pi, pj, psep = resolved_pairs(np.asarray(record["resolved"], dtype=np.int64))
        single = r_precision(consensus_matrix(rollouts, pairs_by_rollout, L),
                             record, tmat, pi, pj, psep)

        distance = jaccard_distance(sets)
        tree = linkage(distance, method="average")
        rng = np.random.default_rng(254)
        row = dict(dataset=target.dataset, stem=target.stem, L=L,
                   mean_jaccard=float(1.0 - distance.mean()), single=single)
        for method in METHODS:
            for k in K_VALUES:
                labels = partition(method, k, tree, sets, rng)
                scores, sizes = [], []
                for label in np.unique(labels):
                    members = [rollouts[m] for m in np.flatnonzero(labels == label)]
                    sizes.append(len(members))
                    scores.append(r_precision(
                        consensus_matrix(members, pairs_by_rollout, L),
                        record, tmat, pi, pj, psep))
                scores, sizes = np.asarray(scores), np.asarray(sizes)
                row[f"{method}_oracle@{k}"] = float(scores.max())
                row[f"{method}_mean@{k}"] = float(scores.mean())
                row[f"{method}_largest@{k}"] = float(scores[int(np.argmax(sizes))])
                row[f"{method}_largest_size@{k}"] = int(sizes.max())
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "exp254_cluster_per_protein.csv", index=False)

    summary = [dict(method="-", readout="single consensus (K=1)",
                    value=float(frame["single"].mean()))]
    for method in METHODS:
        for k in K_VALUES:
            for kind in ("oracle", "largest", "mean"):
                column = f"{method}_{kind}@{k}"
                summary.append(dict(
                    method=method, readout=f"{kind}@K={k}",
                    value=float(frame[column].mean()),
                    delta_vs_single=float((frame[column] - frame["single"]).mean()),
                    largest_cluster=float(frame[f"{method}_largest_size@{k}"].mean())))
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.out / "exp254_cluster_summary.csv", index=False)

    print(f"[cluster] mean pairwise Jaccard between rollouts: "
          f"{frame['mean_jaccard'].mean():.3f}")
    print(f"[cluster] average-linkage largest cluster at K=5: "
          f"{frame['average_largest_size@5'].mean():.0f}/100 rollouts")
    print("\n[cluster] all-range R-precision (n=97):")
    for method in METHODS:
        print(f"\n--- {method} ---")
        print(summary_frame[summary_frame.method == method]
              .drop(columns="method").round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
