# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare pooled consensus with consensuses of rollout clusters.

For each protein, partition its 100 contact lists using k-means, average-linkage
Jaccard clustering, or an equal-sized random partition. Score the pooled map,
the best cluster (ground-truth oracle), the largest cluster, and the mean cluster.
The random partition controls for splitting the sample without fitting clusters.

An oracle bounds selectors only for these candidate maps under this contact
metric. It is not an upper bound on downstream folding accuracy or on other
clustering methods. The saved k-means candidates have positive oracle headroom;
imbalanced average-linkage clusters do not prove a single posterior fold mode.

Consensus uses exp89's resolved-pair metric. Individual-rollout historical
precision has a different denominator; see audit_conclusions.py for fixed-R
comparisons.

    uv run python cluster_rollouts.py --run /path/to/inputs --out data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from build_metrics import (
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
