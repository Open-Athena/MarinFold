"""Plot the val-side chain-cluster size distribution from the report CSVs.

Left panel: cluster size against rank, log-log, which is where the head-heaviness
shows. Right panel: what fraction of chains the largest N clusters cover, which is
the number phase 5's draw depends on.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg selection)


def expand(path: Path) -> np.ndarray:
    """Cluster sizes, descending, one entry per cluster, from the histogram CSV."""
    sizes: list[int] = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            sizes.extend([int(row["cluster_size"])] * int(row["n_clusters"]))
    return np.array(sorted(sizes, reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=Path,
                        default=Path("data/curation/chain_cluster_sizes.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/curation/chain_cluster_sizes.png"))
    args = parser.parse_args()

    sizes = expand(args.sizes)
    rank = np.arange(1, len(sizes) + 1)
    coverage = np.cumsum(sizes) / sizes.sum()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.loglog(rank, sizes, lw=1.4, color="#22456b")
    left.set_xlabel("cluster rank")
    left.set_ylabel("chains in cluster")
    left.set_title(f"{len(sizes):,} clusters over {sizes.sum():,} chains")
    left.axhline(1, color="#999999", lw=0.8, ls=":")
    left.annotate(f"largest {sizes[0]:,}", xy=(1, sizes[0]), xytext=(3, sizes[0] * 0.45),
                  fontsize=9, color="#22456b")
    singletons = int((sizes == 1).sum())
    left.annotate(f"{singletons:,} singletons\n({singletons / len(sizes):.0%} of clusters, "
                  f"{singletons / sizes.sum():.1%} of chains)",
                  xy=(len(sizes), 1), xytext=(len(sizes) * 0.02, 1.6),
                  fontsize=9, color="#666666")
    left.grid(alpha=0.25, which="both")

    right.semilogx(rank, coverage, lw=1.6, color="#7a3b2e")
    for fraction in (0.5, 0.9):
        at = int(np.searchsorted(coverage, fraction) + 1)
        right.axhline(fraction, color="#999999", lw=0.8, ls=":")
        right.annotate(f"{fraction:.0%} of chains in the largest {at:,}",
                       xy=(at, fraction), xytext=(1.4, fraction - 0.075), fontsize=9,
                       color="#7a3b2e")
    right.set_xlabel("largest N clusters")
    right.set_ylabel("fraction of chains covered")
    right.set_ylim(0, 1.02)
    right.set_title("coverage by the head of the distribution")
    right.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
