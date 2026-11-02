"""Summarize a Foldseek chain clustering into small committable artifacts.

Input is the two-column `easy-cluster` TSV (representative, member), gzipped,
as published by the phase 3 run to
s3://marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/cluster/chain_cluster.tsv.gz.

The TSV itself is 857,599 rows and does not belong in git. This writes three
things that do: the full size distribution (one row per distinct cluster size),
the largest clusters by name, and a summary JSON holding the numbers phase 4 and
phase 5 actually plan against.
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def read_clusters(path: Path) -> dict[str, str]:
    """Map each member accession to its representative, from the easy-cluster TSV."""
    rep: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            rep[row[1]] = row[0]
    return rep


def cluster_pairs(assignment: Path, split: str, missing: set[str],
                  rep: dict[str, str]) -> tuple[Counter[tuple[str, str]], dict[str, int]]:
    """Label every complex by the unordered pair of its two chains' clusters.

    This is the quantity phase 5 draws against: one complex per distinct cluster
    pair is the ceiling on the validation set. Complexes whose model 404'd during
    the fetch are excluded, as is any complex with a chain that never reached the
    clustering.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    tally = {"n_complexes": 0, "n_excluded_404": 0, "n_unmapped_chain": 0, "n_homodimers": 0}
    with gzip.open(assignment, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            tally["n_complexes"] += 1
            if row["model_entity_id"] in missing:
                tally["n_excluded_404"] += 1
                continue
            first, second = rep.get(row["uniprot_1"]), rep.get(row["uniprot_2"])
            if first is None or second is None:
                tally["n_unmapped_chain"] += 1
                continue
            if row["uniprot_1"] == row["uniprot_2"]:
                tally["n_homodimers"] += 1
            pairs[(first, second) if first <= second else (second, first)] += 1
    return pairs, tally


def coverage_points(sizes: list[int], fractions: tuple[float, ...]) -> dict[str, int]:
    """How many of the largest clusters it takes to cover each fraction of chains.

    Phase 5 draws one complex per cluster pair, so the shape of the head decides
    how much of the corpus a small draw can represent.
    """
    total = sum(sizes)
    points: dict[str, int] = {}
    running = 0
    ordered = sorted(sizes, reverse=True)
    targets = sorted(fractions)
    index = 0
    for rank, size in enumerate(ordered, start=1):
        running += size
        while index < len(targets) and running >= targets[index] * total:
            points[f"clusters_to_cover_{targets[index]:.2f}"] = rank
            index += 1
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True,
                        help="gzipped easy-cluster TSV (representative, member)")
    parser.add_argument("--out", type=Path, default=Path("data/curation"))
    parser.add_argument("--top", type=int, default=25,
                        help="how many of the largest clusters to name")
    parser.add_argument("--assignment", type=Path, default=None,
                        help="split assignment; supplied, the report also labels every "
                             "complex by its pair of chain clusters")
    parser.add_argument("--missing", type=Path, default=None,
                        help="models the fetch could not retrieve, excluded from pairs")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    rep = read_clusters(args.tsv)
    sizes = Counter(rep.values())
    counts = sorted(sizes.values(), reverse=True)
    n_chains = sum(counts)
    histogram = Counter(counts)

    dist_path = args.out / "chain_cluster_sizes.csv"
    with open(dist_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_size", "n_clusters", "n_chains", "cumulative_chains"])
        running = 0
        for size in sorted(histogram, reverse=True):
            chains = size * histogram[size]
            running += chains
            writer.writerow([size, histogram[size], chains, running])

    top_path = args.out / "chain_cluster_largest.csv"
    ordered = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(top_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "representative", "size", "share_of_chains"])
        for rank, (representative, size) in enumerate(ordered[:args.top], start=1):
            writer.writerow([rank, representative, size, round(size / n_chains, 6)])

    midpoint = sorted(counts)
    summary = {
        "n_chains": n_chains,
        "n_clusters": len(counts),
        "representative_fraction": round(len(counts) / n_chains, 6),
        "largest_cluster": counts[0],
        "median_cluster_size": midpoint[len(midpoint) // 2],
        "mean_cluster_size": round(n_chains / len(counts), 2),
        "n_singletons": histogram[1],
        "singleton_share_of_clusters": round(histogram[1] / len(counts), 4),
        "singleton_share_of_chains": round(histogram[1] / n_chains, 6),
        "n_clusters_ge_100": sum(1 for size in counts if size >= 100),
        "chains_in_clusters_ge_100": sum(size for size in counts if size >= 100),
        **coverage_points(counts, (0.50, 0.80, 0.90, 0.99)),
    }

    if args.assignment is not None:
        missing = {row["model_entity_id"]
                   for row in csv.DictReader(open(args.missing))} if args.missing else set()
        pairs, tally = cluster_pairs(args.assignment, args.split, missing, rep)
        pair_counts = sorted(pairs.values(), reverse=True)
        summary["pairs"] = {
            **tally,
            "n_labeled": sum(pair_counts),
            "n_distinct_pairs": len(pair_counts),
            "n_same_cluster_pairs": sum(1 for first, second in pairs if first == second),
            "n_pairs_seen_once": sum(1 for size in pair_counts if size == 1),
            "largest_pair": pair_counts[0],
            **{f"pairs_{key}": value
               for key, value in coverage_points(pair_counts, (0.50, 0.90)).items()},
        }
        pairs_path = args.out / "chain_cluster_pair_sizes.csv"
        pair_histogram = Counter(pair_counts)
        with open(pairs_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["complexes_in_pair", "n_pairs", "n_complexes"])
            for size in sorted(pair_histogram, reverse=True):
                writer.writerow([size, pair_histogram[size], size * pair_histogram[size]])
        print(f"wrote {pairs_path}")
    summary_path = args.out / "chain_cluster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {dist_path}, {top_path}, {summary_path}")


if __name__ == "__main__":
    main()
