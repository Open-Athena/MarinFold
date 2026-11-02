"""Label every val complex by the pair of clusters its two chains fall into.

Phase 4 of CURATION_PLAN.md. The label is `cluster_{min}_{max}` over the two
representatives, sorted, so a homodimer lands on `cluster_c_c` without a special
case. Phase 5 draws one complex per distinct label.

Every row of the split assignment appears in the output, carrying a `status` that
says whether it could be labeled:

- `labeled`: both chains have a representative.
- `unfetchable`: the model 404'd during the fetch (fetch_missing_404.csv) and no
  structure exists to draw.
- `unmapped_chain`: a chain never reached the clustering. Exactly one complex is
  in this state, and CLUSTERING_REPORT.md explains why.

Nothing is dropped, so the row count reconciles against the assignment and the
cost of each exclusion stays visible.
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

FIELDS = ["model_entity_id", "uniprot_1", "uniprot_2", "kind",
          "cluster_1", "cluster_2", "pair_label", "status"]


def read_representatives(path: Path) -> dict[str, str]:
    """Map each member accession to its cluster representative."""
    rep: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row:
                rep[row[1]] = row[0]
    return rep


def read_missing(path: Path) -> set[str]:
    """Model ids the fetch could not retrieve."""
    with open(path) as handle:
        return {row["model_entity_id"] for row in csv.DictReader(handle)}


def label(row: dict[str, str], rep: dict[str, str], missing: set[str]) -> dict[str, str]:
    """One output row: the complex, its two clusters, and its pair label."""
    first, second = row["uniprot_1"], row["uniprot_2"]
    out = {
        "model_entity_id": row["model_entity_id"],
        "uniprot_1": first,
        "uniprot_2": second,
        "kind": "homodimer" if first == second else "heterodimer",
        "cluster_1": "",
        "cluster_2": "",
        "pair_label": "",
        "status": "",
    }
    if row["model_entity_id"] in missing:
        out["status"] = "unfetchable"
        return out
    left, right = rep.get(first), rep.get(second)
    if left is None or right is None:
        out["status"] = "unmapped_chain"
        return out
    low, high = (left, right) if left <= right else (right, left)
    out["cluster_1"], out["cluster_2"] = low, high
    out["pair_label"] = f"cluster_{low}_{high}"
    out["status"] = "labeled"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path,
                        default=Path("data/curation/dimer_split_assignment_id30_cov50.csv.gz"))
    parser.add_argument("--clusters", type=Path,
                        default=Path("data/curation/chain_cluster.tsv.gz"))
    parser.add_argument("--missing", type=Path,
                        default=Path("data/curation/fetch_missing_404.csv"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", type=Path,
                        default=Path("data/curation/complex_cluster_pairs.csv.gz"))
    parser.add_argument("--index-out", type=Path,
                        default=Path("data/curation/cluster_pair_index.csv.gz"),
                        help="one row per distinct pair label, with its complex counts")
    args = parser.parse_args()

    rep = read_representatives(args.clusters)
    missing = read_missing(args.missing)
    print(f"{len(rep):,} chains mapped to {len(set(rep.values())):,} clusters, "
          f"{len(missing):,} models unfetchable")

    status = Counter()
    kinds = Counter()
    pair_total: Counter[str] = Counter()
    pair_homo: Counter[str] = Counter()
    seen_models = Counter()
    with gzip.open(args.assignment, "rt") as source, \
            gzip.open(args.out, "wt", newline="", compresslevel=6) as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        writer.writeheader()
        for row in csv.DictReader(source):
            if row["split"] != args.split:
                continue
            out = label(row, rep, missing)
            writer.writerow(out)
            status[out["status"]] += 1
            seen_models[out["model_entity_id"]] += 1
            if out["status"] != "labeled":
                continue
            kinds[out["kind"]] += 1
            pair_total[out["pair_label"]] += 1
            if out["kind"] == "homodimer":
                pair_homo[out["pair_label"]] += 1

    with gzip.open(args.index_out, "wt", newline="", compresslevel=6) as sink:
        writer = csv.writer(sink)
        writer.writerow(["pair_label", "cluster_1", "cluster_2", "same_cluster",
                         "n_complexes", "n_homodimers"])
        for pair_label, count in sorted(pair_total.items(), key=lambda kv: (-kv[1], kv[0])):
            low, high = pair_label[len("cluster_"):].rsplit("_", 1)
            writer.writerow([pair_label, low, high, int(low == high),
                             count, pair_homo[pair_label]])

    counts = sorted(pair_total.values(), reverse=True)
    summary = {
        "split": args.split,
        "n_rows": sum(status.values()),
        "n_duplicate_model_ids": sum(1 for n in seen_models.values() if n > 1),
        **{f"n_{key}": value for key, value in sorted(status.items())},
        "n_homodimers": kinds["homodimer"],
        "n_heterodimers": kinds["heterodimer"],
        "n_distinct_pairs": len(counts),
        "n_same_cluster_pairs": sum(1 for pair in pair_total if
                                    pair[len("cluster_"):].rsplit("_", 1)[0] ==
                                    pair[len("cluster_"):].rsplit("_", 1)[1]),
        "n_pairs_seen_once": sum(1 for count in counts if count == 1),
        "largest_pair": counts[0],
        "median_complexes_per_pair": sorted(counts)[len(counts) // 2],
    }
    summary_path = args.out.parent / "cluster_pair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}, {args.index_out}, {summary_path}")


if __name__ == "__main__":
    main()
