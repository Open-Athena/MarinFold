"""Choose the candidate complexes phase 5 will measure an interface for.

Phase 5 draws one complex per cluster pair. Measuring the interface is the only
step that needs the structures, so rather than computing it for all 880,247
labeled complexes we take the best few per pair and measure only those.

Ranking is by ipTM descending, from the confidence tables written in phase 1: a
validation set built from predicted structures should be represented by the
prediction the model was most confident about, and ipTM is the interface-specific
term. Ties break on `model_entity_id` so the choice is reproducible.

`--per-pair` candidates are kept for each pair, so a pair whose best complex fails
the interface filter still has fallbacks without a second pass over the shards.
The 8,621 pairs that hold exactly one complex have no fallback by construction.
"""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["model_entity_id", "pair_label", "kind", "rank", "iptm",
          "pdockq2_max", "n_interactions", "n_in_pair"]


def read_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    """Labeled complexes grouped by pair label."""
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "labeled":
                pairs[row["pair_label"]].append(row)
    return pairs


def read_confidence(paths: list[Path], wanted: set[str]) -> dict[str, dict[str, float]]:
    """ipTM, pDockQ2 and interaction count for the complexes we care about."""
    scores: dict[str, dict[str, float]] = {}
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for row in csv.DictReader(handle):
                model = row["model_entity_id"]
                if model not in wanted:
                    continue
                scores[model] = {
                    "iptm": float(row["ipTM"]),
                    "pdockq2_max": float(row["pDockQ2_max"]),
                    "n_interactions": float(row["n_interactions"]),
                }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=Path("data/curation/complex_cluster_pairs.csv.gz"))
    parser.add_argument("--confidence", type=Path, nargs="+",
                        default=[Path("data/curation/confident_homodimers.csv.gz"),
                                 Path("data/curation/confident_heterodimers.csv.gz")])
    parser.add_argument("--per-pair", type=int, default=3)
    parser.add_argument("--out", type=Path,
                        default=Path("data/curation/phase5_candidates.csv.gz"))
    args = parser.parse_args()

    pairs = read_labels(args.labels)
    wanted = {row["model_entity_id"] for rows in pairs.values() for row in rows}
    print(f"{len(pairs):,} pairs over {len(wanted):,} labeled complexes")

    scores = read_confidence(args.confidence, wanted)
    print(f"{len(scores):,} complexes carry a confidence row")
    unscored = len(wanted) - len(scores)
    if unscored:
        # Not fatal, but it decides ranking, so it has to be visible.
        print(f"WARNING {unscored:,} complexes have no confidence row and rank last")

    depth = Counter()
    written = 0
    with gzip.open(args.out, "wt", newline="", compresslevel=6) as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        writer.writeheader()
        for pair_label, rows in sorted(pairs.items()):
            ordered = sorted(
                rows,
                key=lambda row: (-scores.get(row["model_entity_id"], {}).get("iptm", -1.0),
                                 row["model_entity_id"]))
            for rank, row in enumerate(ordered[:args.per_pair], start=1):
                score = scores.get(row["model_entity_id"], {})
                writer.writerow({
                    "model_entity_id": row["model_entity_id"],
                    "pair_label": pair_label,
                    "kind": row["kind"],
                    "rank": rank,
                    "iptm": score.get("iptm", ""),
                    "pdockq2_max": score.get("pdockq2_max", ""),
                    "n_interactions": score.get("n_interactions", ""),
                    "n_in_pair": len(rows),
                })
                written += 1
            depth[min(len(rows), args.per_pair)] += 1

    summary = {
        "per_pair": args.per_pair,
        "n_pairs": len(pairs),
        "n_candidates": written,
        "n_labeled_complexes": len(wanted),
        "n_without_confidence": unscored,
        "pairs_by_candidate_count": {str(k): v for k, v in sorted(depth.items())},
        "fraction_of_complexes_measured": round(written / len(wanted), 4),
    }
    summary_path = args.out.parent / "phase5_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}, {summary_path}")


if __name__ == "__main__":
    main()
