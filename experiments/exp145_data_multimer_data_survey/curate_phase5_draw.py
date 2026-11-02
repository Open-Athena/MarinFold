"""Draw the validation set: one complex per cluster pair.

Phase 5 step 4 of CURATION_PLAN.md. Candidates were ranked by ipTM in
curate_phase5_candidates.py and measured in curate_phase5_interfaces.py; this takes
the best-ranked candidate of each pair that clears PINDER's min_interface_length,
and reports what each criterion cost.

A pair is lost when every candidate measured for it fails the interface filter. For
the 8,621 pairs holding a single complex that means one failure is fatal, which is
the price of measuring candidates rather than the whole corpus, and the summary
names those pairs so the cost is checkable rather than asserted.
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ["model_entity_id", "pair_label", "cluster_1", "cluster_2", "kind",
          "uniprot_1", "uniprot_2", "iptm", "pdockq2_max",
          "interface_a", "interface_b", "min_side", "drawn_rank", "n_in_pair"]


def read_candidates(path: Path) -> dict[str, dict[str, str]]:
    with gzip.open(path, "rt") as handle:
        return {row["model_entity_id"]: row for row in csv.DictReader(handle)}


def read_interfaces(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return {row["model_entity_id"]: row for row in csv.DictReader(handle)}


def read_complexes(path: Path) -> dict[str, dict[str, str]]:
    """Labeled complexes, keyed by model id, for the chain identities."""
    rows: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "labeled":
                rows.setdefault(row["model_entity_id"], row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=Path("data/curation/phase5_candidates.csv.gz"))
    parser.add_argument("--interfaces", type=Path,
                        default=Path("data/curation/phase5_interfaces.csv.gz"))
    parser.add_argument("--labels", type=Path,
                        default=Path("data/curation/complex_cluster_pairs.csv.gz"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/curation/val_set_dimers.csv"))
    args = parser.parse_args()

    candidates = read_candidates(args.candidates)
    interfaces = read_interfaces(args.interfaces)
    complexes = read_complexes(args.labels)
    missing = set(candidates) - set(interfaces)
    if missing:
        raise SystemExit(f"{len(missing):,} candidates were never measured, "
                         f"first: {sorted(missing)[:3]}")

    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for model, row in candidates.items():
        by_pair[row["pair_label"]].append(row)

    drawn: list[dict] = []
    lost: list[str] = []
    thin = 0
    for pair_label, rows in sorted(by_pair.items()):
        ordered = sorted(rows, key=lambda row: int(row["rank"]))
        pick = None
        for row in ordered:
            measured = interfaces[row["model_entity_id"]]
            if measured["passes"] == "1":
                pick = (row, measured)
                break
            thin += 1
        if pick is None:
            lost.append(pair_label)
            continue
        row, measured = pick
        complex_row = complexes[row["model_entity_id"]]
        drawn.append({
            "model_entity_id": row["model_entity_id"],
            "pair_label": pair_label,
            "cluster_1": complex_row["cluster_1"],
            "cluster_2": complex_row["cluster_2"],
            "kind": row["kind"],
            "uniprot_1": complex_row["uniprot_1"],
            "uniprot_2": complex_row["uniprot_2"],
            "iptm": row["iptm"],
            "pdockq2_max": row["pdockq2_max"],
            "interface_a": measured["interface_a"],
            "interface_b": measured["interface_b"],
            "min_side": measured["min_side"],
            "drawn_rank": row["rank"],
            "n_in_pair": row["n_in_pair"],
        })

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(drawn)

    n_labeled = len(complexes)
    homodimers = sum(1 for row in drawn if row["kind"] == "homodimer")
    summary = {
        "n_cluster_pairs": len(by_pair),
        "n_drawn": len(drawn),
        "n_homodimers_drawn": homodimers,
        "n_heterodimers_drawn": len(drawn) - homodimers,
        "n_pairs_lost_to_interface": len(lost),
        "n_candidates_failing_interface": thin,
        "n_drawn_at_rank_1": sum(1 for row in drawn if row["drawn_rank"] == "1"),
        "n_labeled_complexes": n_labeled,
        "n_discarded_sharing_a_pair": n_labeled - len(drawn) - thin,
        "median_min_side": sorted(int(row["min_side"]) for row in drawn)[len(drawn) // 2],
        "median_iptm": sorted(float(row["iptm"]) for row in drawn)[len(drawn) // 2],
    }
    if lost:
        lost_path = args.out.parent / "val_set_pairs_lost.csv"
        lost_path.write_text("pair_label\n" + "\n".join(lost) + "\n")
        summary["lost_pairs_listed_in"] = str(lost_path)
    summary_path = args.out.parent / "val_set_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}, {summary_path}")


if __name__ == "__main__":
    main()
