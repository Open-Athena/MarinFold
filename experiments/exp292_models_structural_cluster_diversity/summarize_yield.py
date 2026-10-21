"""Summarize threshold sensitivity without treating curation hits as training data.

The candidate reservoir and all coverage/length rules stay fixed. The optional
core diagnostic reruns greedy selection using max(whole-chain TM, core TM), so
rejecting a core-covered candidate cannot hide a different admissible member.
These are descriptive rates for the sampled clusters, before domain review and
final sequence exclusion. Population weights, if supplied, describe only the
documented sampling frame; they do not justify a production-corpus size claim.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from structure_audit import read_csv, select_candidates, write_csv


def measured_pairs(path: Path) -> list[dict]:
    """Restore numeric measurements, preserving missing core scores explicitly."""
    identifiers = {"struct_cluster_id", "entry_a", "entry_b", "anchor_a", "anchor_b"}
    return [
        {k: v if k in identifiers else float(v) if v else None for k, v in r.items()}
        for r in read_csv(path)
    ]


def main() -> None:
    """Write per-stratum yield and selected identities for each diagnostic policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.audit / "sample.csv")
    pairs = measured_pairs(args.audit / "pairs.csv")
    clusters = read_csv(args.audit / "clusters.csv")
    strata = {
        r.get("struct_cluster_id", r.get("cluster_id")): (
            r["length_bin"],
            r["size_bin"],
        )
        for r in clusters
    }
    if len(strata) != len(clusters):
        raise ValueError("Cluster manifest contains duplicate identifiers")
    if not {r["struct_cluster_id"] for r in rows} <= strata.keys():
        raise ValueError("Audited structures are absent from the cluster manifest")
    population = {}
    if args.population:
        population = {
            (r["length_bin"], r["size_bin"]): float(r["eligible_clusters"])
            for r in read_csv(args.population)
        }
        if not set(strata.values()) <= population.keys():
            raise ValueError("Population weights omit a sampled stratum")
    core_pairs = [
        {
            **p,
            "tm_max": max(p["tm_max"], p["core_tm_max"])
            if p["core_tm_max"] is not None
            else 1.0,
        }
        for p in pairs
    ]
    sample_counts = Counter(strata.values())
    passing_counts = Counter(strata[c] for c in {r["struct_cluster_id"] for r in rows})
    candidate_counts = Counter(
        strata[r["struct_cluster_id"]] for r in rows if r["is_anchor"].lower() != "true"
    )
    summaries, identities, totals = [], [], []
    for mode, measurements in [("whole_chain", pairs), ("whole_and_core", core_pairs)]:
        for threshold in [0.7, 0.8, 0.9]:
            selected = [
                r
                for r in select_candidates(rows, measurements, threshold)
                if r["selected_order"] > 0
            ]
            selected_counts = Counter(r["struct_cluster_id"] for r in selected)
            bins = defaultdict(list)
            for cluster, stratum in strata.items():
                bins[stratum].append(selected_counts[cluster])
            for stratum, counts in sorted(bins.items()):
                summaries.append(
                    {
                        "mode": mode,
                        "tm_threshold": threshold,
                        "length_bin": stratum[0],
                        "size_bin": stratum[1],
                        "sample_clusters": sample_counts[stratum],
                        "clusters_after_sequence_quality": passing_counts[stratum],
                        "candidates_after_quality": candidate_counts[stratum],
                        "clusters_with_hit": sum(n > 0 for n in counts),
                        "selected_additions": sum(counts),
                        "additions_per_sampled_cluster": sum(counts) / len(counts),
                        "frame_clusters": population.get(stratum),
                        "frame_projected_additions": population[stratum]
                        * sum(counts)
                        / len(counts)
                        if stratum in population
                        else None,
                    }
                )
            totals.append(
                {
                    "mode": mode,
                    "tm_threshold": threshold,
                    "selected_additions": len(selected),
                    "clusters_with_hit": len(selected_counts),
                    "sample_clusters": len(clusters),
                    "frame_projected_additions": sum(
                        population[s] * sum(counts) / len(counts)
                        for s, counts in bins.items()
                    )
                    if population
                    else None,
                }
            )
            identities.extend(
                {
                    "mode": mode,
                    "tm_threshold": threshold,
                    "struct_cluster_id": r["struct_cluster_id"],
                    "entry_id": r["entry_id"],
                    "selected_order": r["selected_order"],
                }
                for r in selected
            )
    write_csv(args.output / "yield_by_stratum.csv", summaries)
    write_csv(args.output / "yield_totals.csv", totals)
    if identities:
        write_csv(args.output / "selected_identities.csv", identities)
    record = {
        "inputs": {
            name: hashlib.sha256((args.audit / name).read_bytes()).hexdigest()
            for name in ["sample.csv", "pairs.csv", "clusters.csv"]
        },
        "population": str(args.population) if args.population else None,
        "population_sha256": hashlib.sha256(args.population.read_bytes()).hexdigest()
        if args.population
        else None,
        "status": "descriptive sensitivity; pre-domain-review and pre-final-decontamination",
        "limits": "No uncertainty interval or production-size claim. Candidate reservoirs are capped at 32; selected additions at three. Core masks are independent and do not establish domain novelty.",
    }
    (args.output / "yield.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
