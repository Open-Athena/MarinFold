# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduce one permissive cross-source MMseqs search into threshold rows."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

COLUMNS = [
    "query",
    "target",
    "identity",
    "query_coverage",
    "target_coverage",
    "evalue",
    "bits",
    "alignment_length",
    "query_length",
    "target_length",
]
THRESHOLDS = (1.0, 0.95, 0.90, 0.70, 0.50, 0.40, 0.30)
AFDB_POPULATION = 3_963_003


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson 95% interval for the sampled AFDB-query hit fraction."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    fraction = successes / trials
    scale = 1 + z**2 / trials
    centre = (fraction + z**2 / (2 * trials)) / scale
    radius = z * math.sqrt(fraction * (1 - fraction) / trials + z**2 / (4 * trials**2)) / scale
    return centre - radius, centre + radius


def summarize(frame: pd.DataFrame, query_count: int) -> pd.DataFrame:
    """One row per identity threshold, retaining source-priority ambiguity."""
    rows = []
    for threshold in THRESHOLDS:
        selected = frame.loc[frame.identity >= threshold]
        query_hits = selected.groupby("query").size()
        target_hits = selected.groupby("target").size()
        queries = int(len(query_hits))
        lower, upper = wilson_interval(queries, query_count)
        rows.append(
            {
                "min_sequence_identity": threshold,
                "min_query_coverage": 0.8,
                "min_target_coverage": 0.8,
                "sampled_afdb_queries": query_count,
                "alignment_edges": int(len(selected)),
                "afdb_queries_with_hit": queries,
                "afdb_query_hit_fraction": queries / query_count,
                "afdb_query_hit_fraction_ci95_low": lower,
                "afdb_query_hit_fraction_ci95_high": upper,
                "projected_afdb_documents_removable_if_esm_preferred": round(
                    queries / query_count * AFDB_POPULATION
                ),
                "projected_afdb_documents_ci95_low": round(lower * AFDB_POPULATION),
                "projected_afdb_documents_ci95_high": round(upper * AFDB_POPULATION),
                "observed_unique_esm_targets": int(len(target_hits)),
                "observed_esm_targets_with_multiple_sampled_queries": int((target_hits > 1).sum()),
                "max_targets_per_sampled_afdb_query": int(query_hits.max()) if queries else 0,
                "median_targets_per_hit_afdb_query": float(query_hits.median()) if queries else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/exp336_dedup/cross_source_30id_80cov_pilot10000-seed336.tsv"),
    )
    parser.add_argument("--query-count", type=int, default=10_000)
    parser.add_argument("--max-seqs", type=int, default=1_000)
    parser.add_argument("--prefilter-median", type=int, default=1_000)
    parser.add_argument("--prefilter-overflows", type=int, default=1_202)
    parser.add_argument("--output", type=Path, default=Path("data/cross_source_sequence_pilot.csv"))
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/cross_source_sequence_pilot.provenance.json"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep="\t", names=COLUMNS)
    if len(frame) == 0:
        raise ValueError(f"no alignments in {args.input}")
    below_floor = (
        frame.identity.min() < 0.30
        or frame.query_coverage.min() < 0.80
        or frame.target_coverage.min() < 0.80
    )
    if below_floor:
        raise ValueError("input contains a pair below the search floor")
    result = summarize(frame, args.query_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    saturated = args.prefilter_median >= args.max_seqs
    args.provenance.write_text(
        json.dumps(
            {
                "status": "pilot_complete_lower_bound",
                "input": str(args.input),
                "input_sha256": digest,
                "alignment_edges": len(frame),
                "sampled_afdb_queries": args.query_count,
                "sample_seed": 336,
                "afdb_population": AFDB_POPULATION,
                "mmseqs_version": "4fe20e6bb0aa219a35f6190322a91ab19cb6898d",
                "search_sensitivity": 7.5,
                "max_seqs": args.max_seqs,
                "prefilter_median_result_list_length": args.prefilter_median,
                "prefilter_overflows": args.prefilter_overflows,
                "prefilter_saturated": saturated,
                "interpretation": (
                    "Because the prefilter cap saturated, counts are lower bounds. The AFDB "
                    "query fraction is projected only to illustrate scale and is not a final "
                    "estimate. "
                    "Unique ESM targets are not extrapolated because target degree is unknown."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
