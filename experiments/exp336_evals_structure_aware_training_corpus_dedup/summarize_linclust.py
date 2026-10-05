# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize full-corpus Linclust sequence stars by source and size."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

EXPECTED_DOCUMENTS = 69_516_181


def source(column: str) -> str:
    """DuckDB expression mapping #213 headers to a source arm."""
    return f"CASE WHEN starts_with({column}, 'afdb|') THEN 'afdb' ELSE 'esm_atlas' END"


def summarize(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Scan the membership TSV once into a temp table and aggregate it."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=64")
        con.execute("SET memory_limit='200GB'")
        con.execute(
            f"""
            CREATE TEMP TABLE memberships AS
            SELECT
                column0 AS representative,
                column1 AS member,
                {source('column0')} AS representative_source,
                {source('column1')} AS member_source
            FROM read_csv(
                '{str(path).replace("'", "''")}',
                delim='\t', header=false,
                columns={{'column0': 'VARCHAR', 'column1': 'VARCHAR'}}
            )
            """
        )
        documents, clusters = con.execute(
            "SELECT count(*), count(DISTINCT representative) FROM memberships"
        ).fetchone()
        if documents != EXPECTED_DOCUMENTS:
            raise ValueError(f"membership rows {documents:,} != expected {EXPECTED_DOCUMENTS:,}")
        relationships = con.execute(
            """
            SELECT
                representative_source,
                member_source,
                count(*) FILTER (WHERE representative = member) AS self_rows,
                count(*) FILTER (WHERE representative != member) AS candidate_removals
            FROM memberships
            GROUP BY ALL
            ORDER BY ALL
            """
        ).fetchdf()
        con.execute(
            """
            CREATE TEMP TABLE cluster_stats AS
            SELECT
                representative,
                representative_source,
                count(*) AS cluster_size,
                count(*) FILTER (WHERE member_source = 'afdb') AS afdb_rows,
                count(*) FILTER (WHERE member_source = 'esm_atlas') AS esm_atlas_rows
            FROM memberships
            GROUP BY representative, representative_source
            """
        )
        size_bins = con.execute(
            """
            SELECT
                CASE
                    WHEN cluster_size = 1 THEN '1'
                    WHEN cluster_size = 2 THEN '2'
                    WHEN cluster_size = 3 THEN '3'
                    WHEN cluster_size = 4 THEN '4'
                    WHEN cluster_size = 5 THEN '5'
                    WHEN cluster_size <= 10 THEN '6-10'
                    WHEN cluster_size <= 100 THEN '11-100'
                    ELSE '>100'
                END AS size_bin,
                count(*) AS clusters,
                sum(cluster_size) AS documents,
                count(*) FILTER (WHERE afdb_rows > 0 AND esm_atlas_rows > 0) AS mixed_clusters
            FROM cluster_stats
            GROUP BY size_bin
            ORDER BY min(cluster_size)
            """
        ).fetchdf()
        nonsingleton, mixed, max_size = con.execute(
            """
            SELECT
                count(*) FILTER (WHERE cluster_size > 1),
                count(*) FILTER (WHERE afdb_rows > 0 AND esm_atlas_rows > 0),
                max(cluster_size)
            FROM cluster_stats
            """
        ).fetchone()
    finally:
        con.close()
    summary = {
        "documents": int(documents),
        "sequence_stars": int(clusters),
        "candidate_removals": int(documents - clusters),
        "candidate_removal_fraction": (documents - clusters) / documents,
        "nonsingleton_stars": int(nonsingleton),
        "mixed_source_stars": int(mixed),
        "max_star_size": int(max_size),
    }
    return relationships, size_bins, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/exp336_dedup/current_linclust_id050_cov080_e1000p0.tsv"),
    )
    parser.add_argument(
        "--relationships", type=Path, default=Path("data/linclust_50_relationships.csv")
    )
    parser.add_argument("--sizes", type=Path, default=Path("data/linclust_50_star_sizes.csv"))
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/linclust_50.provenance.json")
    )
    args = parser.parse_args()
    relationships, sizes, summary = summarize(args.input)
    args.relationships.parent.mkdir(parents=True, exist_ok=True)
    relationships.to_csv(args.relationships, index=False)
    sizes.to_csv(args.sizes, index=False)
    args.provenance.write_text(
        json.dumps(
            {
                "status": "complete_candidate_generation_not_final_dedup",
                "input": str(args.input),
                "input_bytes": args.input.stat().st_size,
                "min_sequence_identity": 0.5,
                "min_bidirectional_coverage": 0.8,
                "evalue_ceiling": 1000.0,
                **summary,
                "warning": (
                    "Linclust stars are candidate organization. Final removal requires direct "
                    "representative/member sequence realignment plus structure/contact evidence."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(relationships.to_string(index=False))


if __name__ == "__main__":
    main()
