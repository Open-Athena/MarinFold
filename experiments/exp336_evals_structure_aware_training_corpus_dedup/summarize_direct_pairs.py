# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize directly verified within-star sequence alignments."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def source(column: str) -> str:
    """DuckDB expression mapping #213 headers to a source arm."""
    return f"CASE WHEN starts_with({column}, 'afdb|') THEN 'afdb' ELSE 'esm_atlas' END"


def summarize(
    path: Path, memberships_path: Path, identity: float, coverage: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate source relationships and verify alignment invariants."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=64")
        con.execute("SET memory_limit='200GB'")
        con.execute(
            f"""
            CREATE TEMP TABLE alignments AS
            SELECT
                column0 AS query,
                column1 AS target,
                least(column0, column1) AS pair_left,
                greatest(column0, column1) AS pair_right,
                column2::DOUBLE AS sequence_identity,
                column12::DOUBLE AS query_coverage,
                column13::DOUBLE AS target_coverage,
                {source('column0')} AS query_source,
                {source('column1')} AS target_source
            FROM read_csv(
                '{str(path).replace("'", "''")}',
                delim='\t', header=false,
                columns={{
                    'column0': 'VARCHAR', 'column1': 'VARCHAR',
                    'column2': 'DOUBLE', 'column3': 'BIGINT',
                    'column4': 'BIGINT', 'column5': 'BIGINT',
                    'column6': 'BIGINT', 'column7': 'BIGINT',
                    'column8': 'BIGINT', 'column9': 'BIGINT',
                    'column10': 'DOUBLE', 'column11': 'DOUBLE',
                    'column12': 'DOUBLE', 'column13': 'DOUBLE'
                }}
            )
            """
        )
        (
            rows,
            distinct_pairs,
            self_pairs,
            minimum_identity,
            minimum_qcov,
            minimum_tcov,
        ) = con.execute(
            """
            SELECT
                count(*),
                count(DISTINCT (query, target)),
                count(*) FILTER (WHERE query = target),
                min(sequence_identity),
                min(query_coverage),
                min(target_coverage)
            FROM alignments
            """
        ).fetchone()
        below_threshold = con.execute(
            """
            SELECT count(*)
            FROM alignments
            WHERE sequence_identity + 1e-12 < ?
               OR query_coverage + 1e-12 < ?
               OR target_coverage + 1e-12 < ?
            """,
            [identity, coverage, coverage],
        ).fetchone()[0]
        relationships = con.execute(
            """
            SELECT
                least(query_source, target_source) AS source_a,
                greatest(query_source, target_source) AS source_b,
                count(*) AS directly_verified_pairs
            FROM alignments
            GROUP BY ALL
            ORDER BY ALL
            """
        ).fetchdf()
        con.execute(
            f"""
            CREATE TEMP TABLE central_candidates AS
            SELECT
                column0 AS representative,
                column1 AS member,
                least(column0, column1) AS pair_left,
                greatest(column0, column1) AS pair_right,
                {source('column0')} AS representative_source,
                {source('column1')} AS member_source
            FROM read_csv(
                '{str(memberships_path).replace("'", "''")}',
                delim='\t', header=false,
                columns={{'column0': 'VARCHAR', 'column1': 'VARCHAR'}}
            )
            WHERE column0 != column1
            """
        )
        central_candidates, verified_central_candidates = con.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE a.query IS NOT NULL)
            FROM central_candidates c
            LEFT JOIN alignments a
              ON c.pair_left = a.pair_left AND c.pair_right = a.pair_right
            """
        ).fetchone()
        central_relationships = con.execute(
            """
            SELECT
                representative_source,
                member_source,
                count(*) AS linclust_candidates,
                count(*) FILTER (WHERE a.query IS NOT NULL) AS directly_verified_candidates
            FROM central_candidates c
            LEFT JOIN alignments a
              ON c.pair_left = a.pair_left AND c.pair_right = a.pair_right
            GROUP BY ALL
            ORDER BY ALL
            """
        ).fetchdf()
    finally:
        con.close()
    if distinct_pairs != rows:
        raise ValueError(f"direct alignment output repeats {rows - distinct_pairs:,} pairs")
    if self_pairs:
        raise ValueError(f"direct alignment output contains {self_pairs:,} self pairs")
    if below_threshold:
        raise ValueError(f"direct alignment output has {below_threshold:,} below-threshold rows")
    summary = {
        "alignment_rows": int(rows),
        "distinct_pairs": int(distinct_pairs),
        "directly_verified_pairs": int(rows),
        "linclust_central_candidates": int(central_candidates),
        "directly_verified_central_candidates": int(verified_central_candidates),
        "failed_direct_central_candidates": int(
            central_candidates - verified_central_candidates
        ),
        "minimum_sequence_identity": float(minimum_identity),
        "minimum_query_coverage": float(minimum_qcov),
        "minimum_target_coverage": float(minimum_tcov),
    }
    return relationships, central_relationships, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/data/exp336_dedup/current_linclust_id050_cov080_e1000p0_direct_align.tsv"
        ),
    )
    parser.add_argument(
        "--memberships",
        type=Path,
        default=Path("/data/exp336_dedup/current_linclust_id050_cov080_e1000p0.tsv"),
    )
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument(
        "--relationships",
        type=Path,
        default=Path("data/linclust_50_direct_relationships.csv"),
    )
    parser.add_argument(
        "--central-relationships",
        type=Path,
        default=Path("data/linclust_50_direct_central_relationships.csv"),
    )
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/linclust_50_direct.provenance.json")
    )
    args = parser.parse_args()
    relationships, central_relationships, summary = summarize(
        args.input, args.memberships, args.identity, args.coverage
    )
    args.relationships.parent.mkdir(parents=True, exist_ok=True)
    relationships.to_csv(args.relationships, index=False)
    central_relationships.to_csv(args.central_relationships, index=False)
    args.provenance.write_text(
        json.dumps(
            {
                "status": "complete_direct_sequence_verification_not_final_dedup",
                "input": str(args.input),
                "input_bytes": args.input.stat().st_size,
                "linclust_memberships": str(args.memberships),
                "linclust_memberships_bytes": args.memberships.stat().st_size,
                "required_sequence_identity": args.identity,
                "required_bidirectional_coverage": args.coverage,
                **summary,
                "warning": (
                    "Directly verified pairs are eligible for removal only after applying "
                    "the structure/contact witness rule."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(relationships.to_string(index=False))
    print(central_relationships.to_string(index=False))


if __name__ == "__main__":
    main()
