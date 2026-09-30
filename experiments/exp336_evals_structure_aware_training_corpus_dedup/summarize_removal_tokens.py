# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Join sequence-only removals to available #232 parquet token ledgers."""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def sql_paths(paths: list[Path]) -> str:
    """Render explicit parquet paths as a DuckDB list literal."""
    if not paths:
        raise ValueError("at least one parquet path is required")
    return "[" + ",".join(f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in paths) + "]"


def summarize_source(
    removals_path: Path,
    corpus_paths: list[Path],
    *,
    source: str,
    expected_shards: int,
) -> dict:
    """Sum source tokens for removals whose source shards are locally available."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=64")
        con.execute("SET memory_limit='200GB'")
        con.execute(
            f"""
            CREATE TEMP VIEW corpus AS
            SELECT
                regexp_extract(filename, '(?:contacts_v1|shard)-(\\d+)-of-', 1)::INTEGER
                    AS shard,
                entry_id,
                num_tokens + 1 AS source_tokens
            FROM read_parquet({sql_paths(corpus_paths)}, filename=true)
            """
        )
        available_shards = con.execute("SELECT count(DISTINCT shard) FROM corpus").fetchone()[0]
        total_candidates = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE removed_source = ?",
            [str(removals_path), source],
        ).fetchone()[0]
        candidate_rows, matched_rows, removed_tokens = con.execute(
            """
            WITH candidates AS (
                SELECT removed_shard AS shard, removed_entry_id AS entry_id
                FROM read_parquet(?)
                WHERE removed_source = ?
                  AND removed_shard IN (SELECT DISTINCT shard FROM corpus)
            )
            SELECT
                count(*),
                count(c.entry_id),
                coalesce(sum(c.source_tokens), 0)
            FROM candidates r
            LEFT JOIN corpus c USING (shard, entry_id)
            """,
            [str(removals_path), source],
        ).fetchone()
    finally:
        con.close()
    if matched_rows != candidate_rows:
        raise ValueError(
            f"{source}: matched {matched_rows:,}/{candidate_rows:,} removals in available shards"
        )
    return {
        "source": source,
        "available_shards": int(available_shards),
        "expected_shards": expected_shards,
        "complete_source_coverage": available_shards == expected_shards,
        "total_candidate_removals": int(total_candidates),
        "candidate_removals_in_available_shards": int(candidate_rows),
        "matched_removals": int(matched_rows),
        "source_tokens_removed_in_available_shards": int(removed_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument(
        "--afdb",
        type=Path,
        default=Path("/data/exp225_decontam/contacts_v1_decontam"),
    )
    parser.add_argument(
        "--esm",
        type=Path,
        default=Path("/data/exp225_decontam/contacts_v1_esm_atlas_decontam"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/sequence_50_token_accounting.csv")
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/sequence_50_token_accounting.provenance.json"),
    )
    args = parser.parse_args()
    removals = (
        args.work / "current_linclust_id050_cov080_e1000p0_selected_removals.parquet"
    )
    records = [
        summarize_source(
            removals,
            sorted(args.afdb.glob("*.parquet")),
            source="afdb",
            expected_shards=2_067,
        ),
        summarize_source(
            removals,
            sorted(args.esm.glob("*.parquet")),
            source="esm_atlas",
            expected_shards=3_338,
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    args.provenance.write_text(
        json.dumps(
            {
                "status": "partial_until_all_esm_shards_are_scanned",
                "removals": str(removals),
                "source_token_definition": "num_tokens + one EOS token",
                "records": records,
                "warning": (
                    "AFDB is exact when all 2,067 shards are present. The local ESM directory "
                    "is incomplete and is not extrapolated; run source-local over all 3,338 "
                    "published shards for the exact pooled token result."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
