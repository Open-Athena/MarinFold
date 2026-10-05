# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve numeric direct-witness removals to stable corpus row identifiers."""

import argparse
import json
import time
from pathlib import Path

import duckdb

DOCUMENTS = 69_516_181


def row_fields(header: str) -> tuple[str, int, int, str]:
    """Parse a #213 stable header as source, shard, original row and entry ID."""
    try:
        source, remainder = header.split("|", maxsplit=1)
        shard, row, entry_id = remainder.split("_", maxsplit=2)
    except ValueError as error:
        raise ValueError(f"malformed corpus row ID: {header!r}") from error
    return source, int(shard), int(row), entry_id


def materialize(removals: Path, lookup: Path, output: Path) -> dict:
    """Join numeric removals to the MMseqs lookup and write compressed parquet."""
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET threads=64")
        con.execute("SET memory_limit='200GB'")
        started = time.perf_counter()
        con.execute(
            f"""
            COPY (
                WITH removals AS (
                    SELECT
                        column0::UBIGINT AS removed_key,
                        column1::UBIGINT AS witness_key,
                        column2::UBIGINT AS sequence_star_key
                    FROM read_csv(
                        '{str(removals).replace("'", "''")}',
                        delim='\t', header=false,
                        columns={{
                            'column0': 'UBIGINT', 'column1': 'UBIGINT',
                            'column2': 'UBIGINT'
                        }}
                    )
                ),
                lookup AS (
                    SELECT column0::UBIGINT AS key, column1 AS row_id
                    FROM read_csv(
                        '{str(lookup).replace("'", "''")}',
                        delim='\t', header=false,
                        columns={{
                            'column0': 'UBIGINT', 'column1': 'VARCHAR',
                            'column2': 'UTINYINT'
                        }}
                    )
                )
                SELECT
                    r.removed_key,
                    removed.row_id AS removed_id,
                    split_part(removed.row_id, '|', 1) AS removed_source,
                    split_part(split_part(removed.row_id, '|', 2), '_', 1)::USMALLINT
                        AS removed_shard,
                    split_part(split_part(removed.row_id, '|', 2), '_', 2)::UINTEGER
                        AS removed_original_row,
                    regexp_extract(removed.row_id, '^[^|]+\\|[^_]+_[^_]+_(.*)$', 1)
                        AS removed_entry_id,
                    r.witness_key,
                    witness.row_id AS witness_id,
                    split_part(witness.row_id, '|', 1) AS witness_source,
                    split_part(split_part(witness.row_id, '|', 2), '_', 1)::USMALLINT
                        AS witness_shard,
                    split_part(split_part(witness.row_id, '|', 2), '_', 2)::UINTEGER
                        AS witness_original_row,
                    regexp_extract(witness.row_id, '^[^|]+\\|[^_]+_[^_]+_(.*)$', 1)
                        AS witness_entry_id,
                    r.sequence_star_key
                FROM removals r
                JOIN lookup removed ON r.removed_key = removed.key
                JOIN lookup witness ON r.witness_key = witness.key
            ) TO '{str(output).replace("'", "''")}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        elapsed = time.perf_counter() - started
        rows, distinct_removed, malformed = con.execute(
            f"""
            SELECT
                count(*), count(DISTINCT removed_key),
                count(*) FILTER (
                    WHERE removed_entry_id = '' OR witness_entry_id = ''
                )
            FROM read_parquet('{str(output).replace("'", "''")}')
            """
        ).fetchone()
    finally:
        con.close()
    if rows != distinct_removed:
        raise ValueError(f"removal manifest repeats {rows - distinct_removed:,} removed rows")
    if malformed:
        raise ValueError(f"removal manifest has {malformed:,} malformed stable row IDs")
    return {
        "rows": int(rows),
        "distinct_removed_rows": int(distinct_removed),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    args = parser.parse_args()
    tag = "id050_cov080_e1000p0"
    record = materialize(
        args.work / f"current_linclust_{tag}_selected_removals_numeric.tsv",
        args.work / "current_esm_then_afdb_db.lookup",
        args.work / f"current_linclust_{tag}_selected_removals.parquet",
    )
    record.update(
        {
            "status": "complete_sequence_only_removal_manifest",
            "min_sequence_identity": 0.5,
            "min_bidirectional_coverage": 0.8,
            "warning": "Structure/contact evidence has not yet been applied.",
        }
    )
    (args.work / f"current_linclust_{tag}_selected_removals.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
