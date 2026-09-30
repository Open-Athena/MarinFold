# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Price AFDB redundancy from existing cluster lineage before pairwise work.

This is deliberately a *pilot*, not the threshold answer. AFDB50 and Foldseek
cluster IDs are transitive assignments, not guarantees that every two members
clear 50% identity or a particular TM-score. The pilot does three useful things
cheaply:

1. proves that the local/published #232 AFDB corpus reproduces its exact
   document and source-token census;
2. measures the scale of repeated sequence and structure lineage among the
   selected rows; and
3. supplies an upper-priority representative for later direct pairwise audits.

``source_tokens`` follows #232's cache accounting: the document's ``num_tokens``
plus one EOS token per document.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

EXPECTED_AFDB_DOCUMENTS = 3_963_003
EXPECTED_AFDB_SOURCE_TOKENS = 4_432_940_838


def sql_string(value: str | Path) -> str:
    """Quote a trusted path as a DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def materialize(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> None:
    """Project the complete AFDB corpus to a compact in-process table."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE corpus AS
        SELECT
            entry_id,
            coalesce(seq_cluster_id, entry_id) AS seq_cluster_id,
            coalesce(struct_cluster_id, entry_id) AS struct_cluster_id,
            global_plddt,
            num_tokens + 1 AS source_tokens
        FROM read_parquet({sql_string(parquet_glob)})
        """
    )
    duplicate_ids = con.execute(
        "SELECT count(*) - count(DISTINCT entry_id) FROM corpus"
    ).fetchone()[0]
    if duplicate_ids:
        raise ValueError(f"AFDB corpus contains {duplicate_ids:,} duplicate entry IDs")


def representative_stats(
    con: duckdb.DuckDBPyConnection, rule: str, partition_columns: tuple[str, ...]
) -> dict[str, int | float | str]:
    """Select the highest-confidence stable representative per lineage group."""
    if partition_columns:
        partition = ", ".join(partition_columns)
        source = f"""
            SELECT * EXCLUDE (representative_rank)
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY {partition}
                    ORDER BY global_plddt DESC, entry_id ASC
                ) AS representative_rank
                FROM corpus
            )
            WHERE representative_rank = 1
        """
    else:
        source = "SELECT * FROM corpus"
    documents_after, tokens_after = con.execute(
        f"SELECT count(*), sum(source_tokens) FROM ({source})"
    ).fetchone()
    documents_before, tokens_before = con.execute(
        "SELECT count(*), sum(source_tokens) FROM corpus"
    ).fetchone()
    return {
        "scope": "current_exp232_afdb",
        "rule": rule,
        "documents_before": int(documents_before),
        "documents_after": int(documents_after),
        "documents_removed": int(documents_before - documents_after),
        "document_loss_fraction": (documents_before - documents_after) / documents_before,
        "source_tokens_before": int(tokens_before),
        "source_tokens_after": int(tokens_after),
        "source_tokens_removed": int(tokens_before - tokens_after),
        "source_token_loss_fraction": (tokens_before - tokens_after) / tokens_before,
    }


def cluster_stats(
    con: duckdb.DuckDBPyConnection, label: str, columns: tuple[str, ...]
) -> dict[str, int | float | str]:
    """Distribution of lineage-group sizes for interpreting the proxy."""
    group = ", ".join(columns)
    groups, singleton_groups, max_size, mean_size = con.execute(
        f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE n = 1),
            max(n),
            avg(n)
        FROM (SELECT count(*) AS n FROM corpus GROUP BY {group})
        """
    ).fetchone()
    return {
        "lineage": label,
        "groups": int(groups),
        "singleton_groups": int(singleton_groups),
        "max_group_size": int(max_size),
        "mean_group_size": float(mean_size),
    }


def run(parquet_glob: str, output: Path, provenance: Path) -> pd.DataFrame:
    """Run the census and write small, reviewable outputs."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=16")
        con.execute("SET preserve_insertion_order=false")
        materialize(con, parquet_glob)
        documents, source_tokens = con.execute(
            "SELECT count(*), sum(source_tokens) FROM corpus"
        ).fetchone()
        if documents != EXPECTED_AFDB_DOCUMENTS or source_tokens != EXPECTED_AFDB_SOURCE_TOKENS:
            raise ValueError(
                "AFDB census does not match #232: "
                f"got {documents:,} documents / {source_tokens:,} tokens, expected "
                f"{EXPECTED_AFDB_DOCUMENTS:,} / {EXPECTED_AFDB_SOURCE_TOKENS:,}"
            )
        rows = [
            representative_stats(con, "none", ()),
            representative_stats(con, "one_per_afdb50_sequence_lineage", ("seq_cluster_id",)),
            representative_stats(
                con,
                "one_per_sequence_and_foldseek_lineage_intersection",
                ("seq_cluster_id", "struct_cluster_id"),
            ),
            representative_stats(
                con, "one_per_foldseek_structure_lineage", ("struct_cluster_id",)
            ),
        ]
        cluster_rows = [
            cluster_stats(con, "afdb50_sequence", ("seq_cluster_id",)),
            cluster_stats(
                con,
                "sequence_and_foldseek_intersection",
                ("seq_cluster_id", "struct_cluster_id"),
            ),
            cluster_stats(con, "foldseek_structure", ("struct_cluster_id",)),
        ]
    finally:
        con.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    provenance.write_text(
        json.dumps(
            {
                "status": "complete",
                "input": parquet_glob,
                "expected_documents": EXPECTED_AFDB_DOCUMENTS,
                "expected_source_tokens": EXPECTED_AFDB_SOURCE_TOKENS,
                "source_token_definition": "num_tokens + one EOS token per document",
                "representative_order": "global_plddt DESC, entry_id ASC",
                "cluster_distributions": cluster_rows,
                "warning": (
                    "Lineage groups are transitive source metadata, not direct pairwise "
                    "identity/TM witnesses. These rows are pilot proxies, not threshold results."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--afdb",
        default="/data/exp225_decontam/contacts_v1_decontam/*.parquet",
        help="glob for the complete #232 AFDB parquet corpus",
    )
    parser.add_argument("--output", type=Path, default=Path("data/afdb_lineage_pilot.csv"))
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/afdb_lineage_pilot.provenance.json")
    )
    args = parser.parse_args()
    frame = run(args.afdb, args.output, args.provenance)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
