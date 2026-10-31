"""Audit whether both candidate chains co-occur as homologs in one training example."""

import argparse
import csv
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
ALIGNMENT_COLUMNS = [
    "query",
    "train_target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
    "nident",
    "qlen",
    "tlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
]


def alignment_read_sql() -> str:
    """Return a parameterized DuckDB TSV reader with explicit column types."""
    columns = ", ".join(
        f"'{name}': '{'VARCHAR' if name in {'query', 'train_target'} else 'DOUBLE'}'"
        for name in ALIGNMENT_COLUMNS
    )
    return f"read_csv(?, delim='\\t', header=false, columns={{{columns}}})"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a deterministic CSV with Unix line endings."""
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def records(result: duckdb.DuckDBPyRelation) -> list[dict]:
    """Convert a DuckDB relation into dictionaries without a pandas dependency."""
    columns = [column[0] for column in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def main() -> None:
    """Write pair-level survival, per-complex statuses and alignment witnesses."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--work", type=Path, default=Path("/data/exp350"))
    ap.add_argument(
        "--complex-alignments",
        type=Path,
        nargs="+",
        help="Complex-arm TSVs to union; defaults to the two broad-search outputs.",
    )
    ap.add_argument(
        "--database", type=Path, default=Path("/data/exp350/pair_analysis.duckdb")
    )
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()
    con = duckdb.connect(str(args.database))
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA memory_limit='100GB'")
    con.execute(
        "CREATE OR REPLACE TABLE membership AS "
        "SELECT source, target, CAST(chain_index AS INTEGER) chain_index, query "
        "FROM read_csv_auto(?, header=true)",
        [str(args.data / "query_membership.csv")],
    )
    con.execute(
        "CREATE OR REPLACE TABLE candidates AS "
        "SELECT * FROM read_csv_auto(?, header=true)",
        [str(args.data / "candidates.csv")],
    )
    con.execute(
        "CREATE OR REPLACE TABLE eligible AS "
        "SELECT source, target, complex_type FROM candidates "
        "WHERE eligibility='candidate'"
    )

    reader = alignment_read_sql()
    complex_paths = args.complex_alignments or [
        args.work / "complex_query_alignments.tsv",
        args.work / "complex_target_alignments.tsv",
    ]
    complex_union = " UNION ALL ".join(
        f"SELECT * FROM {reader}" for _ in complex_paths
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE complex_hits AS
        SELECT DISTINCT query, train_target,
          CAST(nident AS BIGINT) nident, CAST(alnlen AS BIGINT) alnlen,
          CAST(qlen AS BIGINT) qlen, CAST(tlen AS BIGINT) tlen,
          CAST(qstart AS BIGINT) qstart, CAST(qend AS BIGINT) qend,
          CAST(tstart AS BIGINT) tstart, CAST(tend AS BIGINT) tend,
          CAST(bits AS DOUBLE) bits
        FROM ({complex_union})
        WHERE CAST(nident AS BIGINT) * 10 >= CAST(alnlen AS BIGINT) * 3
          AND (
            2 * (CAST(qend AS BIGINT) - CAST(qstart AS BIGINT) + 1)
              >= CAST(qlen AS BIGINT)
            OR 2 * (CAST(tend AS BIGINT) - CAST(tstart AS BIGINT) + 1)
              >= CAST(tlen AS BIGINT)
          )""",
        [str(path) for path in complex_paths],
    )
    con.execute(
        """CREATE OR REPLACE TABLE complex_edges AS
        SELECT DISTINCT m.source, m.target, m.chain_index candidate_chain,
          split_part(h.train_target, '|', 1) train_arm,
          split_part(h.train_target, '|', 2) train_shard,
          split_part(h.train_target, '|', 3) train_row,
          CAST(split_part(h.train_target, '|', 4) AS INTEGER) train_chain,
          split_part(h.train_target, '|', 5) = 'homodimer' train_homodimer,
          h.query, h.train_target, h.nident, h.alnlen, h.bits
        FROM complex_hits h
        JOIN membership m USING(query)
        JOIN eligible e ON e.source=m.source AND e.target=m.target"""
    )
    # exp294's corpus contains dimers. build_complex_sequences.py deduplicates
    # identical sequences within a document, so an AFCDB homodimer has one
    # stored target sequence but two chain instances available for matching.
    con.execute(
        """CREATE OR REPLACE TABLE complex_pair_hits AS
        SELECT e0.source, e0.target, e0.train_arm, e0.train_shard, e0.train_row,
          e0.query query_0, e0.train_target train_target_0,
          e0.nident nident_0, e0.alnlen alnlen_0, e0.bits bits_0,
          e1.query query_1, e1.train_target train_target_1,
          e1.nident nident_1, e1.alnlen alnlen_1, e1.bits bits_1,
          e0.train_homodimer
        FROM complex_edges e0
        JOIN complex_edges e1
          ON e0.source=e1.source AND e0.target=e1.target
          AND e0.train_arm=e1.train_arm
          AND e0.train_shard=e1.train_shard AND e0.train_row=e1.train_row
        WHERE e0.candidate_chain=0 AND e1.candidate_chain=1
          AND (e0.train_chain <> e1.train_chain OR e0.train_homodimer)"""
    )

    helico_paths = [
        str(args.work / "helico_query_alignments.tsv"),
        str(args.work / "helico_target_alignments.tsv"),
    ]
    con.execute(
        f"""CREATE OR REPLACE TABLE helico_hits AS
        SELECT DISTINCT query, train_target,
          CAST(nident AS BIGINT) nident, CAST(alnlen AS BIGINT) alnlen,
          CAST(qlen AS BIGINT) qlen, CAST(tlen AS BIGINT) tlen,
          CAST(qstart AS BIGINT) qstart, CAST(qend AS BIGINT) qend,
          CAST(tstart AS BIGINT) tstart, CAST(tend AS BIGINT) tend,
          CAST(bits AS DOUBLE) bits
        FROM (SELECT * FROM {reader} UNION ALL SELECT * FROM {reader})
        WHERE CAST(nident AS BIGINT) * 10 >= CAST(alnlen AS BIGINT) * 3
          AND (
            2 * (CAST(qend AS BIGINT) - CAST(qstart AS BIGINT) + 1)
              >= CAST(qlen AS BIGINT)
            OR 2 * (CAST(tend AS BIGINT) - CAST(tstart AS BIGINT) + 1)
              >= CAST(tlen AS BIGINT)
          )""",
        helico_paths,
    )
    con.execute(
        """CREATE OR REPLACE TABLE helico_edges AS
        SELECT DISTINCT m.source, m.target, m.chain_index candidate_chain,
          lower(split_part(split_part(h.train_target, '|', 2), '_', 1)) train_pdb,
          split_part(split_part(h.train_target, '|', 2), '_', 2) train_chain,
          h.query, h.train_target, h.nident, h.alnlen, h.bits
        FROM helico_hits h
        JOIN membership m USING(query)
        JOIN eligible e ON e.source=m.source AND e.target=m.target"""
    )
    con.execute(
        """CREATE OR REPLACE TABLE helico_pair_hits AS
        SELECT e0.source, e0.target, e0.train_pdb,
          e0.query query_0, e0.train_target train_target_0,
          e0.nident nident_0, e0.alnlen alnlen_0, e0.bits bits_0,
          e1.query query_1, e1.train_target train_target_1,
          e1.nident nident_1, e1.alnlen alnlen_1, e1.bits bits_1
        FROM helico_edges e0
        JOIN helico_edges e1
          ON e0.source=e1.source AND e0.target=e1.target
          AND e0.train_pdb=e1.train_pdb
        WHERE e0.candidate_chain=0 AND e1.candidate_chain=1
          AND e0.train_chain <> e1.train_chain"""
    )

    per_complex = records(
        con.sql(
            """WITH complex_status AS (
              SELECT source, target,
                bool_or(train_arm='afcdb') afcdb_pair_hit,
                bool_or(train_arm='pinder') pinder_pair_hit
              FROM complex_pair_hits GROUP BY source, target
            ), helico_status AS (
              SELECT source, target, true helico_pair_hit
              FROM helico_pair_hits GROUP BY source, target
            )
            SELECT c.*,
              CASE WHEN coalesce(s.afcdb_pair_hit, false) THEN 'hit' ELSE 'no_hit' END
                afcdb_pair_status,
              CASE WHEN coalesce(s.pinder_pair_hit, false) THEN 'hit' ELSE 'no_hit' END
                pinder_pair_status,
              CASE WHEN coalesce(s.afcdb_pair_hit, false)
                     OR coalesce(s.pinder_pair_hit, false)
                   THEN 'rejected' ELSE 'pair_clean' END complex_pair_status,
              CASE WHEN coalesce(h.helico_pair_hit, false) THEN 'hit' ELSE 'no_hit' END
                helico_finetune_pair_status
            FROM candidates c
            LEFT JOIN complex_status s USING(source, target)
            LEFT JOIN helico_status h USING(source, target)
            ORDER BY c.source, c.target"""
        )
    )
    write_csv(args.data / "pair_per_complex.csv", per_complex)

    summary = []
    for source in ["foldbench", "pinder"]:
        source_rows = [row for row in per_complex if row["source"] == source]
        for complex_type in ["all", "homodimer", "heterodimer"]:
            cohort = [
                row
                for row in source_rows
                if complex_type == "all" or row.get("complex_type") == complex_type
            ]
            eligible_rows = [row for row in cohort if row["eligibility"] == "candidate"]
            after_afcdb = [
                row for row in eligible_rows if row["afcdb_pair_status"] == "no_hit"
            ]
            after_both = [
                row for row in after_afcdb if row["pinder_pair_status"] == "no_hit"
            ]
            after_helico = [
                row
                for row in after_both
                if row["helico_finetune_pair_status"] == "no_hit"
            ]
            for stage, rows in [
                ("source_candidates", cohort),
                ("quality_and_scope", eligible_rows),
                ("after_afcdb_pair", after_afcdb),
                ("after_pinder_pair", after_both),
                ("after_helico_finetune_pair", after_helico),
            ]:
                summary.append(
                    {
                        "source": source,
                        "complex_type": complex_type,
                        "stage": stage,
                        "remaining": len(rows),
                        "pdb_entries": len({row["pdb_id"] for row in rows}),
                    }
                )
    write_csv(args.data / "pair_survival.csv", summary)

    complex_witnesses = records(
        con.sql(
            """SELECT 'marinfold_complex' screen, source, target, train_arm,
              train_arm || '|' || train_shard || '|' || train_row train_document,
              query_0, train_target_0, nident_0, alnlen_0,
              query_1, train_target_1, nident_1, alnlen_1
            FROM complex_pair_hits
            QUALIFY row_number() OVER (
              PARTITION BY source, target, train_arm
              ORDER BY bits_0 + bits_1 DESC
            ) = 1"""
        )
    )
    helico_witnesses = records(
        con.sql(
            """SELECT 'helico_finetune_pdb' screen, source, target,
              'helico_finetune' train_arm, train_pdb train_document,
              query_0, train_target_0, nident_0, alnlen_0,
              query_1, train_target_1, nident_1, alnlen_1
            FROM helico_pair_hits
            QUALIFY row_number() OVER (
              PARTITION BY source, target
              ORDER BY bits_0 + bits_1 DESC
            ) = 1"""
        )
    )
    write_csv(
        args.data / "pair_exclusion_witnesses.csv",
        complex_witnesses + helico_witnesses,
    )
    for row in summary:
        if row["complex_type"] == "all":
            print(row)


if __name__ == "__main__":
    main()
