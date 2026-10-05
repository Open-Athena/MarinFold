"""Sample omitted AFDB members against the actual decontaminated training set.

This is a curation sample, not a training manifest. New candidate sequences must
still undergo the full exp225 evaluation-exclusion search before training.
Reads metadata only; never reads structures during population selection.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import duckdb


def sql_string(value: str) -> str:
    """Quote a trusted filesystem path as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def sample(
    metadata: str,
    training_index: str,
    droplist: str,
    output: Path,
    clusters_per_bin: int,
    candidates_per_cluster: int,
    seed: int,
) -> dict:
    """Write a stratified sample with all current training anchors per cluster."""
    if clusters_per_bin < 1 or candidates_per_cluster < 1:
        raise ValueError("Cluster and candidate counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    started = perf_counter()
    con.execute(
        f"CREATE TABLE metadata AS SELECT * FROM read_parquet({sql_string(metadata)})"
    )
    con.execute(
        f"CREATE TABLE excluded AS SELECT DISTINCT entry_id FROM read_parquet({sql_string(droplist)}) WHERE arm='afdb'"
    )
    total, distinct = con.execute(
        "SELECT count(*),count(DISTINCT entry_id) FROM metadata"
    ).fetchone()
    if total != distinct:
        raise ValueError(f"Duplicate metadata entries: {total} rows / {distinct} ids")
    con.execute(f"""CREATE TABLE anchors AS
        SELECT i.entry_id, i.struct_cluster_id FROM read_parquet({sql_string(training_index)}) i
        ANTI JOIN excluded d
        USING (entry_id) WHERE i.split='train'""")
    missing = con.execute(
        "SELECT count(*) FROM anchors ANTI JOIN metadata USING(entry_id)"
    ).fetchone()[0]
    if missing:
        raise ValueError(f"{missing} training anchors missing from metadata")
    mixed = con.execute("""SELECT count(*) FROM
        (SELECT struct_cluster_id FROM metadata GROUP BY 1 HAVING count(DISTINCT split)>1)""").fetchone()[
        0
    ]
    if mixed:
        raise ValueError(
            f"{mixed} clusters cross original train/validation/test splits"
        )
    con.execute("""CREATE TABLE annotated AS SELECT m.*, a.entry_id IS NOT NULL AS is_anchor,
        e.entry_id IS NOT NULL AS was_excluded
        FROM metadata m LEFT JOIN anchors a USING(entry_id)
        LEFT JOIN excluded e USING(entry_id) WHERE m.split='train'""")
    con.execute("""CREATE TABLE populations AS
        WITH counts AS (
            SELECT struct_cluster_id, count(*) AS cluster_size,
                count(*) FILTER (WHERE is_anchor) AS n_anchors,
                count(*) FILTER (WHERE NOT is_anchor AND NOT was_excluded AND global_plddt >= 80 AND seq_len BETWEEN 60 AND 1000)
                    AS n_candidates,
                median(seq_len) FILTER (WHERE is_anchor) AS anchor_length
            FROM annotated GROUP BY 1
        ) SELECT *,
            CASE WHEN anchor_length<250 THEN '60-249' WHEN anchor_length<500 THEN '250-499'
                ELSE '500-1000' END AS length_bin,
            CASE WHEN cluster_size<20 THEN '<20' WHEN cluster_size<100 THEN '20-99'
                ELSE '100+' END AS size_bin
        FROM counts WHERE n_anchors>0 AND n_candidates>0 AND anchor_length BETWEEN 60 AND 1000
    """)
    con.execute(f"""CREATE TABLE chosen AS SELECT * EXCLUDE(sample_rank) FROM
        (SELECT *,row_number() OVER (PARTITION BY length_bin,size_bin
            ORDER BY md5(struct_cluster_id || ':{seed}')) AS sample_rank FROM populations)
        WHERE sample_rank<={clusters_per_bin}""")
    con.execute(f"""CREATE TABLE selected AS
        SELECT * EXCLUDE(candidate_rank) FROM (
            SELECT m.*,p.cluster_size,p.n_anchors,p.n_candidates,p.length_bin,p.size_bin,
                row_number() OVER (PARTITION BY m.struct_cluster_id,m.is_anchor
                    ORDER BY md5(m.entry_id || ':{seed}')) AS candidate_rank
            FROM annotated m JOIN chosen p USING(struct_cluster_id)
            WHERE m.is_anchor OR (NOT m.was_excluded AND m.global_plddt>=80 AND m.seq_len BETWEEN 60 AND 1000)
        ) WHERE is_anchor OR candidate_rank<={candidates_per_cluster}""")
    for table, name in [("selected", "sample"), ("chosen", "clusters")]:
        con.execute(
            f"COPY (SELECT * FROM {table} ORDER BY struct_cluster_id,entry_id) TO {sql_string(str(output / (name + '.csv')))} (HEADER)"
            if table == "selected"
            else f"COPY (SELECT * FROM {table} ORDER BY struct_cluster_id) TO {sql_string(str(output / (name + '.csv')))} (HEADER)"
        )
    con.execute(f"""COPY (SELECT length_bin,size_bin,count(*) AS eligible_clusters,
        sum(n_candidates) AS eligible_candidates,sum(n_anchors) AS training_anchors
        FROM populations GROUP BY 1,2 ORDER BY 1,2) TO
        {sql_string(str(output / "population.csv"))} (HEADER)""")
    stats = {
        "metadata_rows": total,
        "training_anchor_rows": con.execute("SELECT count(*) FROM anchors").fetchone()[
            0
        ],
        "sample_clusters": con.execute("SELECT count(*) FROM chosen").fetchone()[0],
        "sample_anchors": con.execute(
            "SELECT count(*) FROM selected WHERE is_anchor"
        ).fetchone()[0],
        "sample_candidates": con.execute(
            "SELECT count(*) FROM selected WHERE NOT is_anchor"
        ).fetchone()[0],
        "seed": seed,
        "clusters_per_bin": clusters_per_bin,
        "candidates_per_cluster": candidates_per_cluster,
        "elapsed_seconds": perf_counter() - started,
        "metadata": metadata,
        "training_index": training_index,
        "droplist": droplist,
        "sample_sha256": hashlib.sha256(
            (output / "sample.csv").read_bytes()
        ).hexdigest(),
        "candidate_policy": "pLDDT >=80, length 60-1000, omitted from current AFDB train, absent from known droplist",
        "status": "visual curation only; candidate sequence decontamination pending",
    }
    (output / "sampling.json").write_text(json.dumps(stats, indent=2) + "\n")
    con.close()
    return stats


def main() -> None:
    """Run the local metadata-only sampling pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--training-index", required=True)
    parser.add_argument("--droplist", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters-per-bin", type=int, default=4)
    parser.add_argument("--candidates-per-cluster", type=int, default=8)
    parser.add_argument("--seed", type=int, default=292)
    args = parser.parse_args()
    print(
        json.dumps(
            sample(
                args.metadata,
                args.training_index,
                args.droplist,
                args.output,
                args.clusters_per_bin,
                args.candidates_per_cluster,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
