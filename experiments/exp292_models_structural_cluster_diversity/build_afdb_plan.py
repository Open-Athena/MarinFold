"""Build the AFDB production cluster plan from metadata only."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import duckdb

from build_esm_plan import sql_string

SEED = 292
RESERVOIR_CAP = 8


def build_plan(
    metadata_glob: str,
    training_index: Path,
    droplist: Path,
    output: Path,
    *,
    threads: int,
    memory_limit: str,
    reservoir_cap: int = RESERVOIR_CAP,
    seed: int = SEED,
) -> dict:
    """Write all current anchors plus a bounded omitted-member reservoir."""
    if reservoir_cap < 3:
        raise ValueError("The AFDB reservoir must hold all three output slots")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        shutil.rmtree(output)
        output.mkdir()
    database = output.parent / "afdb-plan.duckdb"
    temporary = output.parent / "duckdb-tmp"
    temporary.mkdir(exist_ok=True)
    con = duckdb.connect(str(database))
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit={sql_string(memory_limit)}")
    con.execute(f"SET temp_directory={sql_string(temporary)}")
    con.execute("SET preserve_insertion_order=false")
    started = perf_counter()
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE excluded AS
            SELECT DISTINCT entry_id
            FROM read_parquet({sql_string(droplist)})
            WHERE arm = 'afdb'
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE metadata AS
            SELECT *
            FROM read_parquet({sql_string(metadata_glob)})
            WHERE split = 'train' AND struct_cluster_id IS NOT NULL
            """
        )
        metadata_counts = con.execute(
            "SELECT count(*),count(DISTINCT entry_id) FROM metadata"
        ).fetchone()
        if metadata_counts[0] != metadata_counts[1]:
            raise ValueError("AFDB metadata contains duplicate entry IDs")
        mixed_splits = con.execute(
            f"""
            SELECT count(*) FROM (
                SELECT struct_cluster_id
                FROM read_parquet({sql_string(metadata_glob)})
                GROUP BY 1 HAVING count(DISTINCT split) > 1
            )
            """
        ).fetchone()[0]
        if mixed_splits:
            raise ValueError(f"{mixed_splits} AFDB structural clusters cross splits")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE anchors AS
            SELECT m.*
            FROM read_parquet({sql_string(training_index)}) i
            JOIN metadata m USING (entry_id, struct_cluster_id)
            ANTI JOIN excluded e USING (entry_id)
            WHERE i.split = 'train'
            """
        )
        missing_anchors = con.execute(
            f"""
            SELECT count(*)
            FROM read_parquet({sql_string(training_index)}) i
            ANTI JOIN excluded e USING (entry_id)
            ANTI JOIN anchors a USING (entry_id)
            WHERE i.split = 'train'
            """
        ).fetchone()[0]
        if missing_anchors:
            raise ValueError(f"{missing_anchors} current AFDB anchors are absent from metadata")
        current_anchor_count = con.execute("SELECT count(*) FROM anchors").fetchone()[0]
        con.execute(
            """
            CREATE OR REPLACE TABLE candidates AS
            SELECT m.*
            FROM metadata m
            ANTI JOIN anchors a USING (entry_id)
            ANTI JOIN excluded e USING (entry_id)
            WHERE m.global_plddt >= 80 AND m.seq_len BETWEEN 60 AND 1000
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE populations AS
            SELECT
                a.struct_cluster_id,
                count(DISTINCT a.entry_id) AS n_anchors,
                count(DISTINCT c.entry_id) AS n_candidates
            FROM anchors a
            JOIN candidates c USING (struct_cluster_id)
            GROUP BY a.struct_cluster_id
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE candidate_reservoir AS
            SELECT * EXCLUDE (candidate_rank)
            FROM (
                SELECT
                    c.*,
                    row_number() OVER (
                        PARTITION BY c.struct_cluster_id
                        ORDER BY md5('{seed}:' || c.entry_id)
                    ) AS candidate_rank
                FROM candidates c
                JOIN populations p USING (struct_cluster_id)
            )
            WHERE candidate_rank <= {reservoir_cap}
            """
        )
        # A single writer produces one complete parquet per hash partition. This
        # preserves cluster locality and avoids thousands of tiny plan objects.
        con.execute("SET threads=1")
        con.execute("SET partitioned_write_max_open_files=512")
        con.execute(
            f"""
            COPY (
                SELECT
                    substr(md5(a.struct_cluster_id || ':{seed}'), 1, 2) AS shard,
                    a.*,
                    true AS is_anchor,
                    0::INTEGER AS reservoir_rank,
                    p.n_anchors,
                    p.n_candidates
                FROM anchors a JOIN populations p USING (struct_cluster_id)
                UNION ALL
                SELECT
                    substr(md5(c.struct_cluster_id || ':{seed}'), 1, 2) AS shard,
                    c.*,
                    false AS is_anchor,
                    row_number() OVER (
                        PARTITION BY c.struct_cluster_id
                        ORDER BY md5('{seed}:' || c.entry_id)
                    )::INTEGER AS reservoir_rank,
                    p.n_anchors,
                    p.n_candidates
                FROM candidate_reservoir c JOIN populations p USING (struct_cluster_id)
            ) TO {sql_string(output)} (
                FORMAT PARQUET,
                PARTITION_BY (shard),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000,
                OVERWRITE_OR_IGNORE
            )
            """
        )
        glob = str(output / "*" / "*.parquet")
        counts = con.execute(
            f"""
            SELECT
                count(DISTINCT struct_cluster_id) AS clusters,
                count(*) FILTER (WHERE is_anchor) AS anchors,
                count(*) FILTER (WHERE NOT is_anchor) AS reservoir_candidates
            FROM read_parquet({sql_string(glob)}, hive_partitioning=true)
            """
        ).fetchone()
        theoretical_additions = con.execute(
            "SELECT sum(least(3,n_candidates)) FROM populations"
        ).fetchone()[0]
        eligible_candidates = con.execute(
            "SELECT sum(n_candidates) FROM populations"
        ).fetchone()[0]
        shard_count = con.execute(
            f"SELECT count(DISTINCT shard) FROM read_parquet({sql_string(glob)}, hive_partitioning=true)"
        ).fetchone()[0]
    finally:
        con.close()
    files = sorted(output.rglob("*.parquet"))
    return {
        "metadata_train_rows": metadata_counts[0],
        "current_training_anchors": current_anchor_count,
        "planned_anchor_rows": counts[1],
        "eligible_clusters": counts[0],
        "eligible_omitted_members": eligible_candidates,
        "reservoir_candidate_rows": counts[2],
        "theoretical_additions_before_sequence_filter": theoretical_additions,
        "shards": shard_count,
        "parquet_files": len(files),
        "parquet_bytes": sum(path.stat().st_size for path in files),
        "reservoir_cap": reservoir_cap,
        "seed": seed,
        "elapsed_seconds": perf_counter() - started,
        "metadata_glob": metadata_glob,
        "training_index_sha256": hashlib.sha256(training_index.read_bytes()).hexdigest(),
        "droplist_sha256": hashlib.sha256(droplist.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Build and record the AFDB production plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--droplist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()
    stats = build_plan(
        args.metadata,
        args.training_index,
        args.droplist,
        args.output,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    record = {"status": "complete", "source": "afdb_v4_afdb24m", **stats}
    (args.output.parent / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
