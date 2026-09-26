"""Plan the AFDB structural clusters that never entered training at all.

Exp53 built the current corpus with ``min_cluster_size = 3``: a structural
cluster with fewer than three *usable* members (``seq_len`` in ``[2, 2000]``)
was discarded entirely, not merely thinned. That removed 704,560 of the
1,645,588 training-split structural clusters, and because the main exp292 plan
anchors on the current training index, the supplement inherited the exclusion —
no anchor and no candidate in it comes from a dropped cluster.

This planner covers exactly those clusters. They differ from the anchored ones
in a way that simplifies everything downstream: **each holds at most two usable
members**, so the three-slot policy degenerates to "take every member that
passes quality". There is no retained anchor to measure novelty against, no
selection to make between candidates, and therefore no alignment work at all.
Rows are emitted with the same schema and the same cluster-local hash
partitioning as the main plan so the existing validation job runs unchanged.

The exp225 droplist is deliberately not applied here. It lists entries removed
*from the training corpus* for matching a held-out sequence, and these clusters
were never in that corpus. The binding protection is the frozen held-out screen
that curation runs against every candidate, which this arm gets in full.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import duckdb

from build_esm_plan import sql_string

SEED = 292
# Exp53's usable-member window; the cluster-size rule was defined over it.
USABLE_MIN_SEQ_LEN = 2
USABLE_MAX_SEQ_LEN = 2000
EXP53_MIN_CLUSTER_SIZE = 3
# The frozen exp292 candidate-quality bar, identical to the anchored arm.
MIN_PLDDT = 80
MIN_SEQ_LEN = 60
MAX_SEQ_LEN = 1000


def build_plan(
    metadata_glob: str,
    output: Path,
    *,
    threads: int,
    memory_limit: str,
    anchored_plan_glob: str | None = None,
    min_plddt: int = MIN_PLDDT,
    seed: int = SEED,
) -> dict:
    """Write every quality-passing member of a never-trained structural cluster."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        shutil.rmtree(output)
        output.mkdir()
    temporary = output.parent / "duckdb-tmp"
    temporary.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit={sql_string(memory_limit)}")
    con.execute(f"SET temp_directory={sql_string(temporary)}")
    con.execute("SET preserve_insertion_order=false")
    started = perf_counter()
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE metadata AS
            SELECT *
            FROM read_parquet({sql_string(metadata_glob)})
            WHERE split = 'train' AND struct_cluster_id IS NOT NULL
            """
        )
        counts = con.execute(
            "SELECT count(*), count(DISTINCT entry_id) FROM metadata"
        ).fetchone()
        if counts[0] != counts[1]:
            raise ValueError("AFDB metadata contains duplicate entry IDs")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE cluster_sizes AS
            SELECT
                struct_cluster_id,
                count(*) FILTER (
                    WHERE seq_len BETWEEN {USABLE_MIN_SEQ_LEN} AND {USABLE_MAX_SEQ_LEN}
                ) AS usable_members,
                count(*) AS total_members
            FROM metadata
            GROUP BY struct_cluster_id
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE untrained AS
            SELECT * FROM cluster_sizes
            WHERE usable_members < {EXP53_MIN_CLUSTER_SIZE}
            """
        )
        untrained = con.execute(
            "SELECT count(*), sum(total_members), sum(usable_members) FROM untrained"
        ).fetchone()
        con.execute(
            f"""
            CREATE OR REPLACE TABLE members AS
            SELECT m.*
            FROM metadata m JOIN untrained u USING (struct_cluster_id)
            WHERE m.global_plddt >= {min_plddt}
              AND m.seq_len BETWEEN {MIN_SEQ_LEN} AND {MAX_SEQ_LEN}
            """
        )
        # The two arms must be disjoint: a cluster cannot both hold a retained
        # anchor and never have entered training.
        overlap = 0
        if anchored_plan_glob:
            overlap = con.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT DISTINCT struct_cluster_id
                    FROM read_parquet({sql_string(anchored_plan_glob)})
                    INTERSECT
                    SELECT DISTINCT struct_cluster_id FROM members
                )
                """
            ).fetchone()[0]
            if overlap:
                raise ValueError(
                    f"{overlap} clusters appear in both the anchored and untrained plans"
                )
        oversized = con.execute(
            f"""
            SELECT count(*) FROM (
                SELECT struct_cluster_id FROM members
                GROUP BY 1 HAVING count(*) > {EXP53_MIN_CLUSTER_SIZE - 1}
            )
            """
        ).fetchone()[0]
        if oversized:
            raise ValueError(
                f"{oversized} untrained clusters hold more than "
                f"{EXP53_MIN_CLUSTER_SIZE - 1} quality members, which is impossible"
            )
        con.execute("SET threads=1")
        con.execute("SET partitioned_write_max_open_files=512")
        con.execute(
            f"""
            COPY (
                SELECT
                    substr(md5(m.struct_cluster_id || ':{seed}'), 1, 2) AS shard,
                    m.*,
                    false AS is_anchor,
                    row_number() OVER (
                        PARTITION BY m.struct_cluster_id
                        ORDER BY m.global_plddt DESC, m.entry_id ASC
                    )::INTEGER AS reservoir_rank,
                    0::BIGINT AS n_anchors,
                    count(*) OVER (PARTITION BY m.struct_cluster_id) AS n_candidates
                FROM members m
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
        written = con.execute(
            f"""
            SELECT count(*), count(DISTINCT struct_cluster_id), count(DISTINCT seq_cluster_id)
            FROM read_parquet({sql_string(glob)}, hive_partitioning=true)
            """
        ).fetchone()
        shard_count = con.execute(
            f"SELECT count(DISTINCT shard) FROM read_parquet({sql_string(glob)}, hive_partitioning=true)"
        ).fetchone()[0]
    finally:
        con.close()
    files = sorted(output.rglob("*.parquet"))
    return {
        "metadata_train_rows": counts[0],
        "untrained_clusters": untrained[0],
        "untrained_members": untrained[1],
        "untrained_usable_members": untrained[2],
        "planned_rows": written[0],
        "planned_clusters": written[1],
        "planned_sequence_clusters": written[2],
        "anchored_overlap_clusters": overlap,
        "shards": shard_count,
        "parquet_files": len(files),
        "parquet_bytes": sum(path.stat().st_size for path in files),
        "min_plddt": min_plddt,
        "min_seq_len": MIN_SEQ_LEN,
        "max_seq_len": MAX_SEQ_LEN,
        "exp53_min_cluster_size": EXP53_MIN_CLUSTER_SIZE,
        "seed": seed,
        "elapsed_seconds": perf_counter() - started,
        "metadata_glob": metadata_glob,
    }


def main() -> None:
    """Build and record the untrained-cluster plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchored-plan")
    parser.add_argument("--min-plddt", type=int, default=MIN_PLDDT)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit", default="24GB")
    args = parser.parse_args()
    stats = build_plan(
        args.metadata,
        args.output,
        threads=args.threads,
        memory_limit=args.memory_limit,
        anchored_plan_glob=args.anchored_plan,
        min_plddt=args.min_plddt,
    )
    record = {"status": "complete", "source": "afdb_v4_afdb24m_untrained_clusters", **stats}
    path = args.output.parent / "small-plan.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    record["plan_json_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
