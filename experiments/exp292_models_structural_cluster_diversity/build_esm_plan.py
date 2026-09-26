"""Build the production ESM cluster plan beside the original AWS sources.

This stage reads no coordinates. It joins the frozen exp91 representative and
membership files to the current exp225 droplist, removes clusters without a
retained training anchor, and keeps a deterministic eight-member reservoir of
omitted proteins per eligible original cluster. The compact, partitioned plan
feeds the later sequence-quality and structural-ranking workers.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import boto3
import duckdb
import requests

SOURCE_BUCKET = "marinfold-exp91-usw2"
SELECTED_KEY = "exp91/out/selected_manifest.csv"
SELECTED_ETAG = '"10ff0d023ef357e1e37daf8f377c8e88-1030"'
SELECTED_BYTES = 8_639_986_351
MEMBERSHIP_KEY = "exp91/out/clu_cluster.tsv"
MEMBERSHIP_ETAG = '"916af8a6dfca31c05825dfbb34550e2e-1284"'
MEMBERSHIP_BYTES = 10_767_514_098
SEED = 292
RESERVOIR_CAP = 8


def sql_string(value: str | Path) -> str:
    """Quote a filesystem path as a DuckDB SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def require_source_region() -> None:
    """Refuse the production source scan outside AWS us-west-2."""
    session = requests.Session()
    session.trust_env = False
    base = "http://169.254.169.254/latest/"
    token = session.put(
        base + "api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        timeout=5,
    )
    token.raise_for_status()
    identity = session.get(
        base + "dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token.text},
        timeout=5,
    )
    identity.raise_for_status()
    if identity.json()["region"] != "us-west-2":
        raise ValueError("The exp91 production sources must be scanned in us-west-2")


def download_pinned_source(
    s3, key: str, etag: str, expected_bytes: int, destination: Path
) -> dict:
    """Download one immutable input after checking its recorded object identity."""
    head = s3.head_object(Bucket=SOURCE_BUCKET, Key=key)
    if head["ETag"] != etag or head["ContentLength"] != expected_bytes:
        raise ValueError(
            f"Source object changed: s3://{SOURCE_BUCKET}/{key} has "
            f"ETag={head['ETag']} bytes={head['ContentLength']}"
        )
    if not destination.exists() or destination.stat().st_size != expected_bytes:
        temporary = destination.with_suffix(destination.suffix + ".part")
        s3.download_file(SOURCE_BUCKET, key, str(temporary))
        temporary.replace(destination)
    return {
        "uri": f"s3://{SOURCE_BUCKET}/{key}",
        "etag": etag,
        "bytes": expected_bytes,
    }


def build_plan(
    selected: Path,
    membership: Path,
    droplist: Path,
    output: Path,
    *,
    threads: int,
    memory_limit: str,
    reservoir_cap: int = RESERVOIR_CAP,
    seed: int = SEED,
) -> dict:
    """Join frozen metadata and write a sharded candidate-reservoir dataset."""
    if reservoir_cap < 3:
        raise ValueError("The reservoir must hold at least the three output slots")
    output.mkdir(parents=True, exist_ok=True)
    database = output.parent / "plan.duckdb"
    temporary = output.parent / "duckdb-tmp"
    temporary.mkdir(exist_ok=True)
    con = duckdb.connect(str(database))
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit={sql_string(memory_limit)}")
    con.execute(f"SET temp_directory={sql_string(temporary)}")
    con.execute("SET preserve_insertion_order=false")
    started = perf_counter()
    try:
        con.execute("DROP TABLE IF EXISTS anchors")
        con.execute(
            f"""
            CREATE TABLE anchors AS
            SELECT
                s.cluster_id,
                s.protein_hash AS anchor_id,
                CAST(s.seq_len AS INTEGER) AS anchor_seq_len,
                CAST(s.mean_plddt AS DOUBLE) AS anchor_mean_plddt,
                CAST(s.ptm AS DOUBLE) AS anchor_ptm,
                CAST(s.cluster_size AS BIGINT) AS cluster_size
            FROM read_csv(
                {sql_string(selected)},
                header=true,
                delim=',',
                columns={{
                    'cluster_id': 'VARCHAR',
                    'protein_hash': 'VARCHAR',
                    'seq_len': 'INTEGER',
                    'mean_plddt': 'DOUBLE',
                    'ptm': 'DOUBLE',
                    'plddt_std': 'DOUBLE',
                    'cluster_size': 'BIGINT'
                }},
                parallel=true
            ) s
            ANTI JOIN (
                SELECT DISTINCT entry_id
                FROM read_parquet({sql_string(droplist)})
                WHERE arm = 'esm_atlas'
            ) d ON s.protein_hash = d.entry_id
            WHERE s.mean_plddt >= 0.8
              AND s.ptm >= 0.5
              AND s.seq_len BETWEEN 60 AND 1000
              AND s.cluster_size BETWEEN 2 AND 100000
            """
        )
        duplicate_clusters = con.execute(
            "SELECT count(*) FROM (SELECT cluster_id FROM anchors GROUP BY 1 HAVING count(*) != 1)"
        ).fetchone()[0]
        if duplicate_clusters:
            raise ValueError(f"Selected manifest has {duplicate_clusters} duplicate clusters")
        anchor_count = con.execute("SELECT count(*) FROM anchors").fetchone()[0]
        if not anchor_count:
            raise ValueError("No eligible retained ESM anchors")

        if any(output.iterdir()):
            shutil.rmtree(output)
            output.mkdir()
        membership_reader = f"""read_csv(
            {sql_string(membership)},
            header=false,
            delim='\\t',
            columns={{'cluster_id': 'VARCHAR', 'member_id': 'VARCHAR'}},
            parallel=true,
            strict_mode=true
        )"""
        con.execute(
            f"""
            CREATE OR REPLACE TABLE reservoirs AS
                WITH unique_members AS (
                    SELECT
                        m.cluster_id,
                        m.member_id,
                        count(*) AS multiplicity
                    FROM {membership_reader} m
                    JOIN anchors a USING (cluster_id)
                    GROUP BY m.cluster_id, m.member_id
                ) SELECT
                        a.cluster_id,
                        a.anchor_id,
                        a.anchor_seq_len,
                        a.anchor_mean_plddt,
                        a.anchor_ptm,
                        a.cluster_size,
                        min_by(
                            m.member_id,
                            md5('{seed}:' || m.member_id),
                            {reservoir_cap}
                        ) FILTER (WHERE m.member_id != a.anchor_id) AS candidate_ids,
                        count(*) FILTER (WHERE m.member_id != a.anchor_id)
                            AS unique_omitted_members,
                        sum(m.multiplicity) AS membership_rows_observed,
                        sum(m.multiplicity) FILTER (WHERE m.member_id = a.anchor_id)
                            AS anchor_multiplicity,
                        sum(m.multiplicity - 1) AS duplicate_membership_rows
                FROM unique_members m
                JOIN anchors a USING (cluster_id)
                GROUP BY ALL
            """
        )
        mismatch_count = con.execute(
            "SELECT count(*) FROM reservoirs WHERE membership_rows_observed != cluster_size"
        ).fetchone()[0]
        mismatch_path = output.parent / "cardinality_mismatches.parquet"
        if mismatch_count:
            con.execute(
                f"""
                COPY (
                    SELECT
                        cluster_id,
                        anchor_id,
                        cluster_size,
                        membership_rows_observed,
                        unique_omitted_members,
                        anchor_multiplicity,
                        duplicate_membership_rows,
                        candidate_ids
                    FROM reservoirs
                    WHERE membership_rows_observed != cluster_size
                    ORDER BY cluster_id
                ) TO {sql_string(mismatch_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        missing_membership_anchors = con.execute(
            "SELECT count(*) FROM anchors ANTI JOIN reservoirs USING (cluster_id)"
        ).fetchone()[0]
        if missing_membership_anchors:
            raise ValueError(
                f"{missing_membership_anchors} eligible anchors have no membership rows"
            )
        con.execute(
            f"""
            COPY (
                SELECT
                    substr(md5(cluster_id || ':{seed}'), 1, 2) AS shard,
                    *,
                    len(candidate_ids) AS reservoir_size
                FROM reservoirs
                WHERE len(candidate_ids) > 0
                  AND anchor_multiplicity >= 1
                  AND membership_rows_observed = cluster_size
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
                count(*) AS clusters,
                sum(reservoir_size) AS candidate_rows,
                sum(least(3, unique_omitted_members)) AS theoretical_additions,
                sum(unique_omitted_members) AS omitted_members,
                count(DISTINCT anchor_id) AS distinct_anchors,
                sum(duplicate_membership_rows) AS duplicate_membership_rows,
                count(*) FILTER (WHERE anchor_multiplicity > 1)
                    AS clusters_with_duplicate_anchor_rows
            FROM read_parquet({sql_string(glob)}, hive_partitioning=true)
            """
        ).fetchone()
        if counts[0] != counts[4]:
            raise ValueError("An anchor occurs in more than one eligible cluster")
        shard_count = con.execute(
            f"SELECT count(DISTINCT shard) FROM read_parquet({sql_string(glob)}, hive_partitioning=true)"
        ).fetchone()[0]
    finally:
        con.close()
    files = sorted(output.rglob("*.parquet"))
    return {
        "eligible_anchors_before_membership_join": anchor_count,
        "planned_clusters": counts[0],
        "reservoir_candidate_rows": counts[1],
        "theoretical_additions_before_candidate_quality": counts[2],
        "eligible_omitted_members": counts[3],
        "membership_cardinality_mismatches_excluded": mismatch_count,
        "eligible_anchors_missing_membership": missing_membership_anchors,
        "distinct_anchors": counts[4],
        "duplicate_membership_rows": counts[5],
        "clusters_with_duplicate_anchor_rows": counts[6],
        "shards": shard_count,
        "parquet_files": len(files),
        "parquet_bytes": sum(path.stat().st_size for path in files),
        "reservoir_cap": reservoir_cap,
        "seed": seed,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    """Download pinned inputs, build the plan, and record exact provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--droplist", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit", default="100GB")
    args = parser.parse_args()
    require_source_region()
    args.work.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-west-2")
    selected = args.work / "selected_manifest.csv"
    membership = args.work / "clu_cluster.tsv"
    inputs = {
        "selected_manifest": download_pinned_source(
            s3, SELECTED_KEY, SELECTED_ETAG, SELECTED_BYTES, selected
        ),
        "membership": download_pinned_source(
            s3, MEMBERSHIP_KEY, MEMBERSHIP_ETAG, MEMBERSHIP_BYTES, membership
        ),
        "droplist": {
            "path": str(args.droplist),
            "bytes": args.droplist.stat().st_size,
            "sha256": hashlib.sha256(args.droplist.read_bytes()).hexdigest(),
        },
    }
    stats = build_plan(
        selected,
        membership,
        args.droplist,
        args.output,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    record = {
        "status": "complete",
        "source": "esmfold2_atlas_v1_exp91",
        "inputs": inputs,
        **stats,
    }
    (args.output.parent / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
