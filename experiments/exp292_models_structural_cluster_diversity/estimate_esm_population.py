"""Estimate the eligible ESM cluster population from sampled corpus metadata.

The first-stage sample chose 100 of 3,337 current training shards uniformly,
excluding the development shard. Recover its exact frozen projections and expand
stratum counts by the inverse shard inclusion probability. These are population
estimates, not a census or uncertainty intervals. Candidate quality and structural
yield are assessed separately by the source audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

from structure_audit import write_csv


def main() -> None:
    """Validate cached shard projections and save estimated eligible populations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sampling = json.loads(args.sampling.read_text())
    provenance = sampling["provenance"]
    indices = [r["corpus_shard"] for r in provenance]
    if len(set(indices)) != len(indices) or not set(indices) <= set(range(1, 3338)):
        raise ValueError("Expected distinct training shards from the documented frame")
    if len(indices) != sampling["corpus_shards"]:
        raise ValueError("Shard count disagrees with the sampling record")
    paths = []
    for row in provenance:
        path = args.cache / f"current-{row['corpus_shard']:05d}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["projected_sha256"]:
            raise ValueError(f"Metadata projection changed: {path}")
        paths.append(str(path))
    con = duckdb.connect()
    con.read_parquet(paths).create_view("current_metadata")
    observed, distinct = con.execute(
        "SELECT count(*),count(DISTINCT entry_id) FROM current_metadata"
    ).fetchone()
    if observed != distinct or observed != sampling["sampled_current_anchors"]:
        raise ValueError("Anchor count or uniqueness differs from the frozen sample")
    con.execute("""CREATE VIEW eligible AS SELECT * FROM current_metadata
        WHERE split='train' AND global_plddt>=80 AND seq_len BETWEEN 60 AND 1000
        AND cluster_size BETWEEN 2 AND 100000""")
    con.execute("""SELECT
        CASE WHEN seq_len<250 THEN '60-249' WHEN seq_len<500 THEN '250-499'
            ELSE '500-1000' END AS length_bin,
        CASE WHEN cluster_size<10 THEN '2-9' WHEN cluster_size<100 THEN '10-99'
            ELSE '100+' END AS size_bin,
        count(*) AS sampled_frame_clusters,
        sum(cluster_size-1) AS omitted_members,
        sum(least(3,cluster_size-1)) AS capacity_three,
        sum(least(32,cluster_size-1)) AS inspection_capacity
        FROM eligible GROUP BY 1,2 ORDER BY 1,2""")
    columns = [d[0] for d in con.description]
    factor = 3337 / len(indices)
    rows = [
        {**dict(zip(columns, row, strict=True)), "eligible_clusters": row[2] * factor}
        for row in con.fetchall()
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "population_estimate.csv", rows)
    record = {
        "sampling_sha256": hashlib.sha256(args.sampling.read_bytes()).hexdigest(),
        "sampled_shards": len(indices),
        "frame_shards": 3337,
        "shard_expansion_factor": factor,
        "sampled_current_anchors": observed,
        "projected_eligible_clusters": sum(r["eligible_clusters"] for r in rows),
        "projected_omitted_members": sum(r["omitted_members"] for r in rows) * factor,
        "projected_capacity_three": sum(r["capacity_three"] for r in rows) * factor,
        "projected_inspection_capacity_32": sum(r["inspection_capacity"] for r in rows)
        * factor,
        "status": "estimated sampling frame; candidate source quality, structural selection, domain review and sequence exclusion are additional filters",
        "limits": "Point estimates only; excludes development shard 0, anchors below pLDDT 80 or outside lengths 60-1000, and clusters larger than 100000. No cross-source deduplication is implied.",
    }
    (args.output / "population_estimate.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
