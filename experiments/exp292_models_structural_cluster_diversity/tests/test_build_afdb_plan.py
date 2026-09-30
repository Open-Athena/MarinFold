"""Tests for AFDB metadata-only production planning."""

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from build_afdb_plan import build_plan


def test_afdb_plan_keeps_anchors_and_bounds_candidates(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.parquet"
    index = tmp_path / "index.parquet"
    droplist = tmp_path / "droplist.parquet"
    output = tmp_path / "output" / "plan"
    rows = []
    for cluster, count in [("cluster-a", 12), ("cluster-b", 2)]:
        for item in range(count):
            rows.append(
                {
                    "entry_id": f"{cluster}-{item}",
                    "gcs_uri": f"gs://source/{cluster}-{item}.cif",
                    "struct_cluster_id": cluster,
                    "seq_cluster_id": f"seq-{cluster}-{item}",
                    "global_plddt": 90.0,
                    "seq_len": 100,
                    "split": "train",
                    "uniprot_accession": f"U{cluster}{item}",
                    "tax_id": 1,
                    "organism_name": "test",
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    pq.write_table(
        pa.table(
            {
                "entry_id": ["cluster-a-0", "cluster-a-1", "cluster-b-0"],
                "struct_cluster_id": ["cluster-a", "cluster-a", "cluster-b"],
                "split": ["train", "train", "train"],
            }
        ),
        index,
    )
    pq.write_table(
        pa.table(
            {
                "arm": ["afdb", "afdb", "esm_atlas"],
                "entry_id": ["cluster-a-11", "cluster-b-1", "unrelated"],
            }
        ),
        droplist,
    )
    stats = build_plan(
        str(metadata),
        index,
        droplist,
        output,
        threads=2,
        memory_limit="1GB",
    )
    table = ds.dataset(output, format="parquet", partitioning="hive").to_table()
    assert table.num_rows == 10
    assert sum(table["is_anchor"].to_pylist()) == 2
    assert stats["eligible_clusters"] == 1
    assert stats["reservoir_candidate_rows"] == 8
    assert stats["eligible_omitted_members"] == 9
    assert stats["theoretical_additions_before_sequence_filter"] == 3
