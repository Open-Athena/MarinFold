"""Tests for flattening and exact-joining an ESM production plan."""

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from locate_esm_rows import exact_join, prepare_wanted


def test_prepare_and_exact_join_preserve_lineage(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan" / "shard=aa"
    plan_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "cluster_id": ["c1"],
                "anchor_id": ["00000000000000000000000000000001"],
                "anchor_seq_len": [100],
                "anchor_mean_plddt": [0.9],
                "anchor_ptm": [0.8],
                "cluster_size": [3],
                "candidate_ids": pa.array(
                    [[
                        "00000000000000000000000000000002",
                        "00000000000000000000000000000003",
                    ]],
                    type=pa.list_(pa.string()),
                ),
                "unique_omitted_members": [2],
                "membership_rows_observed": [3],
                "anchor_multiplicity": [1],
                "duplicate_membership_rows": [0],
                "reservoir_size": [2],
            }
        ),
        plan_dir / "part.parquet",
    )
    wanted = tmp_path / "wanted.parquet"
    assert prepare_wanted(str(tmp_path / "plan" / "*" / "*.parquet"), wanted) == 3
    rows = pq.read_table(wanted).to_pylist()
    assert sorted(row["reservoir_rank"] for row in rows if not row["is_anchor"]) == [
        1,
        2,
    ]

    emit = tmp_path / "emit.parquet"
    pq.write_table(
        pa.table(
            {
                "row_index": [10, 20, 30, 40],
                "protein_hash": [
                    "00000000000000000000000000000001",
                    "00000000000000000000000000000002",
                    "00000000000000000000000000000003",
                    "0000000000000000ffffffffffffffff",
                ],
            }
        ),
        emit,
    )
    output = tmp_path / "located"
    stats = exact_join(wanted, [emit], output)
    located = ds.dataset(output, format="parquet", partitioning="hive").to_table()
    assert stats["located_rows"] == 3
    assert set(located["row_index"].to_pylist()) == {10, 20, 30}
    assert set(located["cluster_id"].to_pylist()) == {"c1"}
