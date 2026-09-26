"""Production ESM planning tests using tiny frozen-source analogues."""

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from build_esm_plan import build_plan


def write_selected(path: Path, rows: list[dict]) -> None:
    """Write rows with the exact exp91 manifest header."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "protein_hash",
                "seq_len",
                "mean_plddt",
                "ptm",
                "plddt_std",
                "cluster_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_plan_filters_anchors_and_bounds_reservoir(tmp_path: Path) -> None:
    selected = tmp_path / "selected.csv"
    membership = tmp_path / "membership.tsv"
    droplist = tmp_path / "droplist.parquet"
    output = tmp_path / "output" / "plan"
    write_selected(
        selected,
        [
            {
                "cluster_id": "cluster-a",
                "protein_hash": "anchor-a",
                "seq_len": 100,
                "mean_plddt": 0.9,
                "ptm": 0.8,
                "plddt_std": 0.1,
                "cluster_size": 13,
            },
            {
                "cluster_id": "cluster-b",
                "protein_hash": "anchor-b",
                "seq_len": 100,
                "mean_plddt": 0.9,
                "ptm": 0.8,
                "plddt_std": 0.1,
                "cluster_size": 2,
            },
            {
                "cluster_id": "cluster-c",
                "protein_hash": "anchor-c",
                "seq_len": 40,
                "mean_plddt": 0.9,
                "ptm": 0.8,
                "plddt_std": 0.1,
                "cluster_size": 2,
            },
        ],
    )
    lines = ["cluster-a\tanchor-a", "cluster-a\tanchor-a"] + [
        f"cluster-a\tmember-{index}" for index in range(10)
    ]
    lines.append("cluster-a\tmember-0")
    lines += ["cluster-b\tanchor-b", "cluster-b\tmember-b"]
    lines += ["cluster-c\tanchor-c", "cluster-c\tmember-c"]
    membership.write_text("\n".join(lines) + "\n")
    pq.write_table(
        pa.table(
            {
                "arm": ["afdb", "esm_atlas"],
                "entry_id": ["anchor-a", "anchor-b"],
            }
        ),
        droplist,
    )

    stats = build_plan(
        selected,
        membership,
        droplist,
        output,
        threads=2,
        memory_limit="1GB",
    )

    table = ds.dataset(output, format="parquet", partitioning="hive").to_table()
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["cluster_id"] == "cluster-a"
    assert row["anchor_id"] == "anchor-a"
    assert len(row["candidate_ids"]) == 8
    assert "anchor-a" not in row["candidate_ids"]
    assert stats["planned_clusters"] == 1
    assert stats["reservoir_candidate_rows"] == 8
    assert stats["theoretical_additions_before_candidate_quality"] == 3
    assert stats["eligible_omitted_members"] == 10
    assert stats["duplicate_membership_rows"] == 2
    assert stats["clusters_with_duplicate_anchor_rows"] == 1
