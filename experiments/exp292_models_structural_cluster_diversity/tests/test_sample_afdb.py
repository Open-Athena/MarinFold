"""Check sample membership against the actual retained training corpus."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sample_afdb import sample
from structure_audit import read_csv


def test_decontamination_and_split_membership_are_preserved(tmp_path: Path) -> None:
    rows = []
    for cluster, split in [
        ("train-cluster", "train"),
        ("test-cluster", "test"),
        ("purged-cluster", "train"),
    ]:
        for i in range(8):
            rows.append(
                {
                    "entry_id": f"{cluster}-{i}",
                    "struct_cluster_id": cluster,
                    "seq_cluster_id": cluster,
                    "split": split,
                    "seq_len": 150,
                    "global_plddt": 95.0 - i,
                    "gcs_uri": f"gs://test/{cluster}-{i}",
                }
            )
    metadata = tmp_path / "metadata.parquet"
    index = tmp_path / "index.parquet"
    drops = tmp_path / "drops.parquet"
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    training = [r for r in rows if int(r["entry_id"].rsplit("-", 1)[1]) < 2]
    pq.write_table(pa.Table.from_pylist(training), index)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"entry_id": "purged-cluster-0", "arm": "afdb"},
                {"entry_id": "purged-cluster-1", "arm": "afdb"},
                {"entry_id": "train-cluster-7", "arm": "afdb"},
            ]
        ),
        drops,
    )
    stats = sample(str(metadata), str(index), str(drops), tmp_path / "out", 4, 8, 292)
    selected = read_csv(tmp_path / "out/sample.csv")
    assert stats["training_anchor_rows"] == 2
    assert stats["sample_clusters"] == 1
    assert len(selected) == 7
    assert "train-cluster-7" not in {r["entry_id"] for r in selected}
    assert {r["struct_cluster_id"] for r in selected} == {"train-cluster"}
    assert {r["entry_id"] for r in selected if r["is_anchor"] == "true"} == {
        "train-cluster-0",
        "train-cluster-1",
    }
