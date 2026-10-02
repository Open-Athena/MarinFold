"""Verify rank-disjoint coverage and exact continuation across shard boundaries."""

import pyarrow as pa
import pyarrow.parquet as pq
from train import DocumentStream


def test_resume_cursor_and_rank_coverage(tmp_path) -> None:
    for shard in range(4):
        rows = [
            {"entry_id": f"{shard}-{i}", "contacts_v1": [i, i + 1]} for i in range(3)
        ]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"{shard}.parquet")
    ranks = [DocumentStream(str(tmp_path), "contacts_v1", rank, 2) for rank in range(2)]
    seen = [{stream.next()["entry_id"] for _ in range(6)} for stream in ranks]
    assert len(seen[0] | seen[1]) == 12
    assert not seen[0] & seen[1]

    original = DocumentStream(str(tmp_path), "contacts_v1", 0, 2)
    for _ in range(5):
        original.next()
    cursor = original.state.copy()
    expected = [original.next()["entry_id"] for _ in range(8)]
    resumed = DocumentStream(str(tmp_path), "contacts_v1", 0, 2)
    resumed.state = cursor
    assert [resumed.next()["entry_id"] for _ in range(8)] == expected
