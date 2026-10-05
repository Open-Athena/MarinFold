"""Preparation may resume only complete outputs from unchanged source shards."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prepare import prepare_shard


def test_resume_checks_source_and_output_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"document": ["source"]}), source)
    out = tmp_path / "out"
    (out / "counts").mkdir(parents=True)
    (out / "train").mkdir()
    counts = {
        "source": str(source),
        "source_identity": {"size": source.stat().st_size, "etag": None},
        "train": 2,
        "validation": 0,
    }
    (out / "counts/source.parquet.json").write_text(json.dumps(counts))
    output = out / "train/source.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), output)
    assert prepare_shard((str(source), str(out), True)) == counts
    pq.write_table(pa.table({"id": [1]}), output)
    with pytest.raises(ValueError, match="Incomplete prepared shard"):
        prepare_shard((str(source), str(out), True))
    pq.write_table(pa.table({"document": ["source changed substantially"]}), source)
    with pytest.raises(ValueError, match="Source shard changed"):
        prepare_shard((str(source), str(out), True))
