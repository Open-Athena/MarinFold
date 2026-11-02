"""Check durable timing/completion records without loading accelerator runtimes."""

import csv
from pathlib import Path

import pyarrow as pa
import pytest

from collect_timings import collect
from log_rollout_worker import done_stems, write_parquet, write_timing


def timing_record(stem: str) -> dict:
    return dict(stem=stem, elapsed_seconds=1.5, model_load_seconds=4.0,
                total_seconds=6.0, model_source="test-model", mode="raw_rollout")


def test_incomplete_output_is_retried_until_timing_is_durable(tmp_path: Path) -> None:
    part = tmp_path / "shard-000-of-001-part-0000.parquet"
    write_parquet(pa.table({"dataset": ["test"], "stem": ["protein"]}), str(part))
    assert done_stems(str(tmp_path), 0, 1) == (set(), 0)
    sidecar = part.with_suffix(".timings.csv")
    write_timing(timing_record("test__protein"), str(sidecar))
    assert done_stems(str(tmp_path), 0, 1) == ({"test__protein"}, 1)
    write_timing(timing_record("wrong__input"), str(sidecar))
    with pytest.raises(ValueError, match="disagree"):
        done_stems(str(tmp_path), 0, 1)


def test_corrupt_completed_output_is_not_silently_skipped(tmp_path: Path) -> None:
    part = tmp_path / "shard-000-of-001-part-0000.parquet"
    part.write_bytes(b"interrupted parquet")
    write_timing(timing_record("test__protein"), str(part.with_suffix(".timings.csv")))
    with pytest.raises(pa.ArrowInvalid):
        done_stems(str(tmp_path), 0, 1)


def test_collect_preserves_measurements_and_rejects_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.timings.csv"
    write_timing(timing_record("test__protein"), str(first))
    output = tmp_path / "collected.csv"
    assert collect(str(tmp_path / "*.timings.csv"), output) == 1
    with output.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["stem"] == "test__protein"
    assert float(row["elapsed_seconds"]) == 1.5
    write_timing(timing_record("test__protein"), str(tmp_path / "duplicate.timings.csv"))
    with pytest.raises(ValueError, match="Duplicate"):
        collect(str(tmp_path / "*.timings.csv"), output)
