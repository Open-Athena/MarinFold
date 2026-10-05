# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from summarize_removal_tokens import summarize_source


def test_summarize_source_adds_eos_and_checks_available_shards(tmp_path: Path) -> None:
    removals = tmp_path / "removals.parquet"
    pq.write_table(
        pa.table(
            {
                "removed_source": ["afdb", "afdb"],
                "removed_shard": [0, 1],
                "removed_entry_id": ["a", "b"],
            }
        ),
        removals,
    )
    shard = tmp_path / "contacts_v1-00000-of-00002.parquet"
    pq.write_table(pa.table({"entry_id": ["a"], "num_tokens": [9]}), shard)

    result = summarize_source(
        removals, [shard], source="afdb", expected_shards=2
    )

    assert not result["complete_source_coverage"]
    assert result["total_candidate_removals"] == 2
    assert result["candidate_removals_in_available_shards"] == 1
    assert result["source_tokens_removed_in_available_shards"] == 10
