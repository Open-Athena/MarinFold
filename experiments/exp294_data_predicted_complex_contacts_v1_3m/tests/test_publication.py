"""Publication must verify the complete planned corpus and every byte count."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish_to_hf import _verify_sizes, plan


def test_missing_shard_is_rejected_before_upload(tmp_path: Path) -> None:
    train = tmp_path / "corpus/train"
    train.mkdir(parents=True)
    (train / "shard-00000-of-00002.parquet").write_bytes(b"shard")
    (tmp_path / "corpus/corpus_stats.json").write_text(json.dumps({"shards": 2}))
    with pytest.raises(ValueError, match="complete set"):
        plan(tmp_path, "data/test")


@pytest.mark.parametrize("published", [{}, {"remote": None}, {"remote": 2}])
def test_publication_requires_present_matching_sizes(tmp_path: Path, published: dict) -> None:
    artifact = tmp_path / "shard"
    artifact.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="did not verify"):
        _verify_sizes([(artifact, "remote")], published, "data/test")
    _verify_sizes([(artifact, "remote")], {"remote": 3}, "data/test")


def test_remote_shards_from_an_older_release_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "shard"
    artifact.write_bytes(b"abc")
    remote = "data/test/train/shard-00000-of-00001.parquet"
    stale = "data/test/train/shard-00000-of-00002.parquet"
    with pytest.raises(RuntimeError, match="unexpected training artifact"):
        _verify_sizes([(artifact, remote)], {remote: 3, stale: 3}, "data/test")
