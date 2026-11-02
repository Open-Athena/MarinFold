"""The held-out shard is chosen by published index, not by listing order."""

from dataclasses import dataclass

import pytest

from experiments.exp343_models_complex_corpus_training import stage
from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_BUCKET_PREFIX,
    COMPLEX_SHARDS,
    STAGED_TRAIN,
    STAGED_VALIDATION,
    VALIDATION_SHARD,
)


@dataclass(frozen=True)
class FakeEntry:
    path: str
    size: int


def listing(names: list[str]) -> list[FakeEntry]:
    return [
        FakeEntry(f"{COMPLEX_BUCKET_PREFIX}/train/{name}", 100 + index)
        for index, name in enumerate(names)
    ]


def shard_names(count: int) -> list[str]:
    return [f"shard-{index:05d}-of-{COMPLEX_SHARDS:05d}.parquet" for index in range(count)]


def test_shards_are_ordered_by_published_index(monkeypatch) -> None:
    shuffled = shard_names(COMPLEX_SHARDS)
    shuffled = shuffled[100:] + shuffled[:100]
    monkeypatch.setattr(stage, "list_bucket_tree", lambda *a, **k: listing(shuffled))
    shards = stage.published_shards(False)
    assert [index for index, _, _ in shards] == list(range(COMPLEX_SHARDS))
    # The size travels with the shard it was listed against, not with its rank.
    by_name = {entry.path: entry.size for entry in listing(shuffled)}
    assert all(by_name[path] == size for _, path, size in shards)


def test_incomplete_listing_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        stage, "list_bucket_tree", lambda *a, **k: listing(shard_names(COMPLEX_SHARDS - 1))
    )
    with pytest.raises(ValueError, match="shard indices are not"):
        stage.published_shards(False)


def test_unexpected_shard_total_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        stage,
        "list_bucket_tree",
        lambda *a, **k: listing(["shard-00000-of-00200.parquet"]),
    )
    with pytest.raises(ValueError, match="claims 00200 shards"):
        stage.published_shards(False)


def test_only_the_held_out_index_routes_to_the_validation_prefix() -> None:
    names = shard_names(COMPLEX_SHARDS)
    validation = [
        index
        for index, name in enumerate(names)
        if stage.destination(index, name).startswith(STAGED_VALIDATION)
    ]
    assert validation == [VALIDATION_SHARD]
    assert all(
        stage.destination(index, names[index]).startswith(STAGED_TRAIN)
        for index in range(COMPLEX_SHARDS)
        if index != VALIDATION_SHARD
    )
