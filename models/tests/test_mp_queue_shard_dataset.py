# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest

from marinfold_models.mp_queue_shard_dataset import MPQueueShardDocumentDataset


def _build_test_shard(epoch: int, shard_index: int):
    pid = os.getpid()
    return tuple(
        {
            "epoch": epoch,
            "shard_index": shard_index,
            "slot_index": slot_index,
            "pid": pid,
        }
        for slot_index in range(3)
    )


def _build_bad_shard(epoch: int, shard_index: int):
    del epoch, shard_index
    return ({"too_short": True},)


def test_mp_queue_shard_dataset_preserves_index_order():
    dataset = MPQueueShardDocumentDataset(
        build_shard=_build_test_shard,
        num_shards=5,
        examples_per_shard=3,
        seed=17,
        transform_workers=2,
        prefetch_shards=2,
        mp_start_method="fork",
    )
    try:
        indices = [0, 1, 2, 3, 7, 15, 16]
        actual = asyncio.run(dataset.get_batch(indices))
        expected_locations = [dataset.location_for_index(index) for index in indices]
        assert [
            (row["epoch"], row["shard_index"], row["slot_index"])
            for row in actual
        ] == expected_locations
        assert dataset.stats.examples_emitted == len(indices)
        assert dataset.stats.shards_scheduled >= 3
    finally:
        dataset.close()


def test_mp_queue_shard_dataset_reuses_built_shard_slots():
    dataset = MPQueueShardDocumentDataset(
        build_shard=_build_test_shard,
        num_shards=2,
        examples_per_shard=3,
        seed=0,
        transform_workers=2,
        prefetch_shards=0,
        mp_start_method="fork",
    )
    try:
        first = asyncio.run(dataset.get_batch([0, 1, 2]))
        assert {row["pid"] for row in first} == {first[0]["pid"]}
        assert dataset.stats.shards_loaded_from_worker == 1

        second = asyncio.run(dataset.get_batch([0, 1]))
        assert [row["pid"] for row in second] == [first[0]["pid"], first[1]["pid"]]
        assert dataset.stats.shards_loaded_from_cache >= 2
    finally:
        dataset.close()


def test_mp_queue_shard_dataset_validates_builder_slot_count():
    dataset = MPQueueShardDocumentDataset(
        build_shard=_build_bad_shard,
        num_shards=1,
        examples_per_shard=3,
        transform_workers=1,
        prefetch_shards=0,
        mp_start_method="fork",
    )
    try:
        with pytest.raises(ValueError, match="expected 3"):
            asyncio.run(dataset.get_batch([0]))
    finally:
        dataset.close()
