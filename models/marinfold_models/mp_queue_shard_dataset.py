# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Multiprocess shard-prefetch dataset helpers.

The Levanter ``DataLoader`` already has a background iterator and a bounded
batch queue, but expensive Python transforms inside ``AsyncDataset.get_batch``
still run behind a single producer. ``MPQueueShardDocumentDataset`` provides a
small shard-oriented producer pool for datasets whose indices naturally map to
fixed-size shard slots: worker processes build whole shards ahead of demand,
and the training process serves examples from a bounded in-memory shard cache.
"""

import asyncio
import multiprocessing as mp
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

from levanter.data.dataset import AsyncDataset


Example = TypeVar("Example")
ShardBuilder = Callable[[int, int], Sequence[Example]]


@dataclass
class MPQueueShardDatasetStats:
    """Operational counters for ``MPQueueShardDocumentDataset``."""

    shards_scheduled: int = 0
    shards_loaded_from_cache: int = 0
    shards_loaded_from_worker: int = 0
    examples_emitted: int = 0


def _build_shard_in_worker(builder: ShardBuilder[Example], epoch: int, shard_index: int) -> tuple[Example, ...]:
    return tuple(builder(epoch, shard_index))


class MPQueueShardDocumentDataset(AsyncDataset[Example], Generic[Example]):
    """Infinite fixed-slot dataset with multiprocess shard prefetch.

    Each global example index maps to ``(epoch, shard_index, slot_index)``.
    ``build_shard(epoch, shard_index)`` must return exactly
    ``examples_per_shard`` examples for that shard. The main process preserves
    deterministic index order, while worker processes build shard slots ahead of
    the trainer.

    This class is intentionally shard-level rather than per-example: for
    document datasets, reading a parquet shard, reconstructing documents, and
    packing slots is the expensive unit of work. Building a shard once and
    serving many subsequent slots avoids duplicating that setup for every
    example.
    """

    def __init__(
        self,
        *,
        build_shard: ShardBuilder[Example],
        num_shards: int,
        examples_per_shard: int,
        seed: int = 0,
        transform_workers: int = 4,
        prefetch_shards: int | None = None,
        shard_cache_size: int | None = None,
        mp_start_method: str = "spawn",
    ):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be positive")
        if transform_workers <= 0:
            raise ValueError("transform_workers must be positive")

        self.build_shard = build_shard
        self.num_shards = num_shards
        self.examples_per_shard = examples_per_shard
        self.seed = seed
        self.transform_workers = transform_workers
        self.prefetch_shards = transform_workers if prefetch_shards is None else prefetch_shards
        self.shard_cache_size = max(
            1,
            transform_workers + self.prefetch_shards if shard_cache_size is None else shard_cache_size,
        )
        if self.prefetch_shards < 0:
            raise ValueError("prefetch_shards must be non-negative")
        if self.shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")

        self.mp_start_method = mp_start_method
        self.stats = MPQueueShardDatasetStats()
        self._executor: ProcessPoolExecutor | None = None
        self._async_lock = asyncio.Lock()
        self._shard_orders: dict[int, tuple[int, ...]] = {}
        self._shard_positions: dict[int, dict[int, int]] = {}
        self._shard_cache: OrderedDict[tuple[int, int], tuple[Example, ...]] = OrderedDict()
        self._futures: dict[tuple[int, int], Future[tuple[Example, ...]]] = {}

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("MPQueueShardDocumentDataset is an infinite stream")

    async def getitem_async(self, index: int) -> Example:
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[Example]:
        if not indices:
            return []
        async with self._async_lock:
            return await asyncio.to_thread(self._get_batch_sync, tuple(indices))

    def location_for_index(self, index: int) -> tuple[int, int, int]:
        """Return ``(epoch, shard_index, slot_index)`` for a global index."""
        if index < 0:
            raise IndexError("dataset indices must be non-negative")
        examples_per_epoch = self.num_shards * self.examples_per_shard
        epoch, index_within_epoch = divmod(index, examples_per_epoch)
        shard_position, slot_index = divmod(index_within_epoch, self.examples_per_shard)
        shard_index = self._shard_order(epoch)[shard_position]
        return epoch, shard_index, slot_index

    def close(self) -> None:
        """Stop worker processes and drop pending futures."""
        executor = self._executor
        self._executor = None
        self._futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        self.close()

    def _get_batch_sync(self, indices: tuple[int, ...]) -> list[Example]:
        locations = [self.location_for_index(index) for index in indices]
        for epoch, shard_index, _ in locations:
            self._schedule_prefetch_window(epoch, shard_index)

        output: list[Example] = []
        for epoch, shard_index, slot_index in locations:
            slots = self._slots_for_shard(epoch, shard_index)
            output.append(slots[slot_index])
            self.stats.examples_emitted += 1
        return output

    def _slots_for_shard(self, epoch: int, shard_index: int) -> tuple[Example, ...]:
        key = (epoch, shard_index)
        cached = self._shard_cache.get(key)
        if cached is not None:
            self._shard_cache.move_to_end(key)
            self.stats.shards_loaded_from_cache += 1
            return cached

        self._ensure_scheduled(key)
        future = self._futures.pop(key)
        slots = future.result()
        if len(slots) != self.examples_per_shard:
            raise ValueError(
                f"Shard builder returned {len(slots)} slots for shard {key}; "
                f"expected {self.examples_per_shard}"
            )
        self.stats.shards_loaded_from_worker += 1
        self._remember_shard(key, slots)
        return slots

    def _schedule_prefetch_window(self, epoch: int, shard_index: int) -> None:
        self._ensure_scheduled((epoch, shard_index))
        position = self._shard_position(epoch, shard_index)
        for offset in range(1, self.prefetch_shards + 1):
            prefetch_epoch = epoch
            prefetch_position = position + offset
            while prefetch_position >= self.num_shards:
                prefetch_epoch += 1
                prefetch_position -= self.num_shards
            prefetch_shard = self._shard_order(prefetch_epoch)[prefetch_position]
            self._ensure_scheduled((prefetch_epoch, prefetch_shard))

    def _ensure_scheduled(self, key: tuple[int, int]) -> None:
        if key in self._shard_cache or key in self._futures:
            return
        epoch, shard_index = key
        future = self._pool().submit(_build_shard_in_worker, self.build_shard, epoch, shard_index)
        self._futures[key] = future
        self.stats.shards_scheduled += 1

    def _remember_shard(self, key: tuple[int, int], slots: tuple[Example, ...]) -> None:
        self._shard_cache[key] = slots
        self._shard_cache.move_to_end(key)
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            context = mp.get_context(self.mp_start_method)
            self._executor = ProcessPoolExecutor(max_workers=self.transform_workers, mp_context=context)
        return self._executor

    def _shard_order(self, epoch: int) -> tuple[int, ...]:
        cached = self._shard_orders.get(epoch)
        if cached is not None:
            return cached
        order_list = list(range(self.num_shards))
        random.Random(f"{self.seed}:{epoch}").shuffle(order_list)
        order = tuple(order_list)
        self._shard_orders[epoch] = order
        self._shard_positions[epoch] = {shard_index: position for position, shard_index in enumerate(order)}
        return order

    def _shard_position(self, epoch: int, shard_index: int) -> int:
        self._shard_order(epoch)
        return self._shard_positions[epoch][shard_index]


__all__ = ["MPQueueShardDatasetStats", "MPQueueShardDocumentDataset"]
