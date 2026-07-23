# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateful streaming construction and online packing for premade contacts."""

import asyncio
import bisect
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.utils.jax_utils import local_cpu_mesh

from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    VOCABULARY,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.vocab import EOS


@dataclass(frozen=True)
class EncodedDocument:
    """One generated document waiting to be packed."""

    source_id: str
    tokens: np.ndarray
    segment_id: int


@dataclass
class PackedDocuments:
    """Mutable bin used by the online best-fit packer."""

    documents: list[EncodedDocument] = field(default_factory=list)
    used_tokens: int = 0

    def add(self, document: EncodedDocument) -> None:
        self.documents.append(document)
        self.used_tokens += len(document.tokens)


@dataclass
class StreamingStats:
    """Operational counters for inspecting the stateful loader."""

    shards_read: int = 0
    documents_constructed: int = 0
    documents_dropped: int = 0
    packs_emitted: int = 0
    real_tokens_emitted: int = 0
    padding_tokens_emitted: int = 0
    truncated_documents: int = 0

    @property
    def packing_utilization(self) -> float:
        total = self.real_tokens_emitted + self.padding_tokens_emitted
        return self.real_tokens_emitted / total if total else 0.0


def shard_indices_for_process(
    *,
    num_shards: int,
    seed: int,
    epoch: int,
    process_index: int,
    process_count: int,
) -> tuple[int, ...]:
    """Shuffle shards per epoch and partition them disjointly across processes."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if process_count <= 0:
        raise ValueError("process_count must be positive")
    if not 0 <= process_index < process_count:
        raise ValueError(
            f"process_index must be in [0, {process_count}), got {process_index}"
        )

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    shuffled = rng.permutation(num_shards)
    return tuple(
        int(shard_index)
        for position, shard_index in enumerate(shuffled)
        if position % process_count == process_index
    )


class OnlineBestFitPacker:
    """Bounded best-fit packer that carries partial bins across loader calls."""

    def __init__(
        self,
        *,
        max_seq_len: int,
        max_segments_per_example: int,
        min_fill_fraction: float,
        max_open_packs: int,
        stats: StreamingStats,
    ):
        if not 0.0 < min_fill_fraction <= 1.0:
            raise ValueError("min_fill_fraction must be in (0, 1]")
        if max_open_packs <= 0:
            raise ValueError("max_open_packs must be positive")

        self.max_seq_len = max_seq_len
        self.max_segments_per_example = max_segments_per_example
        self.min_fill_tokens = int(np.ceil(max_seq_len * min_fill_fraction))
        self.max_open_packs = max_open_packs
        self.stats = stats

        self._packs: dict[int, PackedDocuments] = {}
        self._by_remaining: list[tuple[int, int]] = []
        self._ready: deque[PackedDocuments] = deque()
        self._next_pack_id = 0

    @property
    def open_pack_count(self) -> int:
        return len(self._packs)

    def add(self, document: EncodedDocument) -> None:
        if len(document.tokens) >= self.max_seq_len:
            if len(document.tokens) > self.max_seq_len:
                self.stats.truncated_documents += 1
                document = EncodedDocument(
                    source_id=document.source_id,
                    tokens=document.tokens[: self.max_seq_len],
                    segment_id=document.segment_id,
                )
            self._ready.append(
                PackedDocuments(
                    documents=[document],
                    used_tokens=len(document.tokens),
                )
            )
            return

        position = bisect.bisect_left(
            self._by_remaining,
            (len(document.tokens), -1),
        )
        if position == len(self._by_remaining):
            pack_id = self._next_pack_id
            self._next_pack_id += 1
            pack = PackedDocuments()
            self._packs[pack_id] = pack
        else:
            _, pack_id = self._by_remaining.pop(position)
            pack = self._packs[pack_id]

        pack.add(document)
        should_emit = (
            pack.used_tokens >= self.min_fill_tokens
            or len(pack.documents) >= self.max_segments_per_example
        )
        if should_emit:
            self._ready.append(self._packs.pop(pack_id))
        else:
            remaining = self.max_seq_len - pack.used_tokens
            bisect.insort(self._by_remaining, (remaining, pack_id))

        while len(self._packs) > self.max_open_packs:
            _, fullest_pack_id = self._by_remaining.pop(0)
            self._ready.append(self._packs.pop(fullest_pack_id))

    def pop_ready(self) -> PackedDocuments | None:
        return self._ready.popleft() if self._ready else None


class ShuffledShardStream:
    """Read shuffled row blocks from shuffled, process-disjoint parquet shards."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int,
        seed: int,
        process_index: int,
        process_count: int,
        row_block_size: int,
        stats: StreamingStats,
    ):
        if num_shards > total_shards:
            raise ValueError("num_shards cannot exceed total_shards")
        if row_block_size <= 0:
            raise ValueError("row_block_size must be positive")
        if process_count > num_shards:
            raise ValueError(
                f"{process_count} processes cannot receive disjoint work from "
                f"only {num_shards} shards"
            )

        self.data_prefix = data_prefix.rstrip("/")
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.seed = seed
        self.process_index = process_index
        self.process_count = process_count
        self.row_block_size = row_block_size
        self.stats = stats

        self.epoch = 0
        self._local_shards: tuple[int, ...] = ()
        self._shard_cursor = 0
        self._current_table: pa.Table | None = None
        self._current_shard_index: int | None = None
        self._row_blocks: tuple[int, ...] = ()
        self._row_block_cursor = 0
        self._rows: deque[dict[str, Any]] = deque()
        self._reset_epoch()

    def next_row(self) -> dict[str, Any]:
        while not self._rows:
            if (
                self._current_table is None
                or self._row_block_cursor >= len(self._row_blocks)
            ):
                self._load_next_shard()
            self._load_next_row_block()
        return self._rows.popleft()

    def _reset_epoch(self) -> None:
        self._local_shards = shard_indices_for_process(
            num_shards=self.num_shards,
            seed=self.seed,
            epoch=self.epoch,
            process_index=self.process_index,
            process_count=self.process_count,
        )
        self._shard_cursor = 0

    def _load_next_shard(self) -> None:
        if self._shard_cursor >= len(self._local_shards):
            self.epoch += 1
            self._reset_epoch()

        shard_index = self._local_shards[self._shard_cursor]
        self._shard_cursor += 1
        shard_name = (
            f"shard-{shard_index:05d}-of-{self.total_shards:05d}.parquet"
        )
        shard_path = str(PurePosixPath(self.data_prefix) / shard_name)
        with fsspec.open(shard_path, "rb") as source:
            table = pq.read_table(source, columns=list(ANALYZED_ROW_COLUMNS))
        if table.num_rows == 0:
            raise ValueError(f"{shard_path} contains no rows")

        block_count = (table.num_rows + self.row_block_size - 1) // self.row_block_size
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, shard_index])
        )
        self._row_blocks = tuple(int(index) for index in rng.permutation(block_count))
        self._row_block_cursor = 0
        self._current_table = table
        self._current_shard_index = shard_index
        self.stats.shards_read += 1

    def _load_next_row_block(self) -> None:
        assert self._current_table is not None
        assert self._current_shard_index is not None
        block_index = self._row_blocks[self._row_block_cursor]
        self._row_block_cursor += 1
        start = block_index * self.row_block_size
        stop = min(start + self.row_block_size, self._current_table.num_rows)

        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch, self._current_shard_index, block_index]
            )
        )
        row_indices = rng.permutation(np.arange(start, stop, dtype=np.int64))
        rows = self._current_table.take(pa.array(row_indices)).to_pylist()
        self._rows.extend(rows)


class StreamingPremadeContactsDataset(AsyncDataset[GrugLmExample]):
    """Stateful, non-random-access stream of online-packed contacts documents.

    This deliberately adapts a stream to Levanter's ``AsyncDataset`` seam.
    ``get_batch(indices)`` uses only the requested count; it does not interpret
    the index values. The training config must therefore disable outer dataset
    shuffling and set ``mixture_block_size=1``.

    The stream is not currently checkpoint-resume safe. Levanter prefetches
    batches ahead of the optimizer, and this prototype does not checkpoint the
    shard cursor, row buffer, open packing bins, or prefetched examples.
    """

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        min_fill_fraction: float = 0.99,
        max_open_packs: int = 256,
        row_block_size: int = 256,
        process_index: int | None = None,
        process_count: int | None = None,
    ):
        self.data_prefix = data_prefix
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.max_segments_per_example = max_segments_per_example
        self.min_fill_fraction = min_fill_fraction
        self.max_open_packs = max_open_packs
        self.row_block_size = row_block_size
        self.process_index = process_index
        self.process_count = process_count

        self.stats = StreamingStats()
        self._source: ShuffledShardStream | None = None
        self._packer: OnlineBestFitPacker | None = None
        self._next_segment_id = 0
        self._peeked_example: GrugLmExample | None = None
        self._lock = asyncio.Lock()

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("StreamingPremadeContactsDataset is an infinite stream")

    async def getitem_async(self, index: int) -> GrugLmExample:
        del index
        async with self._lock:
            if self._peeked_example is None:
                self._peeked_example = await asyncio.to_thread(self._next_example_sync)
            return self._peeked_example

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        if not indices:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._next_batch_sync, len(indices))

    def _next_batch_sync(self, count: int) -> list[GrugLmExample]:
        output: list[GrugLmExample] = []
        if self._peeked_example is not None:
            output.append(self._peeked_example)
            self._peeked_example = None
        while len(output) < count:
            output.append(self._next_example_sync())
        return output

    def _next_example_sync(self) -> GrugLmExample:
        self._ensure_stream()
        assert self._packer is not None
        pack = self._packer.pop_ready()
        while pack is None:
            self._packer.add(self._next_document())
            pack = self._packer.pop_ready()
        return self._example_from_pack(pack)

    def _ensure_stream(self) -> None:
        if self._source is not None:
            return
        process_index = (
            self.process_index
            if self.process_index is not None
            else jax.process_index()
        )
        process_count = (
            self.process_count
            if self.process_count is not None
            else jax.process_count()
        )
        self._source = ShuffledShardStream(
            data_prefix=self.data_prefix,
            num_shards=self.num_shards,
            total_shards=self.total_shards,
            seed=self.seed,
            process_index=process_index,
            process_count=process_count,
            row_block_size=self.row_block_size,
            stats=self.stats,
        )
        self._packer = OnlineBestFitPacker(
            max_seq_len=self.max_seq_len,
            max_segments_per_example=self.max_segments_per_example,
            min_fill_fraction=self.min_fill_fraction,
            max_open_packs=self.max_open_packs,
            stats=self.stats,
        )

    def _next_document(self) -> EncodedDocument:
        assert self._source is not None
        while True:
            row = self._source.next_row()
            analyzed = analyzed_from_row(row)
            generated = build_document(
                analyzed.entry_id,
                analyzed.residues,
                analyzed.contacts,
                global_plddt=analyzed.global_plddt,
            )
            if generated is None:
                self.stats.documents_dropped += 1
                continue

            tokens = np.asarray(
                [
                    *(int(token) for token in VOCABULARY.encode(generated.document.split())),
                    int(EOS),
                ],
                dtype=np.int32,
            )
            document = EncodedDocument(
                source_id=analyzed.entry_id,
                tokens=tokens,
                segment_id=self._next_segment_id,
            )
            self._next_segment_id += 1
            self.stats.documents_constructed += 1
            return document

    def _example_from_pack(self, pack: PackedDocuments) -> GrugLmExample:
        tokens = np.zeros(self.max_seq_len, dtype=np.int32)
        segment_ids = np.full(self.max_seq_len, -1, dtype=np.int32)
        cursor = 0
        for document in pack.documents:
            stop = cursor + len(document.tokens)
            tokens[cursor:stop] = document.tokens
            segment_ids[cursor:stop] = document.segment_id
            cursor = stop

        self.stats.packs_emitted += 1
        self.stats.real_tokens_emitted += cursor
        self.stats.padding_tokens_emitted += self.max_seq_len - cursor
        with local_cpu_mesh():
            return GrugLmExample.causal(
                tokens=jnp.asarray(tokens),
                loss_weight=jnp.ones(self.max_seq_len, dtype=jnp.float32),
                segment_ids=jnp.asarray(segment_ids),
                max_segments=self.max_segments_per_example + 1,
                block_cross_document_attention=True,
            )


__all__ = [
    "EncodedDocument",
    "OnlineBestFitPacker",
    "PackedDocuments",
    "StreamingPremadeContactsDataset",
    "StreamingStats",
    "shard_indices_for_process",
]
