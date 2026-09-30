# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless on-the-fly document construction with fixed per-shard output."""

import asyncio
import bisect
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import fsspec
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.utils.jax_utils import local_cpu_mesh

from marinfold.document_structures.documents import (
    AttentionLayout,
    Document,
    pack,
)


Example = TypeVar("Example")
DocumentGenerator = Callable[[Mapping[str, Any]], Document | None]
ExampleBuilder = Callable[[tuple[Document, ...], int, int], Example]


@dataclass
class PackedDocuments:
    """One packed model example before padding and example conversion."""

    documents: list[Document] = field(default_factory=list)
    used_tokens: int = 0

    def add(self, document: Document) -> None:
        self.documents.append(document)
        self.used_tokens += len(document)


@dataclass
class FixedQuotaShardStats:
    """Operational counters for a fixed-quota shard dataset."""

    shards_constructed: int = 0
    documents_constructed: int = 0
    documents_dropped_by_generator: int = 0
    documents_truncated: int = 0
    packs_constructed: int = 0
    packs_discarded_by_quota: int = 0
    packs_emitted: int = 0
    padding_packs_emitted: int = 0
    real_tokens_emitted: int = 0
    padding_tokens_emitted: int = 0

    @property
    def packing_utilization(self) -> float:
        total = self.real_tokens_emitted + self.padding_tokens_emitted
        return self.real_tokens_emitted / total if total else 0.0


def best_fit_pack_documents(
    documents: Sequence[Document],
    *,
    max_seq_len: int,
    max_segments_per_example: int,
) -> tuple[tuple[PackedDocuments, ...], int]:
    """Pack a complete shard using deterministic best-fit decreasing.

    Returns:
        The packed bins and the number of documents truncated to
        ``max_seq_len``.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_segments_per_example <= 0:
        raise ValueError("max_segments_per_example must be positive")

    truncated = 0
    normalized: list[Document] = []
    for document in documents:
        if len(document) > max_seq_len:
            document = document.take(np.arange(max_seq_len))
            truncated += 1
        normalized.append(document)

    # Python's sort is stable. Callers can shuffle before this function to
    # randomize equal-length documents while retaining best-fit decreasing.
    normalized.sort(key=len, reverse=True)

    packs: list[PackedDocuments] = []
    by_remaining: list[tuple[int, int]] = []
    for document in normalized:
        position = bisect.bisect_left(by_remaining, (len(document), -1))
        if position == len(by_remaining):
            pack_id = len(packs)
            current = PackedDocuments()
            packs.append(current)
        else:
            _, pack_id = by_remaining.pop(position)
            current = packs[pack_id]

        current.add(document)
        if (
            current.used_tokens < max_seq_len
            and len(current.documents) < max_segments_per_example
        ):
            remaining = max_seq_len - current.used_tokens
            bisect.insort(by_remaining, (remaining, pack_id))

    return tuple(packs), truncated


def fixed_quota_pack_slots(
    packs: Sequence[PackedDocuments],
    *,
    examples_per_shard: int,
    rng: np.random.Generator,
) -> tuple[PackedDocuments | None, ...]:
    """Select or pad to exactly ``examples_per_shard`` slots.

    When a shard is overfull, packed bins are sampled uniformly without
    replacement. Every document therefore has the same conditional inclusion
    probability because each document belongs to exactly one bin.
    """
    if examples_per_shard <= 0:
        raise ValueError("examples_per_shard must be positive")

    if len(packs) >= examples_per_shard:
        selected = rng.permutation(len(packs))[:examples_per_shard]
        return tuple(packs[int(index)] for index in selected)

    slots: list[PackedDocuments | None] = [
        *packs,
        *(None for _ in range(examples_per_shard - len(packs))),
    ]
    order = rng.permutation(examples_per_shard)
    return tuple(slots[int(index)] for index in order)


class FixedQuotaShardDocumentDataset(AsyncDataset[Example], Generic[Example]):
    """Map every example index to a deterministic shard-local packed document.

    Each ``(epoch, shard)`` produces exactly ``examples_per_shard`` examples.
    A small in-memory shard cache avoids reconstruction during sequential
    prefetch, but cache contents have no semantic effect and need not be
    checkpointed.
    """

    def __init__(
        self,
        *,
        data_prefix: str,
        columns: Sequence[str],
        generate_document: DocumentGenerator,
        num_shards: int,
        total_shards: int,
        examples_per_shard: int,
        max_seq_len: int,
        example_builder: ExampleBuilder[Example],
        seed: int = 0,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
        max_segments_per_example: int = 64,
        shard_cache_size: int = 2,
    ):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if num_shards > total_shards:
            raise ValueError("num_shards cannot exceed total_shards")
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be positive")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        if not columns:
            raise ValueError("columns cannot be empty")

        self.data_prefix = data_prefix.rstrip("/")
        self.columns = tuple(columns)
        self.generate_document = generate_document
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.examples_per_shard = examples_per_shard
        self.max_seq_len = max_seq_len
        self.example_builder = example_builder
        self.seed = seed
        self.shard_name_template = shard_name_template
        self.max_segments_per_example = max_segments_per_example
        self.shard_cache_size = shard_cache_size

        self.stats = FixedQuotaShardStats()
        self._shard_orders: dict[int, tuple[int, ...]] = {}
        self._shard_cache: OrderedDict[
            tuple[int, int], tuple[PackedDocuments | None, ...]
        ] = OrderedDict()
        self._lock = asyncio.Lock()

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("FixedQuotaShardDocumentDataset is an infinite stream")

    async def getitem_async(self, index: int) -> Example:
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[Example]:
        if not indices:
            return []
        async with self._lock:
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

    def _get_batch_sync(self, indices: tuple[int, ...]) -> list[Example]:
        output: list[Example] = []
        for index in indices:
            epoch, shard_index, slot_index = self.location_for_index(index)
            slots = self._slots_for_shard(epoch, shard_index)
            current = slots[slot_index]
            if current is None:
                documents: tuple[Document, ...] = ()
                used_tokens = 0
                self.stats.padding_packs_emitted += 1
            else:
                documents = tuple(current.documents)
                used_tokens = current.used_tokens

            self.stats.packs_emitted += 1
            self.stats.real_tokens_emitted += used_tokens
            self.stats.padding_tokens_emitted += self.max_seq_len - used_tokens
            output.append(
                self.example_builder(
                    documents,
                    self.max_seq_len,
                    self.max_segments_per_example,
                )
            )
        return output

    def _shard_order(self, epoch: int) -> tuple[int, ...]:
        cached = self._shard_orders.get(epoch)
        if cached is not None:
            return cached
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        order = tuple(int(index) for index in rng.permutation(self.num_shards))
        self._shard_orders[epoch] = order
        return order

    def _slots_for_shard(
        self, epoch: int, shard_index: int
    ) -> tuple[PackedDocuments | None, ...]:
        key = (epoch, shard_index)
        cached = self._shard_cache.get(key)
        if cached is not None:
            self._shard_cache.move_to_end(key)
            return cached

        slots = self._construct_shard(epoch, shard_index)
        self._shard_cache[key] = slots
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return slots

    def _construct_shard(
        self, epoch: int, shard_index: int
    ) -> tuple[PackedDocuments | None, ...]:
        shard_path = self._shard_path(shard_index)
        with fsspec.open(shard_path, "rb") as source:
            table = pq.read_table(source, columns=list(self.columns))
        if table.num_rows == 0:
            raise ValueError(f"{shard_path} contains no rows")

        row_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, shard_index, 0])
        )
        documents: list[Document] = []
        rows = table.to_pylist()
        for row_index in row_rng.permutation(table.num_rows):
            document = self.generate_document(rows[int(row_index)])
            if document is None:
                self.stats.documents_dropped_by_generator += 1
                continue
            documents.append(document)

        packs, truncated = best_fit_pack_documents(
            documents,
            max_seq_len=self.max_seq_len,
            max_segments_per_example=self.max_segments_per_example,
        )
        slot_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, shard_index, 1])
        )
        slots = fixed_quota_pack_slots(
            packs,
            examples_per_shard=self.examples_per_shard,
            rng=slot_rng,
        )

        self.stats.shards_constructed += 1
        self.stats.documents_constructed += len(documents)
        self.stats.documents_truncated += truncated
        self.stats.packs_constructed += len(packs)
        self.stats.packs_discarded_by_quota += max(
            0, len(packs) - self.examples_per_shard
        )
        return slots

    def _shard_path(self, shard_index: int) -> str:
        shard_name = self.shard_name_template.format(
            shard_index=shard_index,
            total_shards=self.total_shards,
        )
        return f"{self.data_prefix}/{shard_name}"


def causal_lm_example_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> GrugLmExample:
    """Convert zero or more causal documents to one padded Levanter example."""
    if any(document.attention != AttentionLayout.CAUSAL for document in documents):
        raise ValueError("The causal LM adapter only accepts causal documents")
    if any(
        len(document.score_ranges) != 1
        or document.score_ranges[0].start != 0
        or document.score_ranges[0].stop != len(document)
        or not document.score_ranges[0].scored
        or document.score_ranges[0].has_explicit_targets
        for document in documents
    ):
        raise ValueError(
            "The causal LM adapter only accepts default next-token scoring"
        )

    if not documents:
        tokens = np.zeros(max_seq_len, dtype=np.int32)
        segment_ids = np.full(max_seq_len, -1, dtype=np.int32)
    else:
        packed = pack(documents, max_seq_len=max_seq_len)
        if packed.token_ids.shape[0] != 1:
            raise AssertionError(
                "Shard packing bin unexpectedly produced multiple rows"
            )
        tokens = packed.token_ids[0]
        segment_ids = packed.segment_ids[0]

    loss_weight = np.zeros(max_seq_len, dtype=np.float32)
    loss_weight[:-1] = (segment_ids[:-1] >= 0) & (segment_ids[:-1] == segment_ids[1:])
    with local_cpu_mesh():
        return GrugLmExample.causal(
            tokens=jnp.asarray(tokens),
            loss_weight=jnp.asarray(loss_weight),
            segment_ids=jnp.asarray(segment_ids),
            max_segments=max_segments_per_example + 1,
            block_cross_document_attention=True,
        )


__all__ = [
    "FixedQuotaShardDocumentDataset",
    "FixedQuotaShardStats",
    "PackedDocuments",
    "best_fit_pack_documents",
    "causal_lm_example_from_documents",
    "fixed_quota_pack_slots",
]
