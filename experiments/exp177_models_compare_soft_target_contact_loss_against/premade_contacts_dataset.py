# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contacts-v1 dataset adapters for exp177."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import asyncio
import multiprocessing as mp
import fsspec
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import haliax as hax
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.utils.jax_utils import local_cpu_mesh

from marinfold.document_structures.contacts_v1 import (
    ANALYZED_ROW_COLUMNS,
    CONTEXT_LENGTH,
    analyzed_from_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    RELATIVE_POSITION,
    causal_document_from_generation,
)
from marinfold_models.mp_queue_shard_dataset import MPQueueShardDocumentDataset
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    DOC_TYPE,
    END,
    NUM_POSITION_INDICES,
    POSITIONS,
    VOCABULARY,
)
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    POSITION_IDS,
    QUERY,
    AttentionLayout,
    Document,
    pack,
)
from marinfold_models.document_loss import (
    CompactContactDocumentBatch,
    LevanterDocumentBatch,
    SparseContactDocumentBatch,
    compact_contact_document_batch,
    levanter_document_batch,
)
from marinfold_models.shard_documents import (
    FixedQuotaShardDocumentDataset,
    MPFixedQuotaShardDocumentDataset,
    PackedDocuments,
    best_fit_pack_documents,
    causal_lm_example_from_documents,
    fixed_quota_pack_slots,
)


def _generation_from_row(row: Mapping[str, Any]):
    analyzed = analyzed_from_row(row)
    return build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        global_plddt=analyzed.global_plddt,
    )


def causal_contacts_v1_document_from_row(row: Mapping[str, Any]) -> Document | None:
    """Reconstruct the canonical serialized contacts-v1 training document."""
    generated = _generation_from_row(row)
    if generated is None:
        return None
    return causal_document_from_generation(generated)


def _causal_document_with_contacts(generated, contacts: Sequence[Any]) -> Document:
    parts = generated.document.split()
    try:
        prefix_end = parts.index(BEGIN_STRUCTURE.text) + 1
    except ValueError as exc:
        raise ValueError("Generated contacts-v1 document has no <begin_structure> token") from exc

    suffix: list[str] = []
    for contact in contacts:
        first, second = contact.pos_i, contact.pos_j
        if contact.flipped:
            first, second = second, first
        suffix.extend((CONTACT.text, POSITIONS[first].text, POSITIONS[second].text))
    suffix.append(END.text)

    tokens = (*parts[:prefix_end], *suffix)
    if len(tokens) + 1 > CONTEXT_LENGTH:
        raise ValueError(
            f"Augmented causal document needs {len(tokens) + 1} tokens including EOS, "
            f"exceeding max_seq_len={CONTEXT_LENGTH}"
        )
    return causal_document_from_generation(
        replace(generated, document=" ".join(tokens), num_tokens=len(tokens), contacts=tuple(contacts))
    )


def randomized_contact_order_document_from_row(
    row: Mapping[str, Any], *, seed: int, epoch: int, shard_index: int, row_index: int, augmentation_index: int
) -> Document | None:
    """Rebuild one causal CE document with a new contact order/orientation.

    The canonical contacts-v1 generator still selects the contact set and lays
    out the sequence prefix. This augmentation changes only the serialized
    structure suffix: contact statements are permuted and each endpoint order is
    independently coin-flipped from a deterministic augmentation seed.
    """
    generated = _generation_from_row(row)
    if generated is None:
        return None
    if not generated.contacts:
        return causal_document_from_generation(generated)

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, shard_index, row_index, augmentation_index]))
    order = rng.permutation(len(generated.contacts))
    contacts = tuple(
        replace(generated.contacts[int(contact_index)], flipped=bool(rng.integers(2))) for contact_index in order
    )
    return _causal_document_with_contacts(generated, contacts)


def soft_target_contacts_v1_document_from_row(row: Mapping[str, Any]) -> Document | None:
    """Build a compact block-causal contacts-v1 training document."""
    generated = _generation_from_row(row)
    if generated is None:
        return None

    sequence_tokens = [
        VOCABULARY.token(f"<{residue.resname}>") for residue in generated.residues
    ]
    prefix_tokens = [DOC_TYPE, BEGIN_SEQUENCE, *sequence_tokens, BEGIN_STRUCTURE]
    suffix_tokens = []
    for contact in generated.contacts:
        first, second = POSITIONS[contact.seq_i], POSITIONS[contact.seq_j]
        if contact.flipped:
            first, second = second, first
        suffix_tokens.extend((CONTACT, first, second))
    suffix_tokens.append(END)

    token_ids = (*prefix_tokens, *suffix_tokens)
    if len(token_ids) > CONTEXT_LENGTH:
        raise ValueError(
            f"Block-causal document needs {len(token_ids)} tokens, "
            f"exceeding max_seq_len={CONTEXT_LENGTH}"
        )

    prediction_start = len(prefix_tokens) - 1
    query = np.zeros(len(token_ids), dtype=np.bool_)
    query[prediction_start : prediction_start + len(suffix_tokens)] = True
    attention_blocks = (0,) * len(prefix_tokens) + tuple(
        range(1, len(suffix_tokens) + 1)
    )
    relative_positions = (
        (RELATIVE_POSITION.missing,) * 2
        + tuple(range(len(sequence_tokens)))
        + (RELATIVE_POSITION.missing,)
        + tuple(range(len(suffix_tokens)))
    )
    return Document(
        token_ids,
        {
            RELATIVE_POSITION: relative_positions,
            QUERY: query,
            ATTENTION_BLOCK: attention_blocks,
        },
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).unscored()


def document_batch_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> LevanterDocumentBatch:
    """Convert packed structured documents to one Levanter document batch item."""
    del max_segments_per_example
    packed = pack(documents, max_seq_len=max_seq_len)
    if packed.token_ids.shape[0] != 1:
        raise AssertionError("Shard packing bin unexpectedly produced multiple rows")
    return levanter_document_batch(packed, Pos=Axis("position", max_seq_len))


def compact_contact_batch_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> CompactContactDocumentBatch:
    """Convert one compact contacts-v1 document to a Levanter batch item."""
    del max_segments_per_example
    if len(documents) != 1:
        raise ValueError(
            f"Compact soft-target batches require one document, got {len(documents)}"
        )
    packed = pack(documents, max_seq_len=max_seq_len)
    if packed.token_ids.shape[0] != 1:
        raise AssertionError("Shard packing bin unexpectedly produced multiple rows")
    return compact_contact_document_batch(packed, Pos=Axis("position", max_seq_len))


class FixedQuotaPremadeContactsDataset(FixedQuotaShardDocumentDataset):
    """Build fixed-quota canonical contacts-v1 examples from premade contacts."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        shard_cache_size: int = 2,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=causal_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=causal_lm_example_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
            shard_name_template=shard_name_template,
        )


class MPFixedQuotaPremadeContactsDataset(MPFixedQuotaShardDocumentDataset):
    """Multiprocess fixed-quota canonical contacts-v1 next-token dataset."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        transform_workers: int = 4,
        prefetch_shards: int | None = None,
        shard_cache_size: int | None = None,
        mp_start_method: str = "spawn",
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=causal_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=causal_lm_example_from_documents,
            max_segments_per_example=max_segments_per_example,
            transform_workers=transform_workers,
            prefetch_shards=prefetch_shards,
            shard_cache_size=shard_cache_size,
            mp_start_method=mp_start_method,
            shard_name_template=shard_name_template,
        )


@dataclass(frozen=True)
class AugmentedContactOrderShardBuilder:
    """Build fixed-quota causal CE slots with contact-order augmentation."""

    data_prefix: str
    total_shards: int
    examples_per_shard: int
    max_seq_len: int
    seed: int = 0
    augmentations_per_row: int = 4
    shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet"
    max_segments_per_example: int = 64

    def __post_init__(self) -> None:
        if self.augmentations_per_row <= 0:
            raise ValueError("augmentations_per_row must be positive")

    def __call__(self, epoch: int, shard_index: int) -> tuple[PackedDocuments | None, ...]:
        shard_path = self._shard_path(shard_index)
        with fsspec.open(shard_path, "rb") as source:
            table = pq.read_table(source, columns=list(ANALYZED_ROW_COLUMNS))
        if table.num_rows == 0:
            raise ValueError(f"{shard_path} contains no rows")

        row_rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, shard_index, 0]))
        documents: list[Document] = []
        rows = table.to_pylist()
        for row_index in row_rng.permutation(table.num_rows):
            row = rows[int(row_index)]
            for augmentation_index in range(self.augmentations_per_row):
                document = randomized_contact_order_document_from_row(
                    row,
                    seed=self.seed,
                    epoch=epoch,
                    shard_index=shard_index,
                    row_index=int(row_index),
                    augmentation_index=augmentation_index,
                )
                if document is not None:
                    documents.append(document)

        packs, _ = best_fit_pack_documents(
            documents,
            max_seq_len=self.max_seq_len,
            max_segments_per_example=self.max_segments_per_example,
        )
        slot_rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, shard_index, 1]))
        return fixed_quota_pack_slots(
            packs,
            examples_per_shard=self.examples_per_shard,
            rng=slot_rng,
        )

    def _shard_path(self, shard_index: int) -> str:
        shard_name = self.shard_name_template.format(
            shard_index=shard_index,
            total_shards=self.total_shards,
        )
        return f"{self.data_prefix.rstrip('/')}/{shard_name}"


class MPAugmentedContactOrderPremadeContactsDataset(AsyncDataset):
    """Multiprocess next-token CE dataset with contact-order augmentation."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        augmentations_per_row: int = 4,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 64,
        transform_workers: int = 4,
        prefetch_shards: int | None = None,
        shard_cache_size: int | None = None,
        mp_start_method: str = "spawn",
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        builder = AugmentedContactOrderShardBuilder(
            data_prefix=data_prefix.rstrip("/"),
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            max_seq_len=max_seq_len,
            seed=seed,
            augmentations_per_row=augmentations_per_row,
            shard_name_template=shard_name_template,
            max_segments_per_example=max_segments_per_example,
        )
        self.max_seq_len = max_seq_len
        self.max_segments_per_example = max_segments_per_example
        self._slot_dataset = MPQueueShardDocumentDataset[PackedDocuments | None](
            build_shard=builder,
            num_shards=num_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            transform_workers=transform_workers,
            prefetch_shards=prefetch_shards,
            shard_cache_size=shard_cache_size,
            mp_start_method=mp_start_method,
        )

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("MPAugmentedContactOrderPremadeContactsDataset is an infinite stream")

    async def getitem_async(self, index: int):
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]):
        slots = await self._slot_dataset.get_batch(indices)
        return await asyncio.to_thread(self._examples_from_slots, tuple(slots))

    def location_for_index(self, index: int) -> tuple[int, int, int]:
        return self._slot_dataset.location_for_index(index)

    def start_workers(self) -> None:
        self._slot_dataset.start_workers()

    def close(self) -> None:
        if hasattr(self, "_slot_dataset"):
            self._slot_dataset.close()

    def __del__(self):
        self.close()

    def __deepcopy__(self, memo: dict[int, Any]) -> str:
        return repr(self)

    def _examples_from_slots(self, slots: tuple[PackedDocuments | None, ...]):
        output = []
        for current in slots:
            documents = () if current is None else tuple(current.documents)
            output.append(causal_lm_example_from_documents(documents, self.max_seq_len, self.max_segments_per_example))
        return output


class FixedQuotaSoftTargetContactsDataset(FixedQuotaShardDocumentDataset):
    """Build fixed-quota soft-target contacts-v1 examples from premade contacts."""

    def _construct_shard(
        self, epoch: int, shard_index: int
    ) -> tuple[PackedDocuments | None, ...]:
        slots = super()._construct_shard(epoch, shard_index)
        if all(slot is not None for slot in slots):
            return slots

        real_slots = tuple(slot for slot in slots if slot is not None)
        if not real_slots:
            raise ValueError(f"Shard {shard_index} yielded no soft-target packs")

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, shard_index, 2])
        )
        return tuple(
            slot if slot is not None else real_slots[int(rng.integers(len(real_slots)))]
            for slot in slots
        )

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 1,
        shard_cache_size: int = 2,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=soft_target_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=compact_contact_batch_from_documents,
            max_segments_per_example=max_segments_per_example,
            shard_cache_size=shard_cache_size,
            shard_name_template=shard_name_template,
        )


PRECOMPUTED_SOFT_TARGET_COLUMNS = (
    "token_ids",
    "position_ids",
    "segment_ids",
    "attention_blocks",
    "prediction_start",
    "contact_first_ids",
    "contact_second_ids",
    "contact_count",
    "target_position_count",
)


@dataclass(frozen=True)
class RawPrecomputedSoftTargetExample:
    """Pure-NumPy compact example built in prefetch workers."""

    token_ids: np.ndarray
    position_ids: np.ndarray
    segment_ids: np.ndarray
    attention_blocks: np.ndarray
    first_ids: np.ndarray
    second_ids: np.ndarray
    contact_count: int
    prediction_start: int
    target_position_count: int


@dataclass(frozen=True)
class PrecomputedSoftTargetDatasetConfig:
    """Serializable precomputed soft-target dataset parameters for workers."""

    data_prefix: str
    num_shards: int
    total_shards: int = 3338
    examples_per_shard: int = 2650
    max_seq_len: int = CONTEXT_LENGTH
    seed: int = 0
    shard_cache_size: int = 2
    shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet"


POSITION_TOKEN_START = int(POSITIONS[0])


_WORKER_PRECOMPUTED_DATASETS: dict[PrecomputedSoftTargetDatasetConfig, "PrecomputedSoftTargetContactsDataset"] = {}


def _precomputed_worker_dataset(config: PrecomputedSoftTargetDatasetConfig) -> "PrecomputedSoftTargetContactsDataset":
    dataset = _WORKER_PRECOMPUTED_DATASETS.get(config)
    if dataset is None:
        dataset = PrecomputedSoftTargetContactsDataset(
            data_prefix=config.data_prefix,
            num_shards=config.num_shards,
            total_shards=config.total_shards,
            examples_per_shard=config.examples_per_shard,
            max_seq_len=config.max_seq_len,
            seed=config.seed,
            shard_cache_size=config.shard_cache_size,
            shard_name_template=config.shard_name_template,
        )
        _WORKER_PRECOMPUTED_DATASETS[config] = dataset
    return dataset


def _build_precomputed_chunk(
    config: PrecomputedSoftTargetDatasetConfig,
    indices: tuple[int, ...],
) -> tuple[tuple[int, RawPrecomputedSoftTargetExample], ...]:
    dataset = _precomputed_worker_dataset(config)
    return tuple((index, dataset._raw_example_for_index(index)) for index in indices)


def _precomputed_worker_pid() -> int:
    return mp.current_process().pid


def _initialize_precomputed_worker() -> None:
    """Keep prefetch workers off accelerator backends.

    Worker processes only do parquet/Python/Numpy example construction. If JAX
    sees the H100s from those processes, each worker can initialize CUDA and
    reserve device memory before the trainer does useful work.
    """

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _sparse_second_endpoint_targets(
    raw: RawPrecomputedSoftTargetExample,
    *,
    max_contacts: int,
    max_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if raw.contact_count > max_contacts:
        raise ValueError(f"contact_count={raw.contact_count} exceeds sparse max_contacts={max_contacts}")
    first = raw.first_ids[: raw.contact_count].astype(np.int32)
    second = raw.second_ids[: raw.contact_count].astype(np.int32)
    local_first = first - POSITION_TOKEN_START
    local_second = second - POSITION_TOKEN_START
    if np.any((local_first < 0) | (local_first >= NUM_POSITION_INDICES)):
        raise ValueError("Sparse first endpoint ids are outside the contacts-v1 position token range")
    if np.any((local_second < 0) | (local_second >= NUM_POSITION_INDICES)):
        raise ValueError("Sparse second endpoint ids are outside the contacts-v1 position token range")

    adjacency: list[dict[int, int]] = [dict() for _ in range(NUM_POSITION_INDICES)]
    for a, b in zip(local_first.tolist(), local_second.tolist(), strict=True):
        token_a = int(a + POSITION_TOKEN_START)
        token_b = int(b + POSITION_TOKEN_START)
        adjacency[a][token_b] = adjacency[a].get(token_b, 0) + 1
        adjacency[b][token_a] = adjacency[b].get(token_a, 0) + 1

    padded_first = np.zeros(max_contacts, dtype=np.int32)
    padded_second = np.zeros(max_contacts, dtype=np.int32)
    neighbor_ids = np.zeros((max_contacts, max_degree), dtype=np.int32)
    neighbor_counts = np.zeros((max_contacts, max_degree), dtype=np.float32)
    neighbor_count = np.zeros(max_contacts, dtype=np.float32)
    padded_first[: raw.contact_count] = first
    padded_second[: raw.contact_count] = second

    def decrement(row: dict[int, int], token_id: int) -> None:
        count = row[token_id] - 1
        if count:
            row[token_id] = count
        else:
            del row[token_id]

    for c, (a, b) in enumerate(zip(local_first.tolist(), local_second.tolist(), strict=True)):
        row = adjacency[a]
        if len(row) > max_degree:
            raise ValueError(f"Sparse neighbor row degree {len(row)} exceeds max_degree={max_degree}")
        items = sorted(row.items())
        if items:
            ids, counts = zip(*items, strict=True)
            neighbor_ids[c, : len(items)] = np.asarray(ids, dtype=np.int32)
            neighbor_counts[c, : len(items)] = np.asarray(counts, dtype=np.float32)
            neighbor_count[c] = float(sum(counts))
        decrement(adjacency[a], int(b + POSITION_TOKEN_START))
        decrement(adjacency[b], int(a + POSITION_TOKEN_START))

    return padded_first, padded_second, neighbor_ids, neighbor_counts, neighbor_count


class PrecomputedSoftTargetContactsDataset(AsyncDataset[CompactContactDocumentBatch]):
    """Read exp177 precomputed compact soft-target rows from parquet shards."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        max_seq_len: int = CONTEXT_LENGTH,
        seed: int = 0,
        shard_cache_size: int = 2,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if num_shards > total_shards:
            raise ValueError("num_shards cannot exceed total_shards")
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be positive")
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.data_prefix = data_prefix.rstrip("/")
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.examples_per_shard = examples_per_shard
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.shard_cache_size = shard_cache_size
        self.shard_name_template = shard_name_template
        self._shard_orders: dict[int, tuple[int, ...]] = {}
        self._shard_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._shard_cache_order: list[tuple[int, int]] = []
        self._lock = asyncio.Lock()
        self._config = PrecomputedSoftTargetDatasetConfig(
            data_prefix=self.data_prefix,
            num_shards=self.num_shards,
            total_shards=self.total_shards,
            examples_per_shard=self.examples_per_shard,
            max_seq_len=self.max_seq_len,
            seed=self.seed,
            shard_cache_size=self.shard_cache_size,
            shard_name_template=self.shard_name_template,
        )

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("PrecomputedSoftTargetContactsDataset is an infinite stream")

    async def getitem_async(self, index: int) -> CompactContactDocumentBatch:
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[CompactContactDocumentBatch]:
        if not indices:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._get_batch_sync, tuple(indices))

    def _get_batch_sync(self, indices: tuple[int, ...]) -> list[CompactContactDocumentBatch]:
        return [self._example_for_index(index) for index in indices]

    def _example_for_index(self, index: int) -> CompactContactDocumentBatch:
        return self._batch_from_raw(self._raw_example_for_index(index))

    def _raw_example_for_index(self, index: int) -> RawPrecomputedSoftTargetExample:
        epoch, shard_index, slot_index = self.location_for_index(index)
        rows = self._rows_for_shard(epoch, shard_index)
        return self._raw_example_from_row(rows[slot_index])

    def location_for_index(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            raise IndexError("dataset indices must be non-negative")
        examples_per_epoch = self.num_shards * self.examples_per_shard
        epoch, index_within_epoch = divmod(index, examples_per_epoch)
        shard_position, slot_index = divmod(index_within_epoch, self.examples_per_shard)
        shard_index = self._shard_order(epoch)[shard_position]
        return epoch, shard_index, slot_index

    def _shard_order(self, epoch: int) -> tuple[int, ...]:
        cached = self._shard_orders.get(epoch)
        if cached is not None:
            return cached
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        order = tuple(int(index) for index in rng.permutation(self.num_shards))
        self._shard_orders[epoch] = order
        return order

    def _rows_for_shard(self, epoch: int, shard_index: int) -> list[dict[str, Any]]:
        key = (epoch, shard_index)
        cached = self._shard_cache.get(key)
        if cached is not None:
            self._shard_cache_order.remove(key)
            self._shard_cache_order.append(key)
            return cached

        shard_path = self._shard_path(shard_index)
        with fsspec.open(shard_path, "rb") as source:
            parquet_file = pq.ParquetFile(source)
            available_columns = set(parquet_file.schema_arrow.names)
            columns = [column for column in PRECOMPUTED_SOFT_TARGET_COLUMNS if column in available_columns]
            table = parquet_file.read(columns=columns)
        if table.num_rows != self.examples_per_shard:
            raise ValueError(
                f"{shard_path} contains {table.num_rows} rows; expected {self.examples_per_shard}"
            )
        rows = table.to_pylist()
        self._shard_cache[key] = rows
        self._shard_cache_order.append(key)
        while len(self._shard_cache_order) > self.shard_cache_size:
            old_key = self._shard_cache_order.pop(0)
            self._shard_cache.pop(old_key, None)
        return rows

    def _shard_path(self, shard_index: int) -> str:
        shard_name = self.shard_name_template.format(
            shard_index=shard_index,
            total_shards=self.total_shards,
        )
        return f"{self.data_prefix}/{shard_name}"

    def _raw_example_from_row(self, row: Mapping[str, Any]) -> RawPrecomputedSoftTargetExample:
        raw_token_ids = np.asarray(row["token_ids"], dtype=np.int32)
        if raw_token_ids.shape[0] > self.max_seq_len:
            raise ValueError(f"Precomputed row has {raw_token_ids.shape[0]} tokens, max_seq_len={self.max_seq_len}")
        raw_position_ids = np.asarray(row["position_ids"], dtype=np.int32)
        if raw_position_ids.shape[0] != raw_token_ids.shape[0]:
            raise ValueError("Precomputed row position_ids length does not match token_ids")

        prediction_start = int(row["prediction_start"])
        max_contacts = (self.max_seq_len - 2) // 3
        if "contact_count" in row:
            contact_count = int(row["contact_count"])
            contact_suffix = None
        else:
            suffix = raw_token_ids[prediction_start + 1 :]
            end_offsets = np.flatnonzero(suffix == int(END))
            if end_offsets.size == 0:
                raise ValueError("Precomputed compact row suffix has no END token")
            contact_suffix = suffix[: int(end_offsets[0])]
            if contact_suffix.size % 3 != 0:
                raise ValueError(f"Contact suffix before END is not triples: {contact_suffix.size} tokens")
            if np.any(contact_suffix[0::3] != int(CONTACT)):
                raise ValueError("Contact suffix triples do not start with CONTACT tokens")
            contact_count = contact_suffix.size // 3
        if contact_count > max_contacts:
            raise ValueError(f"Document has {contact_count} contacts, exceeding compact budget {max_contacts}")

        token_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        position_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        segment_ids = np.full(self.max_seq_len, -1, dtype=np.int32)
        attention_blocks = np.zeros(self.max_seq_len, dtype=np.int32)
        token_ids[: raw_token_ids.shape[0]] = raw_token_ids[: self.max_seq_len]
        position_ids[: raw_position_ids.shape[0]] = np.maximum(raw_position_ids[: self.max_seq_len], 0)
        if "segment_ids" in row:
            raw_segment_ids = np.asarray(row["segment_ids"], dtype=np.int32)
            segment_ids[: raw_segment_ids.shape[0]] = raw_segment_ids[: self.max_seq_len]
        else:
            segment_ids[: raw_token_ids.shape[0]] = 0
        if "attention_blocks" in row:
            raw_attention_blocks = np.asarray(row["attention_blocks"], dtype=np.int32)
            attention_blocks[: raw_attention_blocks.shape[0]] = raw_attention_blocks[: self.max_seq_len]
        elif prediction_start + 1 < raw_token_ids.shape[0]:
            attention_blocks[prediction_start + 1 : raw_token_ids.shape[0]] = np.arange(
                1,
                raw_token_ids.shape[0] - prediction_start,
                dtype=np.int32,
            )

        first_ids = np.zeros(max_contacts, dtype=np.int32)
        second_ids = np.zeros(max_contacts, dtype=np.int32)
        if "contact_first_ids" in row and "contact_second_ids" in row:
            raw_first_ids = np.asarray(row["contact_first_ids"], dtype=np.int32)
            raw_second_ids = np.asarray(row["contact_second_ids"], dtype=np.int32)
            first_ids[: min(raw_first_ids.shape[0], max_contacts)] = raw_first_ids[:max_contacts]
            second_ids[: min(raw_second_ids.shape[0], max_contacts)] = raw_second_ids[:max_contacts]
        else:
            if contact_suffix is None:
                raise ValueError("Precomputed row has contact_count but no contact id columns")
            first_ids[:contact_count] = contact_suffix[1::3]
            second_ids[:contact_count] = contact_suffix[2::3]
        return RawPrecomputedSoftTargetExample(
            token_ids=token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            attention_blocks=attention_blocks,
            first_ids=first_ids,
            second_ids=second_ids,
            contact_count=contact_count,
            prediction_start=prediction_start,
            target_position_count=int(row.get("target_position_count", 3 * contact_count + 1)),
        )

    def _batch_from_raw(self, raw: RawPrecomputedSoftTargetExample) -> CompactContactDocumentBatch:
        Pos = Axis("position", self.max_seq_len)
        axes = (Pos,)
        with local_cpu_mesh():
            tokens = hax.named(jnp.asarray(raw.token_ids), axes)
            segment_ids = hax.named(jnp.asarray(raw.segment_ids), axes)
            position_ids = hax.named(jnp.asarray(raw.position_ids), axes)
            attention_blocks = hax.named(jnp.asarray(raw.attention_blocks), axes)
            return CompactContactDocumentBatch(
                tokens=tokens,
                contact_first_ids=jnp.asarray(raw.first_ids),
                contact_second_ids=jnp.asarray(raw.second_ids),
                contact_count=jnp.asarray(raw.contact_count, dtype=jnp.int32),
                prediction_start=jnp.asarray(raw.prediction_start, dtype=jnp.int32),
                position_ids=position_ids,
                segment_ids=segment_ids,
                attention_blocks=attention_blocks,
                target_position_count=jnp.asarray(raw.target_position_count, dtype=jnp.int32),
                vocabulary=None,
            )


class SparsePrecomputedSoftTargetContactsDataset(PrecomputedSoftTargetContactsDataset):
    """Direct precomputed reader that builds sparse soft-target loss inputs."""

    def __init__(
        self,
        *,
        max_sparse_contacts: int = 2048,
        max_sparse_degree: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_sparse_contacts <= 0:
            raise ValueError("max_sparse_contacts must be positive")
        if max_sparse_degree <= 0:
            raise ValueError("max_sparse_degree must be positive")
        self.max_sparse_contacts = max_sparse_contacts
        self.max_sparse_degree = max_sparse_degree

    async def getitem_async(self, index: int) -> SparseContactDocumentBatch:
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[SparseContactDocumentBatch]:
        if not indices:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._get_batch_sync, tuple(indices))

    def _get_batch_sync(self, indices: tuple[int, ...]) -> list[SparseContactDocumentBatch]:
        return [self._example_for_index(index) for index in indices]

    def _example_for_index(self, index: int) -> SparseContactDocumentBatch:
        return self._batch_from_raw(self._raw_example_for_index(index))

    def _batch_from_raw(self, raw: RawPrecomputedSoftTargetExample) -> SparseContactDocumentBatch:
        first_ids, second_ids, neighbor_ids, neighbor_counts, neighbor_count = _sparse_second_endpoint_targets(
            raw,
            max_contacts=self.max_sparse_contacts,
            max_degree=self.max_sparse_degree,
        )
        Pos = Axis("position", self.max_seq_len)
        axes = (Pos,)
        with local_cpu_mesh():
            tokens = hax.named(jnp.asarray(raw.token_ids), axes)
            segment_ids = hax.named(jnp.asarray(raw.segment_ids), axes)
            position_ids = hax.named(jnp.asarray(raw.position_ids), axes)
            attention_blocks = hax.named(jnp.asarray(raw.attention_blocks), axes)
            return SparseContactDocumentBatch(
                tokens=tokens,
                contact_first_ids=jnp.asarray(first_ids),
                contact_second_ids=jnp.asarray(second_ids),
                second_neighbor_ids=jnp.asarray(neighbor_ids),
                second_neighbor_counts=jnp.asarray(neighbor_counts),
                second_neighbor_count=jnp.asarray(neighbor_count),
                contact_count=jnp.asarray(raw.contact_count, dtype=jnp.int32),
                prediction_start=jnp.asarray(raw.prediction_start, dtype=jnp.int32),
                position_ids=position_ids,
                segment_ids=segment_ids,
                attention_blocks=attention_blocks,
                target_position_count=jnp.asarray(raw.target_position_count, dtype=jnp.int32),
                vocabulary=None,
            )


class MPPrecomputedSoftTargetContactsDataset(AsyncDataset[CompactContactDocumentBatch]):
    """Multiprocess chunk-prefetch reader for precomputed soft-target rows.

    Levanter commonly asks a direct dataset for ``batch_size * prefetch_size``
    examples at once. The plain precomputed reader served those requests through
    one Python thread, so a bs16 smoke had to reconstruct 512 compact examples
    serially before the trainer could advance. This wrapper schedules bounded
    chunks in worker processes and caches individual examples in the parent,
    keeping the memory footprint much smaller than whole-shard caching.
    """

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        max_seq_len: int = CONTEXT_LENGTH,
        seed: int = 0,
        shard_cache_size: int = 2,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
        transform_workers: int = 8,
        prefetch_chunks: int | None = None,
        chunk_size: int = 64,
        example_cache_size: int | None = None,
        mp_start_method: str = "spawn",
    ):
        if transform_workers <= 0:
            raise ValueError("transform_workers must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.config = PrecomputedSoftTargetDatasetConfig(
            data_prefix=data_prefix.rstrip("/"),
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            max_seq_len=max_seq_len,
            seed=seed,
            shard_cache_size=shard_cache_size,
            shard_name_template=shard_name_template,
        )
        self.transform_workers = transform_workers
        self.prefetch_chunks = transform_workers if prefetch_chunks is None else prefetch_chunks
        self.chunk_size = chunk_size
        self.example_cache_size = max(
            chunk_size,
            chunk_size * (transform_workers + self.prefetch_chunks + 2)
            if example_cache_size is None
            else example_cache_size,
        )
        self.mp_start_method = mp_start_method
        self._executor: ProcessPoolExecutor | None = None
        self._lock = asyncio.Lock()
        self._example_cache: OrderedDict[int, CompactContactDocumentBatch] = OrderedDict()
        self._futures: dict[int, Future[tuple[tuple[int, RawPrecomputedSoftTargetExample], ...]]] = {}

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("MPPrecomputedSoftTargetContactsDataset is an infinite stream")

    async def getitem_async(self, index: int) -> CompactContactDocumentBatch:
        return (await self.get_batch([index]))[0]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[CompactContactDocumentBatch]:
        if not indices:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._get_batch_sync, tuple(int(index) for index in indices))

    def start_workers(self) -> None:
        pool = self._pool()
        futures = [pool.submit(_precomputed_worker_pid) for _ in range(self.transform_workers)]
        for future in futures:
            future.result()

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        self._futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        if hasattr(self, "_executor"):
            self.close()

    def __deepcopy__(self, memo: dict[int, Any]) -> str:
        return repr(self)

    def _get_batch_sync(self, indices: tuple[int, ...]) -> list[CompactContactDocumentBatch]:
        for chunk_start in self._chunk_starts_for_indices(indices):
            self._schedule_prefetch_window(chunk_start)
        return [self._example_for_index(index) for index in indices]

    def _example_for_index(self, index: int) -> CompactContactDocumentBatch:
        cached = self._example_cache.get(index)
        if cached is not None:
            self._example_cache.move_to_end(index)
            return cached
        chunk_start = self._chunk_start(index)
        self._ensure_scheduled(chunk_start)
        future = self._futures.pop(chunk_start)
        self._remember_examples(future.result())
        cached = self._example_cache.get(index)
        if cached is None:
            raise KeyError(f"Worker chunk {chunk_start} did not return requested index {index}")
        self._example_cache.move_to_end(index)
        return cached

    def _schedule_prefetch_window(self, chunk_start: int) -> None:
        self._ensure_scheduled(chunk_start)
        for offset in range(1, self.prefetch_chunks + 1):
            self._ensure_scheduled(chunk_start + offset * self.chunk_size)

    def _ensure_scheduled(self, chunk_start: int) -> None:
        if self._chunk_cached(chunk_start) or chunk_start in self._futures:
            return
        indices = tuple(range(chunk_start, chunk_start + self.chunk_size))
        self._futures[chunk_start] = self._pool().submit(_build_precomputed_chunk, self.config, indices)

    def _remember_examples(self, examples: tuple[tuple[int, RawPrecomputedSoftTargetExample], ...]) -> None:
        plain_dataset = PrecomputedSoftTargetContactsDataset(
            data_prefix=self.config.data_prefix,
            num_shards=self.config.num_shards,
            total_shards=self.config.total_shards,
            examples_per_shard=self.config.examples_per_shard,
            max_seq_len=self.config.max_seq_len,
            seed=self.config.seed,
            shard_cache_size=self.config.shard_cache_size,
            shard_name_template=self.config.shard_name_template,
        )
        for index, raw in examples:
            self._example_cache[index] = plain_dataset._batch_from_raw(raw)
            self._example_cache.move_to_end(index)
        while len(self._example_cache) > self.example_cache_size:
            self._example_cache.popitem(last=False)

    def _chunk_cached(self, chunk_start: int) -> bool:
        return all(index in self._example_cache for index in range(chunk_start, chunk_start + self.chunk_size))

    def _chunk_start(self, index: int) -> int:
        if index < 0:
            raise IndexError("dataset indices must be non-negative")
        return (index // self.chunk_size) * self.chunk_size

    def _chunk_starts_for_indices(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(self._chunk_start(index) for index in indices))

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            context = mp.get_context(self.mp_start_method)
            self._executor = ProcessPoolExecutor(
                max_workers=self.transform_workers,
                mp_context=context,
                initializer=_initialize_precomputed_worker,
            )
        return self._executor


class MPFixedQuotaSoftTargetContactsDataset(MPFixedQuotaShardDocumentDataset):
    """Multiprocess fixed-quota soft-target contacts-v1 dataset."""

    def __init__(
        self,
        *,
        data_prefix: str,
        num_shards: int,
        total_shards: int = 3338,
        examples_per_shard: int = 2650,
        seed: int = 0,
        max_seq_len: int = CONTEXT_LENGTH,
        max_segments_per_example: int = 1,
        transform_workers: int = 4,
        prefetch_shards: int | None = None,
        shard_cache_size: int | None = None,
        mp_start_method: str = "spawn",
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
    ):
        super().__init__(
            data_prefix=data_prefix,
            columns=ANALYZED_ROW_COLUMNS,
            generate_document=soft_target_contacts_v1_document_from_row,
            num_shards=num_shards,
            total_shards=total_shards,
            examples_per_shard=examples_per_shard,
            seed=seed,
            max_seq_len=max_seq_len,
            example_builder=compact_contact_batch_from_documents,
            max_segments_per_example=max_segments_per_example,
            transform_workers=transform_workers,
            prefetch_shards=prefetch_shards,
            shard_cache_size=shard_cache_size,
            mp_start_method=mp_start_method,
            shard_name_template=shard_name_template,
        )


__all__ = [
    "FixedQuotaPremadeContactsDataset",
    "FixedQuotaSoftTargetContactsDataset",
    "MPAugmentedContactOrderPremadeContactsDataset",
    "MPFixedQuotaPremadeContactsDataset",
    "MPFixedQuotaSoftTargetContactsDataset",
    "MPPrecomputedSoftTargetContactsDataset",
    "PrecomputedSoftTargetContactsDataset",
    "SparsePrecomputedSoftTargetContactsDataset",
    "causal_contacts_v1_document_from_row",
    "compact_contact_batch_from_documents",
    "soft_target_contacts_v1_document_from_row",
]
