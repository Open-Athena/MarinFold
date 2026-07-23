# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable streaming construction and online packing for training documents."""

import asyncio
import bisect
import hashlib
import json
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Generic, TypeVar
from uuid import uuid4

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.utils.jax_utils import local_cpu_mesh

from marinfold.document_structures.documents import (
    AttentionLayout,
    Document,
    next_token_score,
    pack,
)


Example = TypeVar("Example")
DocumentGenerator = Callable[[Mapping[str, Any]], Document | None]
ExampleBuilder = Callable[[tuple[Document, ...], int, int], Example]


@dataclass(frozen=True)
class RowReference:
    """Stable location of a source row used to regenerate a document."""

    epoch: int
    shard_index: int
    row_index: int


@dataclass(frozen=True)
class SourcedDocument:
    """One generated document and the source row that can regenerate it."""

    source: RowReference
    document: Document


@dataclass
class PackedDocuments:
    """Mutable bin used by the online best-fit packer."""

    documents: list[SourcedDocument] = field(default_factory=list)
    used_tokens: int = 0

    def add(self, document: SourcedDocument) -> None:
        self.documents.append(document)
        self.used_tokens += len(document.document)


@dataclass
class StreamingDocumentStats:
    """Operational counters for inspecting a streaming document loader."""

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


class OnlineBestFitDocumentPacker:
    """Bounded best-fit packer that carries partial bins across loader calls."""

    def __init__(
        self,
        *,
        max_seq_len: int,
        max_segments_per_example: int,
        min_fill_fraction: float,
        max_open_packs: int,
        stats: StreamingDocumentStats,
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

    def add(self, sourced: SourcedDocument) -> None:
        document = sourced.document
        if len(document) >= self.max_seq_len:
            if len(document) > self.max_seq_len:
                self.stats.truncated_documents += 1
                document = document.take(np.arange(self.max_seq_len))
                sourced = SourcedDocument(source=sourced.source, document=document)
            self._ready.append(
                PackedDocuments(documents=[sourced], used_tokens=len(document))
            )
            return

        position = bisect.bisect_left(
            self._by_remaining,
            (len(document), -1),
        )
        if position == len(self._by_remaining):
            pack_id = self._next_pack_id
            self._next_pack_id += 1
            current = PackedDocuments()
            self._packs[pack_id] = current
        else:
            _, pack_id = self._by_remaining.pop(position)
            current = self._packs[pack_id]

        current.add(sourced)
        should_emit = (
            current.used_tokens >= self.min_fill_tokens
            or len(current.documents) >= self.max_segments_per_example
        )
        if should_emit:
            self._ready.append(self._packs.pop(pack_id))
        else:
            remaining = self.max_seq_len - current.used_tokens
            bisect.insort(self._by_remaining, (remaining, pack_id))

        while len(self._packs) > self.max_open_packs:
            _, fullest_pack_id = self._by_remaining.pop(0)
            self._ready.append(self._packs.pop(fullest_pack_id))

    def pop_ready(self) -> PackedDocuments | None:
        return self._ready.popleft() if self._ready else None

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible packing state using source references."""
        return {
            "next_pack_id": self._next_pack_id,
            "packs": {
                str(pack_id): _packed_references(current)
                for pack_id, current in self._packs.items()
            },
            "by_remaining": [list(item) for item in self._by_remaining],
            "ready": [_packed_references(current) for current in self._ready],
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        resolve: Callable[[RowReference], Document],
    ) -> None:
        """Restore packing bins, regenerating documents from source rows."""
        self._next_pack_id = int(state["next_pack_id"])
        self._packs = {
            int(pack_id): _resolve_packed_references(raw, resolve)
            for pack_id, raw in state["packs"].items()
        }
        self._by_remaining = [
            (int(remaining), int(pack_id))
            for remaining, pack_id in state["by_remaining"]
        ]
        self._ready = deque(
            _resolve_packed_references(raw, resolve) for raw in state["ready"]
        )


class ShuffledParquetRowStream:
    """Read shuffled row blocks from process-disjoint Parquet shards."""

    def __init__(
        self,
        *,
        data_prefix: str,
        columns: Sequence[str],
        num_shards: int,
        total_shards: int,
        shard_name_template: str,
        seed: int,
        process_index: int,
        process_count: int,
        row_block_size: int,
        stats: StreamingDocumentStats,
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
        if not columns:
            raise ValueError("columns cannot be empty")

        self.data_prefix = data_prefix.rstrip("/")
        self.columns = tuple(columns)
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.shard_name_template = shard_name_template
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
        self._current_block_index: int | None = None
        self._current_row_indices: tuple[int, ...] = ()
        self._row_cursor = 0
        self._reference_tables: OrderedDict[int, pa.Table] = OrderedDict()
        self._reset_epoch()

    def next_row(self) -> tuple[RowReference, dict[str, Any]]:
        while self._row_cursor >= len(self._current_row_indices):
            if (
                self._current_table is None
                or self._row_block_cursor >= len(self._row_blocks)
            ):
                self._load_next_shard()
            self._load_next_row_block()

        assert self._current_table is not None
        assert self._current_shard_index is not None
        row_index = self._current_row_indices[self._row_cursor]
        self._row_cursor += 1
        row = self._current_table.slice(row_index, 1).to_pylist()[0]
        return (
            RowReference(
                epoch=self.epoch,
                shard_index=self._current_shard_index,
                row_index=row_index,
            ),
            row,
        )

    def row_for_reference(self, reference: RowReference) -> dict[str, Any]:
        """Read a referenced row without changing the stream cursor."""
        if self._current_shard_index == reference.shard_index:
            assert self._current_table is not None
            table = self._current_table
        else:
            table = self._reference_tables.get(reference.shard_index)
            if table is None:
                table = self._read_shard(reference.shard_index, count_stat=False)
                self._reference_tables[reference.shard_index] = table
                while len(self._reference_tables) > 4:
                    self._reference_tables.popitem(last=False)
            else:
                self._reference_tables.move_to_end(reference.shard_index)
        if not 0 <= reference.row_index < table.num_rows:
            raise IndexError(
                f"Row {reference.row_index} is outside shard "
                f"{reference.shard_index} with {table.num_rows} rows"
            )
        return table.slice(reference.row_index, 1).to_pylist()[0]

    def state_dict(self) -> dict[str, Any]:
        """Return the compact cursor state needed for exact reconstruction."""
        return {
            "epoch": self.epoch,
            "shard_cursor": self._shard_cursor,
            "current_shard_index": self._current_shard_index,
            "row_block_cursor": self._row_block_cursor,
            "current_block_index": self._current_block_index,
            "row_cursor": self._row_cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor and deterministically rebuild its shuffled block."""
        self.epoch = int(state["epoch"])
        self._reset_epoch()
        self._shard_cursor = int(state["shard_cursor"])
        current_shard = state["current_shard_index"]
        if current_shard is None:
            return

        self._install_shard(int(current_shard), count_stat=False)
        self._row_block_cursor = int(state["row_block_cursor"])
        current_block = state["current_block_index"]
        if current_block is None:
            self._current_block_index = None
            self._current_row_indices = ()
            self._row_cursor = 0
            return
        self._install_row_block(int(current_block))
        self._row_cursor = int(state["row_cursor"])

    def _reset_epoch(self) -> None:
        self._local_shards = shard_indices_for_process(
            num_shards=self.num_shards,
            seed=self.seed,
            epoch=self.epoch,
            process_index=self.process_index,
            process_count=self.process_count,
        )
        self._shard_cursor = 0
        self._current_table = None
        self._current_shard_index = None
        self._row_blocks = ()
        self._row_block_cursor = 0
        self._current_block_index = None
        self._current_row_indices = ()
        self._row_cursor = 0

    def _load_next_shard(self) -> None:
        if self._shard_cursor >= len(self._local_shards):
            self.epoch += 1
            self._reset_epoch()

        shard_index = self._local_shards[self._shard_cursor]
        self._shard_cursor += 1
        self._install_shard(shard_index, count_stat=True)

    def _install_shard(self, shard_index: int, *, count_stat: bool) -> None:
        table = self._read_shard(shard_index, count_stat=count_stat)
        block_count = (table.num_rows + self.row_block_size - 1) // self.row_block_size
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, shard_index])
        )
        self._current_table = table
        self._current_shard_index = shard_index
        self._row_blocks = tuple(int(index) for index in rng.permutation(block_count))
        self._row_block_cursor = 0
        self._current_block_index = None
        self._current_row_indices = ()
        self._row_cursor = 0

    def _read_shard(self, shard_index: int, *, count_stat: bool) -> pa.Table:
        shard_name = self.shard_name_template.format(
            shard_index=shard_index,
            total_shards=self.total_shards,
        )
        shard_path = str(PurePosixPath(self.data_prefix) / shard_name)
        with fsspec.open(shard_path, "rb") as source:
            table = pq.read_table(source, columns=list(self.columns))
        if table.num_rows == 0:
            raise ValueError(f"{shard_path} contains no rows")
        if count_stat:
            self.stats.shards_read += 1
        return table

    def _load_next_row_block(self) -> None:
        block_index = self._row_blocks[self._row_block_cursor]
        self._row_block_cursor += 1
        self._install_row_block(block_index)

    def _install_row_block(self, block_index: int) -> None:
        assert self._current_table is not None
        assert self._current_shard_index is not None
        start = block_index * self.row_block_size
        stop = min(start + self.row_block_size, self._current_table.num_rows)
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch, self._current_shard_index, block_index]
            )
        )
        self._current_block_index = block_index
        self._current_row_indices = tuple(
            int(index)
            for index in rng.permutation(np.arange(start, stop, dtype=np.int64))
        )
        self._row_cursor = 0


class StreamingDocumentDataset(AsyncDataset[Example], Generic[Example]):
    """Construct and online-pack deterministic documents from Parquet rows.

    The dataset is intentionally stateful: requested index values identify
    optimizer-step boundaries for checkpoint snapshots, while examples are
    produced from the stream in request order. Outer dataset shuffling must be
    disabled.

    Checkpoints contain only source references for documents held in packing
    bins. ``generate_document`` must therefore be deterministic for a source
    row and ``generator_id`` must change whenever its behavior changes.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        *,
        data_prefix: str,
        columns: Sequence[str],
        generate_document: DocumentGenerator,
        generator_id: str,
        num_shards: int,
        total_shards: int,
        max_seq_len: int,
        example_builder: ExampleBuilder[Example],
        seed: int = 0,
        shard_name_template: str = "shard-{shard_index:05d}-of-{total_shards:05d}.parquet",
        max_segments_per_example: int = 64,
        min_fill_fraction: float = 0.99,
        max_open_packs: int = 256,
        row_block_size: int = 256,
        process_index: int | None = None,
        process_count: int | None = None,
        global_batch_size: int | None = None,
        step_state_history: int = 256,
    ):
        if not generator_id:
            raise ValueError("generator_id cannot be empty")
        if global_batch_size is not None and global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive or None")
        if step_state_history <= 0:
            raise ValueError("step_state_history must be positive")

        self.data_prefix = data_prefix
        self.columns = tuple(columns)
        self.generate_document = generate_document
        self.generator_id = generator_id
        self.num_shards = num_shards
        self.total_shards = total_shards
        self.max_seq_len = max_seq_len
        self.example_builder = example_builder
        self.seed = seed
        self.shard_name_template = shard_name_template
        self.max_segments_per_example = max_segments_per_example
        self.min_fill_fraction = min_fill_fraction
        self.max_open_packs = max_open_packs
        self.row_block_size = row_block_size
        self.process_index = process_index
        self.process_count = process_count
        self.global_batch_size = global_batch_size
        self.step_state_history = step_state_history

        self.stats = StreamingDocumentStats()
        self._source: ShuffledParquetRowStream | None = None
        self._packer: OnlineBestFitDocumentPacker | None = None
        self._peeked_pack: PackedDocuments | None = None
        self._step_states: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    def is_finite(self) -> bool:
        return False

    async def async_len(self) -> int:
        raise ValueError("StreamingDocumentDataset is an infinite stream")

    async def getitem_async(self, index: int) -> Example:
        del index
        async with self._lock:
            if self._peeked_pack is None:
                self._peeked_pack = await asyncio.to_thread(self._next_pack_sync)
            return self._example_from_pack(self._peeked_pack, count_stats=False)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[Example]:
        if not indices:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._next_batch_sync, tuple(indices))

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible exact stream and packing state."""
        self._ensure_stream()
        assert self._source is not None
        assert self._packer is not None
        return {
            "version": self.CHECKPOINT_VERSION,
            "config_fingerprint": self._config_fingerprint(),
            "stats": asdict(self.stats),
            "source": self._source.state_dict(),
            "packer": self._packer.state_dict(),
            "peeked": (
                None
                if self._peeked_pack is None
                else _packed_references(self._peeked_pack)
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an exact stream state and reject incompatible configuration."""
        if int(state["version"]) != self.CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported streaming document checkpoint version {state['version']}"
            )
        if state["config_fingerprint"] != self._config_fingerprint():
            raise ValueError(
                "Streaming document checkpoint configuration does not match this dataset"
            )

        restored_stats = StreamingDocumentStats(**state["stats"])
        self.stats = restored_stats
        self._source = None
        self._packer = None
        self._ensure_stream()
        assert self._source is not None
        assert self._packer is not None
        self._source.load_state_dict(state["source"])
        self._packer.load_state_dict(state["packer"], resolve=self._resolve_document)
        raw_peeked = state["peeked"]
        self._peeked_pack = (
            None
            if raw_peeked is None
            else _resolve_packed_references(raw_peeked, self._resolve_document)
        )
        self._step_states.clear()

    def checkpoint_state_for_step(self, step: int) -> dict[str, Any]:
        """Return the state at the start of a prefetched optimizer step."""
        try:
            return self._step_states[step]
        except KeyError:
            available = tuple(self._step_states)
            raise KeyError(
                f"No state retained for step {step}; available steps are {available}"
            ) from None

    def save_checkpoint(self, path: str, *, step: int | None = None) -> None:
        """Atomically save current state or a retained optimizer-step snapshot."""
        state = (
            self.state_dict()
            if step is None
            else self.checkpoint_state_for_step(step)
        )
        filesystem, plain_path = fsspec.core.url_to_fs(path)
        temporary_path = f"{plain_path}.tmp-{uuid4().hex}"
        parent = str(PurePosixPath(plain_path).parent)
        if parent not in ("", "."):
            filesystem.makedirs(parent, exist_ok=True)
        with filesystem.open(temporary_path, "w") as destination:
            json.dump(state, destination, separators=(",", ":"), sort_keys=True)
        filesystem.mv(temporary_path, plain_path)

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint written by :meth:`save_checkpoint`."""
        with fsspec.open(path, "r") as source:
            state = json.load(source)
        self.load_state_dict(state)

    def save_model_checkpoint_sidecar(
        self,
        model_checkpoint_path: str,
        *,
        step: int,
    ) -> str:
        """Save this process's loader state inside a model checkpoint."""
        path = streaming_document_state_path(model_checkpoint_path)
        self.save_checkpoint(path, step=step)
        return path

    def load_model_checkpoint_sidecar(self, model_checkpoint_path: str) -> str:
        """Restore this process's loader state from a model checkpoint."""
        path = streaming_document_state_path(model_checkpoint_path)
        self.load_checkpoint(path)
        return path

    def _next_batch_sync(self, indices: tuple[int, ...]) -> list[Example]:
        output: list[Example] = []
        for index in indices:
            self._capture_step_state(index)
            if self._peeked_pack is not None:
                current = self._peeked_pack
                self._peeked_pack = None
            else:
                current = self._next_pack_sync()
            output.append(self._example_from_pack(current, count_stats=True))
        return output

    def _capture_step_state(self, index: int) -> None:
        if self.global_batch_size is None:
            return
        step = index // self.global_batch_size
        if step in self._step_states:
            return
        self._step_states[step] = self.state_dict()
        while len(self._step_states) > self.step_state_history:
            self._step_states.popitem(last=False)

    def _next_pack_sync(self) -> PackedDocuments:
        self._ensure_stream()
        assert self._packer is not None
        current = self._packer.pop_ready()
        while current is None:
            self._packer.add(self._next_document())
            current = self._packer.pop_ready()
        return current

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
        self._source = ShuffledParquetRowStream(
            data_prefix=self.data_prefix,
            columns=self.columns,
            num_shards=self.num_shards,
            total_shards=self.total_shards,
            shard_name_template=self.shard_name_template,
            seed=self.seed,
            process_index=process_index,
            process_count=process_count,
            row_block_size=self.row_block_size,
            stats=self.stats,
        )
        self._packer = OnlineBestFitDocumentPacker(
            max_seq_len=self.max_seq_len,
            max_segments_per_example=self.max_segments_per_example,
            min_fill_fraction=self.min_fill_fraction,
            max_open_packs=self.max_open_packs,
            stats=self.stats,
        )

    def _next_document(self) -> SourcedDocument:
        assert self._source is not None
        while True:
            reference, row = self._source.next_row()
            document = self.generate_document(row)
            if document is None:
                self.stats.documents_dropped += 1
                continue
            self.stats.documents_constructed += 1
            return SourcedDocument(source=reference, document=document)

    def _resolve_document(self, reference: RowReference) -> Document:
        assert self._source is not None
        row = self._source.row_for_reference(reference)
        document = self.generate_document(row)
        if document is None:
            raise ValueError(
                "Document generator dropped a row referenced by a checkpoint; "
                "the generator is not deterministic"
            )
        return document

    def _example_from_pack(
        self,
        current: PackedDocuments,
        *,
        count_stats: bool,
    ) -> Example:
        if count_stats:
            self.stats.packs_emitted += 1
            self.stats.real_tokens_emitted += current.used_tokens
            self.stats.padding_tokens_emitted += self.max_seq_len - current.used_tokens
        return self.example_builder(
            tuple(item.document for item in current.documents),
            self.max_seq_len,
            self.max_segments_per_example,
        )

    def _config_fingerprint(self) -> str:
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
        config = {
            "columns": self.columns,
            "data_prefix": self.data_prefix,
            "generator_id": self.generator_id,
            "max_open_packs": self.max_open_packs,
            "max_segments_per_example": self.max_segments_per_example,
            "max_seq_len": self.max_seq_len,
            "min_fill_fraction": self.min_fill_fraction,
            "num_shards": self.num_shards,
            "process_count": process_count,
            "process_index": process_index,
            "row_block_size": self.row_block_size,
            "seed": self.seed,
            "shard_name_template": self.shard_name_template,
            "total_shards": self.total_shards,
        }
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def causal_lm_example_from_documents(
    documents: tuple[Document, ...],
    max_seq_len: int,
    max_segments_per_example: int,
) -> GrugLmExample:
    """Pack ordinary causal documents into one Levanter LM example."""
    if any(document.attention != AttentionLayout.CAUSAL for document in documents):
        raise ValueError("The causal LM adapter only accepts causal documents")
    if any(
        score_range.scorer != next_token_score or score_range.target_ids is not None
        for document in documents
        for score_range in document.score_ranges
    ):
        raise ValueError(
            "The causal LM adapter only accepts default next-token scoring"
        )

    packed = pack(documents, max_seq_len=max_seq_len)
    if packed.token_ids.shape[0] != 1:
        raise AssertionError("Online packing bin unexpectedly produced multiple rows")
    tokens = packed.token_ids[0]
    segment_ids = packed.segment_ids[0]
    loss_weight = np.zeros(max_seq_len, dtype=np.float32)
    loss_weight[:-1] = (
        (segment_ids[:-1] >= 0) & (segment_ids[:-1] == segment_ids[1:])
    )
    with local_cpu_mesh():
        return GrugLmExample.causal(
            tokens=jnp.asarray(tokens),
            loss_weight=jnp.asarray(loss_weight),
            segment_ids=jnp.asarray(segment_ids),
            max_segments=max_segments_per_example + 1,
            block_cross_document_attention=True,
        )


def streaming_document_state_path(
    model_checkpoint_path: str,
    *,
    process_index: int | None = None,
) -> str:
    """Canonical per-process loader-state sidecar within a model checkpoint."""
    resolved_process_index = (
        jax.process_index() if process_index is None else process_index
    )
    if resolved_process_index < 0:
        raise ValueError("process_index must be non-negative")
    return (
        f"{model_checkpoint_path.rstrip('/')}/input/"
        f"streaming-documents-process-{resolved_process_index:05d}.json"
    )


def _packed_references(current: PackedDocuments) -> dict[str, Any]:
    return {
        "documents": [
            {
                "source": asdict(item.source),
                "length": len(item.document),
            }
            for item in current.documents
        ],
        "used_tokens": current.used_tokens,
    }


def _resolve_packed_references(
    raw: Mapping[str, Any],
    resolve: Callable[[RowReference], Document],
) -> PackedDocuments:
    documents: list[SourcedDocument] = []
    for item in raw["documents"]:
        reference = RowReference(**item["source"])
        document = resolve(reference)
        stored_length = int(item["length"])
        if stored_length > len(document):
            raise ValueError(
                "Regenerated document is shorter than its checkpointed length; "
                "the generator changed without changing generator_id"
            )
        if stored_length < len(document):
            document = document.take(np.arange(stored_length))
        documents.append(SourcedDocument(source=reference, document=document))
    used_tokens = sum(len(item.document) for item in documents)
    if used_tokens != int(raw["used_tokens"]):
        raise ValueError(
            "Regenerated document lengths do not match the checkpoint; "
            "the generator changed without changing generator_id"
        )
    return PackedDocuments(documents=documents, used_tokens=used_tokens)


__all__ = [
    "OnlineBestFitDocumentPacker",
    "PackedDocuments",
    "RowReference",
    "ShuffledParquetRowStream",
    "SourcedDocument",
    "StreamingDocumentDataset",
    "StreamingDocumentStats",
    "causal_lm_example_from_documents",
    "shard_indices_for_process",
    "streaming_document_state_path",
]
