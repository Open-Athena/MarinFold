# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.mixture import MixtureDataset

from marinfold.document_structures.contacts_v1 import (
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyzed_to_row,
)
from marinfold.document_structures.documents import causal_training_document
from marinfold_models.streaming_documents import (
    OnlineBestFitDocumentPacker,
    RowReference,
    SourcedDocument,
    StreamingDocumentStats,
    shard_indices_for_process,
)

from premade_contacts_dataset import (
    StreamingPremadeContactsDataset,
)


def _analyzed(entry_id: str, length: int) -> AnalyzedStructure:
    residues = tuple(
        ResidueInfo(seq_index=i, resname="ALA", resnum=i + 1, chain="A")
        for i in range(length)
    )
    contacts = tuple(
        RawContact(seq_i=i, seq_j=i + 6, degree=0.5)
        for i in range(max(0, length - 6))
    )
    return AnalyzedStructure(
        entry_id=entry_id,
        residues=residues,
        contacts=contacts,
        global_plddt=90.0,
        source_path=Path("<test>"),
    )


def _write_shards(tmp_path: Path, *, num_shards: int = 4, rows_per_shard: int = 80) -> None:
    for shard_index in range(num_shards):
        rows = [
            analyzed_to_row(
                _analyzed(
                    f"s{shard_index}-r{row_index}",
                    8 + (row_index * 7 + shard_index) % 65,
                )
            )
            for row_index in range(rows_per_shard)
        ]
        path = tmp_path / f"shard-{shard_index:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)


def _dataset(tmp_path: Path, *, process_index: int = 0, process_count: int = 1):
    return StreamingPremadeContactsDataset(
        data_prefix=str(tmp_path),
        num_shards=4,
        total_shards=4,
        seed=17,
        max_seq_len=512,
        min_fill_fraction=0.95,
        max_open_packs=16,
        row_block_size=16,
        process_index=process_index,
        process_count=process_count,
    )


def test_shard_partition_is_disjoint_and_complete():
    assignments = [
        set(
            shard_indices_for_process(
                num_shards=17,
                seed=3,
                epoch=2,
                process_index=process_index,
                process_count=4,
            )
        )
        for process_index in range(4)
    ]
    assert set.union(*assignments) == set(range(17))
    assert sum(len(assignment) for assignment in assignments) == 17
    for left_index, left in enumerate(assignments):
        for right in assignments[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_online_packer_carries_partial_bins():
    stats = StreamingDocumentStats()
    packer = OnlineBestFitDocumentPacker(
        max_seq_len=100,
        max_segments_per_example=8,
        min_fill_fraction=0.95,
        max_open_packs=2,
        stats=stats,
    )
    for index, length in enumerate((61, 61, 39)):
        packer.add(
            SourcedDocument(
                source=RowReference(epoch=0, shard_index=0, row_index=index),
                document=causal_training_document(
                    np.arange(length, dtype=np.int32)
                ),
            )
        )

    packed = packer.pop_ready()
    assert packed is not None
    assert packed.used_tokens == 100
    assert packer.open_pack_count == 1


def test_streaming_dataset_is_deterministic_and_well_packed(tmp_path: Path):
    _write_shards(tmp_path)
    first = _dataset(tmp_path)
    second = _dataset(tmp_path)

    first_examples = asyncio.run(first.get_batch(range(24)))
    second_examples = asyncio.run(second.get_batch(range(24)))
    assert len(first_examples) == len(second_examples) == 24
    for left, right in zip(first_examples, second_examples, strict=True):
        np.testing.assert_array_equal(np.asarray(left.tokens), np.asarray(right.tokens))

    assert first.stats.packing_utilization >= 0.90
    assert first.stats.documents_constructed > len(first_examples)
    assert first.stats.shards_read >= 1


def test_getitem_peek_is_reused_by_first_batch(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path)
    peek = asyncio.run(dataset.getitem_async(0))
    batch = asyncio.run(dataset.get_batch([100, 101]))
    np.testing.assert_array_equal(np.asarray(peek.tokens), np.asarray(batch[0].tokens))
    assert dataset.stats.packs_emitted == 2


def test_single_component_mixture_preserves_streaming_peek(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path)
    mixture = MixtureDataset(
        datasets={"contacts": dataset},
        weights={"contacts": 1.0},
        block_size=1,
        key=0,
    )
    peek = asyncio.run(mixture.getitem_async(0))
    batch = asyncio.run(mixture.get_batch([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(peek.tokens), np.asarray(batch[0].tokens))
    assert dataset.stats.packs_emitted == 3


def test_processes_read_different_streams(tmp_path: Path):
    _write_shards(tmp_path)
    process_zero = _dataset(tmp_path, process_index=0, process_count=2)
    process_one = _dataset(tmp_path, process_index=1, process_count=2)
    zero = asyncio.run(process_zero.get_batch(range(4)))
    one = asyncio.run(process_one.get_batch(range(4)))
    assert any(
        not np.array_equal(np.asarray(left.tokens), np.asarray(right.tokens))
        for left, right in zip(zero, one, strict=True)
    )


def test_checkpoint_round_trip_restores_exact_next_pack(tmp_path: Path):
    _write_shards(tmp_path)
    original = _dataset(tmp_path)
    asyncio.run(original.get_batch(range(7)))
    checkpoint = tmp_path / "loader-state.json"
    original.save_checkpoint(str(checkpoint))
    expected = asyncio.run(original.get_batch(range(7, 17)))

    restored = _dataset(tmp_path)
    restored.load_checkpoint(str(checkpoint))
    actual = asyncio.run(restored.get_batch(range(7, 17)))

    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(np.asarray(left.tokens), np.asarray(right.tokens))
        np.testing.assert_array_equal(
            np.asarray(left.loss_weight), np.asarray(right.loss_weight)
        )


def test_checkpoint_uses_optimizer_step_snapshot_before_prefetch(tmp_path: Path):
    _write_shards(tmp_path)
    original = StreamingPremadeContactsDataset(
        data_prefix=str(tmp_path),
        num_shards=4,
        total_shards=4,
        seed=17,
        max_seq_len=512,
        min_fill_fraction=0.95,
        max_open_packs=16,
        row_block_size=16,
        process_index=0,
        process_count=1,
        global_batch_size=6,
    )
    prefetched = asyncio.run(original.get_batch(range(18)))
    checkpoint = tmp_path / "loader-step-1.json"
    original.save_checkpoint(str(checkpoint), step=1)

    restored = StreamingPremadeContactsDataset(
        data_prefix=str(tmp_path),
        num_shards=4,
        total_shards=4,
        seed=17,
        max_seq_len=512,
        min_fill_fraction=0.95,
        max_open_packs=16,
        row_block_size=16,
        process_index=0,
        process_count=1,
        global_batch_size=6,
    )
    restored.load_checkpoint(str(checkpoint))
    resumed = asyncio.run(restored.get_batch(range(6, 12)))

    for expected, actual in zip(prefetched[6:12], resumed, strict=True):
        np.testing.assert_array_equal(
            np.asarray(expected.tokens), np.asarray(actual.tokens)
        )
