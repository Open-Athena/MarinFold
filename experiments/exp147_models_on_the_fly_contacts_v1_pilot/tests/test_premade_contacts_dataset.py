# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import DirectDatasetComponent
from marin.execution.artifact import ArtifactRecord
from marin.execution.lazy import StepContext
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import TrainLmOnPodConfig
from marinfold.document_structures.contacts_v1 import (
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyzed_to_row,
)
from marinfold.document_structures.documents import causal_training_document
from marinfold_models.shard_documents import (
    PackedDocuments,
    best_fit_pack_documents,
    fixed_quota_pack_slots,
)

from premade_contacts_dataset import (
    FixedQuotaPremadeContactsDataset,
)
from smoke_dataset import main as smoke_dataset_main
from stage_pilot import _destination_path
from train import (
    CONTACTS_TOKENIZER,
    CONTACTS_TOKENIZER_REPO,
    _with_local_tokenizer,
    build_step,
)


def _analyzed(entry_id: str, length: int) -> AnalyzedStructure:
    residues = tuple(
        ResidueInfo(seq_index=i, resname="ALA", resnum=i + 1, chain="A")
        for i in range(length)
    )
    contacts = tuple(
        RawContact(seq_i=i, seq_j=i + 6, degree=0.5) for i in range(max(0, length - 6))
    )
    return AnalyzedStructure(
        entry_id=entry_id,
        residues=residues,
        contacts=contacts,
        global_plddt=90.0,
        source_path=Path("<test>"),
    )


def _write_shards(
    tmp_path: Path, *, num_shards: int = 4, rows_per_shard: int = 80
) -> None:
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


def _dataset(tmp_path: Path, *, examples_per_shard: int = 40):
    return FixedQuotaPremadeContactsDataset(
        data_prefix=str(tmp_path),
        num_shards=4,
        total_shards=4,
        examples_per_shard=examples_per_shard,
        seed=17,
        max_seq_len=512,
        shard_cache_size=2,
    )


def _packed(pack_id: int) -> PackedDocuments:
    document = causal_training_document(np.full(10, pack_id, dtype=np.int32))
    return PackedDocuments(documents=[document], used_tokens=len(document))


def test_best_fit_decreasing_combines_complementary_documents():
    documents = [
        causal_training_document(np.arange(length, dtype=np.int32))
        for length in (61, 61, 39)
    ]

    packs, truncated = best_fit_pack_documents(
        documents,
        max_seq_len=100,
        max_segments_per_example=8,
    )

    assert truncated == 0
    assert sorted(pack.used_tokens for pack in packs) == [61, 100]


def test_fixed_quota_uniformly_selects_bins_and_pads():
    packs = tuple(_packed(index) for index in range(5))
    first = fixed_quota_pack_slots(
        packs,
        examples_per_shard=3,
        rng=np.random.default_rng(4),
    )
    second = fixed_quota_pack_slots(
        packs,
        examples_per_shard=3,
        rng=np.random.default_rng(4),
    )
    assert [slot.documents[0].token_ids[0] for slot in first if slot] == [
        slot.documents[0].token_ids[0] for slot in second if slot
    ]
    assert len(first) == 3

    selection_counts = np.zeros(len(packs), dtype=np.int32)
    for seed in range(1_000):
        selected = fixed_quota_pack_slots(
            packs,
            examples_per_shard=3,
            rng=np.random.default_rng(seed),
        )
        for slot in selected:
            assert slot is not None
            pack_id = int(slot.documents[0].token_ids[0])
            selection_counts[pack_id] += 1
    assert np.all((550 < selection_counts) & (selection_counts < 650))

    padded = fixed_quota_pack_slots(
        packs[:2],
        examples_per_shard=5,
        rng=np.random.default_rng(4),
    )
    assert len(padded) == 5
    assert sum(slot is None for slot in padded) == 3


def test_index_mapping_is_stateless_and_epoch_shuffled(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path, examples_per_shard=40)

    assert dataset.location_for_index(0)[0:2] == dataset.location_for_index(0)[0:2]
    assert dataset.location_for_index(39)[0:2] == dataset.location_for_index(0)[0:2]
    assert dataset.location_for_index(40)[1] != dataset.location_for_index(0)[1]
    assert dataset.location_for_index(160)[0] == 1

    requested = [81, 0, 159, 40, 81]
    first = asyncio.run(dataset.get_batch(requested))
    second = asyncio.run(dataset.get_batch(requested))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left.tokens), np.asarray(right.tokens))
        np.testing.assert_array_equal(
            np.asarray(left.loss_weight), np.asarray(right.loss_weight)
        )


def test_remote_paths_preserve_uri_scheme():
    dataset = FixedQuotaPremadeContactsDataset(
        data_prefix="gs://bucket/prefix",
        num_shards=1,
    )

    assert (
        dataset._shard_path(0)
        == "gs://bucket/prefix/shard-00000-of-03338.parquet"
    )
    assert (
        _destination_path("gs://bucket/pilot", "shard.parquet")
        == "gs://bucket/pilot/contacts/shard.parquet"
    )


def test_padding_slots_have_zero_loss(tmp_path: Path):
    _write_shards(tmp_path, rows_per_shard=4)
    dataset = _dataset(tmp_path, examples_per_shard=20)

    examples = asyncio.run(dataset.get_batch(range(20)))

    assert any(np.asarray(example.loss_weight).sum() == 0 for example in examples)
    assert dataset.stats.padding_packs_emitted > 0


def test_overflow_emits_exact_quota(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path, examples_per_shard=4)

    examples = asyncio.run(dataset.get_batch(range(4)))

    assert len(examples) == 4
    assert dataset.stats.packs_discarded_by_quota > 0
    assert all(np.asarray(example.loss_weight).sum() > 0 for example in examples)


def test_getitem_is_non_consuming_and_reuses_cached_shard(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path)

    first = asyncio.run(dataset.getitem_async(0))
    second = asyncio.run(dataset.getitem_async(0))

    np.testing.assert_array_equal(np.asarray(first.tokens), np.asarray(second.tokens))
    assert dataset.stats.shards_constructed == 1


def test_single_component_mixture_preserves_random_access(tmp_path: Path):
    _write_shards(tmp_path)
    dataset = _dataset(tmp_path)
    mixture = MixtureDataset(
        datasets={"contacts": dataset},
        weights={"contacts": 1.0},
        block_size=1,
        key=0,
    )

    requested = [3, 1, 3]
    mixed = asyncio.run(mixture.get_batch(requested))
    direct = asyncio.run(dataset.get_batch(requested))

    for left, right in zip(mixed, direct, strict=True):
        np.testing.assert_array_equal(np.asarray(left.tokens), np.asarray(right.tokens))


def test_smoke_dataset_reports_real_loader_stats(tmp_path: Path, capsys):
    rows = [
        analyzed_to_row(_analyzed(f"r{row_index}", 20 + row_index))
        for row_index in range(8)
    ]
    path = tmp_path / "shard-00000-of-03338.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    assert (
        smoke_dataset_main(
            [
                "--data-prefix",
                str(tmp_path),
                "--num-shards",
                "1",
                "--examples-per-shard",
                "3",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["examples"] == 3
    assert result["loss_tokens"] > 0
    assert result["stats"]["shards_constructed"] == 1
    assert 0 < result["stats"]["packing_utilization"] <= 1


def test_launch_config_contains_direct_dataset_and_expected_routing():
    step = build_step()
    validation = step.deps[0]
    context = StepContext.for_run(
        "gs://marin-us-east5/test/output",
        "gs://marin-us-east5/test",
        runtime_args=step.runtime_args,
        deps=step.deps,
    )
    cache = TokenizedCache(path="gs://marin-us-east5/test/validation")
    cache.__dict__["record"] = ArtifactRecord(
        config={
            "tokenizer": CONTACTS_TOKENIZER_REPO,
            "format": {"text_key": "document"},
        }
    )
    context._loaded[id(validation)] = cache

    config = step.build_config(context)

    assert isinstance(config, TrainLmOnPodConfig)
    train_config = config.train_config
    component = train_config.data.components[
        "on-the-fly/esm-atlas-contacts-v1"
    ]
    assert isinstance(component, DirectDatasetComponent)
    assert train_config.data.tokenizer == CONTACTS_TOKENIZER
    assert train_config.trainer.train_batch_size == 256
    assert train_config.trainer.per_device_parallelism == 16
    assert train_config.trainer.tracker.entity == "open-athena"
    assert train_config.trainer.tracker.project == "MarinFold"
    assert config.resources.zone == "us-east5-b"

    local_config = _with_local_tokenizer(config, "/tmp/pinned-tokenizer")
    assert local_config.train_config.data.tokenizer == "/tmp/pinned-tokenizer"
    assert config.train_config.data.tokenizer == CONTACTS_TOKENIZER
