# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Real TensorStore caches and stock packer, including oversize fragments."""

import asyncio
import json
from dataclasses import fields, replace

import haliax as hax
import jax
import numpy as np
import pytest
from conftest import make_document
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    PackedTokenDataset,
)
from levanter.data.text.examples import named_lm_example_from_grug
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheLedger, TreeCache
from levanter.store.tree_store import TreeStore
from test_targets import target_q
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AminoAcidAugmentedDataset,
)
from experiments.exp279_models_exact_soft_contact_targets.data import (
    ContactDataConfig,
    ContactPackedDataset,
)
from experiments.exp279_models_exact_soft_contact_targets.inputs import (
    resolve_tokenizer,
)
from experiments.exp279_models_exact_soft_contact_targets.targets import ContactExample
from experiments.exp279_models_exact_soft_contact_targets.train import build_config


def make_cache(path, docs):
    exemplar = {"input_ids": np.zeros(0, np.int32)}
    store = TreeStore.open(exemplar, str(path), mode="w")
    store.extend([{"input_ids": d} for d in docs])
    ledger = CacheLedger(
        total_num_rows=len(docs),
        shard_rows={"fixture": len(docs)},
        is_finished=True,
        finished_shards=["fixture"],
        field_counts={"input_ids": sum(map(len, docs))},
    )
    (path / "shard_ledger.json").write_text(ledger.to_json())
    return TreeCache.load(str(path), exemplar)


def test_production_validation_reads_only_frozen_validation_cache(tmp_path):
    train_doc = make_document([(143, 144)])
    val_doc = make_document([(145, 146)])
    paths = {
        name: tmp_path / name / split
        for name, split in (("afdb", "train"), ("esm", "train"), ("val", "validation"))
    }
    for name, path in paths.items():
        make_cache(path, [val_doc if name == "val" else train_doc] * 8)
    config = build_config(
        {"inputs": {name: {"cache_dir": str(path)} for name, path in paths.items()}},
        arm="soft",
        phase_name="base",
        run_name="exp279-soft-validation-test",
        output=str(tmp_path),
        resume=None,
    )
    data = replace(config.data, tokenizer=resolve_tokenizer())
    Pos = hax.Axis("position", len(val_doc))
    tagged = data.tagged_eval_sets(Pos)
    assert len(tagged) == 1
    dataset, tags = tagged[0]
    assert "val" in tags
    examples = asyncio.run(dataset.get_batch([0, 1]))
    for example in examples:
        assert not isinstance(example, ContactExample)
        np.testing.assert_array_equal(example.tokens.array, val_doc)
    assert set(data.build_caches("train")) == {"afdb", "esm"}
    broken = replace(
        data,
        components={
            **data.components,
            "val": replace(
                data.components["val"], cache_dir=str(paths["val"]), flat_cache=True
            ),
        },
    )
    with pytest.raises(ValueError, match="Required validation dataset"):
        broken.tagged_eval_sets(Pos)


@pytest.mark.parametrize("strategy", ["left", "right"])
def test_real_packing_preserves_tokens_masks_and_full_contacts(
    tmp_path, vocab, strategy
):
    docs = [
        make_document([(143, 144)]),
        make_document([]),
        make_document([(143, 144), (143, 145), (146, 143), (145, 146)]),
    ]
    cache = make_cache(tmp_path / "cache", docs)
    Pos = hax.Axis("position", 24)
    stock = PackedTokenDataset(cache, Pos, slice_strategy=strategy)
    wrapped = ContactPackedDataset(stock, cache, vocab, edge_capacity=8)
    indices = list(reversed(range(asyncio.run(stock.async_len())))) + [0]
    original = asyncio.run(stock.get_batch(indices))
    actual = asyncio.run(wrapped.get_batch(indices))
    for raw, augmented in zip(original, actual, strict=True):
        named = named_lm_example_from_grug(raw, Pos)
        for a, b in zip(
            jax.tree.leaves(named),
            jax.tree.leaves(
                type(named)(
                    augmented.tokens, augmented.loss_weight, augmented.attn_mask
                )
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        assert np.all(np.isfinite(target_q(augmented.targets, augmented.tokens.array)))
    assert asyncio.run(wrapped.get_batch([])) == []


def test_augmentation_random_access_and_restart_are_identical(tmp_path, vocab):
    docs = [make_document([(143, 144), (143, 145)]) for _ in range(12)]
    cache = make_cache(tmp_path / "cache", docs)
    Pos = hax.Axis("position", 64)
    stock = PackedTokenDataset(cache, Pos)
    wrapped = ContactPackedDataset(stock, cache, vocab, edge_capacity=8)
    augmented = AminoAcidAugmentedDataset(
        wrapped, seed=166, batch_schedule=BatchSchedule(2), num_train_steps=2
    )
    order = [0, 1, 2, 3]
    examples = asyncio.run(augmented.get_batch(order))
    reread = asyncio.run(augmented.get_batch([3, 1]))
    for index, actual in zip([3, 1], reread, strict=True):
        for a, b in zip(
            jax.tree.leaves(examples[index]), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_array_equal(a, b)
    before = asyncio.run(wrapped.get_batch(order))
    assert not np.array_equal(examples[2].tokens.array, before[2].tokens.array)
    for a, b in zip(before, examples, strict=True):
        np.testing.assert_array_equal(a.loss_weight.array, b.loss_weight.array)
        for x, y in zip(
            jax.tree.leaves(a.targets), jax.tree.leaves(b.targets), strict=True
        ):
            np.testing.assert_array_equal(x, y)


def test_full_mixture_shuffle_augmentation_matches_stock(tmp_path, vocab):
    names = [f"<unused{i}>" for i in range(vocab.size)]
    for name, value in {
        "<pad>": 0,
        "<eos>": 1,
        "<contacts-v1>": 2,
        "<contact>": 5,
        "<begin_sequence>": 8,
        "<begin_statements>": 9,
        "<end>": 10,
        "<ALA>": 86,
        "<UNK>": 2844,
        **{f"<p{i}>": 143 + i for i in range(2000)},
    }.items():
        names[value] = name
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    Tokenizer(
        WordLevel(dict(zip(names, range(vocab.size), strict=True)), unk_token="<UNK>")
    ).save(str(tokenizer_dir / "tokenizer.json"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<UNK>",
            }
        )
    )
    for name, edge in (("afdb", (143, 144)), ("esm", (143, 145))):
        make_cache(tmp_path / name, [make_document([edge]) for _ in range(32)])
    config = LmDataConfig(
        tokenizer=str(tokenizer_dir),
        auto_build_caches=False,
        components={
            name: DatasetComponent(
                cache_dir=str(tmp_path / name),
                flat_cache=True,
                split="train",
                pack=True,
            )
            for name in ("afdb", "esm")
        },
        train_weights={"afdb": 0.2, "esm": 0.8},
        mixture_block_size=10,
        shuffle=BlockShuffleConfig(io_block_size=2, window_blocks=2),
    )
    new = ContactDataConfig(
        **{f.name: getattr(config, f.name) for f in fields(config)},
        edge_capacity=8,
        augmentation_num_train_steps=2,
    )
    schedule, Pos, key = (
        BatchSchedule(2),
        hax.Axis("position", 64),
        jax.random.PRNGKey(232),
    )
    original = AminoAcidAugmentedDataset(
        config.train_set(Pos, schedule, key=key),
        seed=166,
        batch_schedule=schedule,
        num_train_steps=2,
    )
    actual = new.train_set(Pos, schedule, key=key)
    order = list(range(20)) + [4, 1]
    for a, b in zip(
        asyncio.run(original.get_batch(order)),
        asyncio.run(actual.get_batch(order)),
        strict=True,
    ):
        np.testing.assert_array_equal(a.tokens.array, b.tokens.array)
        np.testing.assert_array_equal(a.loss_weight.array, b.loss_weight.array)
        for x, y in zip(
            jax.tree.leaves(a.attn_mask), jax.tree.leaves(b.attn_mask), strict=True
        ):
            np.testing.assert_array_equal(x, y)
