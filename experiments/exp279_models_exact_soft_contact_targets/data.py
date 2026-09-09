# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach full-document targets to the reference pack/shuffle/mixture path."""

import asyncio
from dataclasses import dataclass

import jax
import numpy as np
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    PackedTokenDataset,
)
from levanter.data.text.examples import named_lm_example_from_grug
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.models.lm_model import LmExample

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AminoAcidAugmentedDataset,
)

from .targets import ContactExample, DocumentSlice, Vocabulary, pack_targets


class ContactPackedDataset(AsyncDataset[LmExample]):
    """Stock packing and masks, with metadata recovered from complete cache rows.

    The pinned packer's document ranges are the only private API used here.
    Read full rows before slicing (including edges beyond a truncated fragment),
    and verify their concatenation against the stock output on every read.
    This initial implementation performs an additional cache read; measure its
    cost before replacing it with a sidecar or changing the existing cache.
    """

    def __init__(
        self, source: PackedTokenDataset, cache, vocab: Vocabulary, edge_capacity: int
    ):
        self.source, self.cache, self.vocab = source, cache, vocab
        self.edge_capacity = edge_capacity

    async def async_len(self):
        return await self.source.async_len()

    def is_finite(self):
        return self.source.is_finite()

    async def get_batch(self, indices):
        if not indices:
            return []
        ranges = [self.source.packed._pack_indices[i] for i in indices]
        doc_ids = sorted({i for r in ranges for i in r})
        examples, rows = await asyncio.gather(
            self.source.get_batch(indices), self.cache.get_batch(doc_ids)
        )
        documents = dict(zip(doc_ids, rows, strict=True))
        result = []
        for raw, doc_range in zip(examples, ranges, strict=True):
            example = named_lm_example_from_grug(raw, self.source.Pos)
            slices = []
            for i in doc_range:
                tokens = np.asarray(documents[i]["input_ids"])
                start, stop = 0, len(tokens)
                if len(tokens) > self.source.Pos.size:
                    if len(doc_range) != 1:
                        raise ValueError(
                            "Stock packer put an oversized document into a multi-document pack"
                        )
                    if self.source.packed.slice_strategy == "right":
                        start = len(tokens) - self.source.Pos.size
                    else:
                        stop = self.source.Pos.size
                slices.append(DocumentSlice(tokens, start, stop))
            targets = pack_targets(
                slices,
                np.asarray(example.tokens.array),
                self.vocab,
                edge_capacity=self.edge_capacity,
                position_axis=self.source.Pos.name,
            )
            result.append(
                ContactExample(
                    example.tokens,
                    example.loss_weight,
                    example.attn_mask,
                    targets=targets,
                )
            )
        return result


@dataclass(frozen=True)
class ContactDataConfig(LmDataConfig):
    """Same train inputs for CE and soft arms; validation remains stock CE."""

    edge_capacity: int = 2731
    augmentation_seed: int = 166
    augmentation_num_train_steps: int = 145200

    def build_token_datasets(self, caches, Pos, *, split):
        # Levanter's annotation fixes this to Grug examples, but its shuffle /
        # mixture implementations are generic. Our train_set consumes Named
        # examples directly; validation continues to return Grug examples.
        datasets: dict[str, AsyncDataset] = dict(
            super().build_token_datasets(caches, Pos, split=split)
        )
        if split != "train":
            return datasets
        vocab = Vocabulary.from_tokenizer(self.the_tokenizer)
        for name, dataset in datasets.items():
            component = self.components[name]
            if (
                not isinstance(component, DatasetComponent)
                or not isinstance(component.format, TextLmDatasetFormat)
                or not isinstance(dataset, PackedTokenDataset)
            ):
                raise ValueError("exp279 requires the reference packed text caches")
            datasets[name] = ContactPackedDataset(
                dataset, caches[name], vocab, self.edge_capacity
            )
        return datasets

    def train_set(self, Pos, batch_schedule, *, key):
        # Same split keys, component shuffle, mixture and order as LmDataConfig.
        # Components already return Named examples; stock NamedLmDataset would
        # discard the extra metadata. No changes to sampling or attention.
        if not isinstance(self.train_weights, dict):
            raise ValueError("exp279 requires the constant reference m2 mixture")
        mix_key, shuffle_key = jax.random.split(key)
        datasets = self.train_sets(
            Pos,
            key=shuffle_key,
            initial_batch_size=batch_schedule.batch_size_at_step(0),
        )
        mixed = MixtureDataset(
            datasets=datasets,
            weights=self.train_weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )
        return AminoAcidAugmentedDataset(
            mixed,
            seed=self.augmentation_seed,
            batch_schedule=batch_schedule,
            num_train_steps=self.augmentation_num_train_steps,
        )
