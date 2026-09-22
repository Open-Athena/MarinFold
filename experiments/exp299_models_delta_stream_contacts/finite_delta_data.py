# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite one-epoch data configuration for the exp277-scale V2 corpus."""

from dataclasses import dataclass, fields

from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig, NamedLmDataset
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.utils.thread_utils import blocking_wait

FULL_CORPUS = "delta-v2/full-corpus"


@dataclass(frozen=True)
class FiniteOneEpochDataConfig(LmDataConfig):
    """Expose one shuffled, finite concatenation without mixture cycling."""

    expected_packed_examples: int = 0

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        datasets = self.train_sets(
            Pos,
            key=key,
            initial_batch_size=batch_schedule.batch_size_at_step(0),
        )
        if set(datasets) != {FULL_CORPUS}:
            raise ValueError(f"expected one concatenated training corpus, found {list(datasets)}")
        dataset = datasets[FULL_CORPUS]
        observed = blocking_wait(dataset.async_len())
        if not dataset.is_finite() or observed != self.expected_packed_examples:
            raise ValueError(
                f"finite coverage changed: {observed} packed examples, expected {self.expected_packed_examples}"
            )
        return NamedLmDataset(dataset, Pos)


def finite_one_epoch_data(data: LmDataConfig, *, expected_packed_examples: int) -> FiniteOneEpochDataConfig:
    """Convert an ordinary LM data config to strict finite-epoch semantics."""
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return FiniteOneEpochDataConfig(**values, expected_packed_examples=expected_packed_examples)
