"""Finite, shuffled coverage of every packed example in the combined corpus.

The standard training mixture cycles sources independently. For one epoch we
instead concatenate all cache-backed sources before shuffling, and expose the
finite result directly to the trainer. This also avoids mixture-block rounding
at the end of the corpus; the loader pads only the final incomplete batch.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig, NamedLmDataset
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.utils.thread_utils import blocking_wait

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AminoAcidAugmentedDataConfig,
    AminoAcidAugmentedDataset,
    _augment_lm_example,
    _validate_contacts_v1_tokenizer,
    augment_amino_acids,
    augmentation_probability,
)

FULL_CORPUS = "input/full-corpus"
STRUCTURE_PROBE_INDEX = 0


@dataclass(frozen=True)
class OneEpochDataConfig(AminoAcidAugmentedDataConfig):
    """Train on the single finite concatenated component without repetition."""

    expected_packed_examples: int = 0

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        datasets = self.train_sets(
            Pos,
            key=key,
            initial_batch_size=batch_schedule.batch_size_at_step(0),
        )
        if set(datasets) != {FULL_CORPUS}:
            raise ValueError(
                f"Expected one concatenated training corpus: {list(datasets)}"
            )
        dataset = datasets[FULL_CORPUS]
        observed = blocking_wait(dataset.async_len())
        if not dataset.is_finite() or observed != self.expected_packed_examples:
            raise ValueError(
                f"One-epoch coverage changed: {observed} packed examples, "
                f"expected {self.expected_packed_examples}"
            )
        return AminoAcidAugmentedDataset(
            NamedLmDataset(dataset, Pos),
            seed=self.augmentation_seed,
            batch_schedule=batch_schedule,
            num_train_steps=self.augmentation_num_train_steps,
        )


class ContinuationEpochDataset(AsyncDataset[LmExample]):
    """Address one local epoch at a restored trainer's absolute data offset.

    Levanter derives the data offset from the absolute optimizer step: the
    loader iterator asks `BatchSchedule.global_data_offset_by_step(step)` for
    every batch it produces, and `train_lm` starts that iterator at the step
    restored from the checkpoint. A continuation that trains a fresh finite
    permutation must therefore answer at those absolute indices; exposing the
    new epoch at local indices 0..N-1 instead would skip everything before the
    restored offset and exhaust the loader long before the step target.

    One read does not follow the absolute step. `DataLoader.__init__` fetches
    global index `STRUCTURE_PROBE_INDEX` once, before iteration begins, purely
    to learn the example structure and build its zeroed padding example. That
    single probe is served from the first example of the new epoch. Any other
    index below the offset means the trainer did not resume where the
    checkpoint says it did, so it is an error rather than a silent reread.
    """

    def __init__(self, dataset: AsyncDataset[LmExample], *, start_offset: int):
        self.dataset = dataset
        self.start_offset = start_offset

    async def async_len(self) -> int:
        return self.start_offset + await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        return await self.dataset.get_batch(
            [self._local_index(int(index)) for index in indices]
        )

    def _local_index(self, index: int) -> int:
        if index >= self.start_offset:
            return index - self.start_offset
        if index == STRUCTURE_PROBE_INDEX:
            return 0
        raise IndexError(
            f"continuation data starts at offset {self.start_offset}, got {index}"
        )


class ContinuedAminoAcidDataset(AsyncDataset[LmExample]):
    """Continue the source run's augmentation ramp, then hold it at full rate."""

    def __init__(
        self,
        dataset: AsyncDataset[LmExample],
        *,
        seed: int,
        batch_schedule: BatchSchedule,
        source_step: int,
        schedule_steps: int,
    ):
        self.dataset = dataset
        self.seed = seed
        self.batch_schedule = batch_schedule
        self.source_step = source_step
        self.schedule_steps = schedule_steps

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        examples = await self.dataset.get_batch(indices)
        return [
            _augment_lm_example(
                example,
                seed=self.seed,
                index=index,
                probability=augmentation_probability(
                    self.source_step
                    + self.batch_schedule.find_step_containing_offset(index),
                    self.schedule_steps,
                ),
            )
            for index, example in zip(indices, examples, strict=True)
        ]


@dataclass(frozen=True)
class ContinuationEpochDataConfig(AminoAcidAugmentedDataConfig):
    """Expose one reshuffled epoch after a full-state checkpoint restore."""

    expected_packed_examples: int = 0
    start_step: int = 0
    source_augmentation_step: int = 0

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        datasets = self.train_sets(
            Pos,
            key=key,
            initial_batch_size=batch_schedule.batch_size_at_step(0),
        )
        if set(datasets) != {FULL_CORPUS}:
            raise ValueError(
                f"Expected one concatenated training corpus: {list(datasets)}"
            )
        dataset = datasets[FULL_CORPUS]
        observed = blocking_wait(dataset.async_len())
        if not dataset.is_finite() or observed != self.expected_packed_examples:
            raise ValueError(
                f"Continuation-epoch coverage changed: {observed} packed examples, "
                f"expected {self.expected_packed_examples}"
            )
        augmented = ContinuedAminoAcidDataset(
            NamedLmDataset(dataset, Pos),
            seed=self.augmentation_seed,
            batch_schedule=batch_schedule,
            source_step=self.source_augmentation_step,
            schedule_steps=self.augmentation_num_train_steps,
        )
        return ContinuationEpochDataset(
            augmented,
            start_offset=self.start_step * batch_schedule.batch_size_at_step(0),
        )


def one_epoch_data(
    data: LmDataConfig, *, num_train_steps: int, expected_packed_examples: int
) -> OneEpochDataConfig:
    """Preserve the augmentation recipe while selecting a finite training set."""
    augmented = augment_amino_acids(data, num_train_steps)
    values = {
        field.name: getattr(augmented, field.name)
        for field in fields(AminoAcidAugmentedDataConfig)
    }
    return OneEpochDataConfig(
        **values, expected_packed_examples=expected_packed_examples
    )


def continuation_epoch_data(
    data: LmDataConfig,
    *,
    start_step: int,
    source_augmentation_step: int,
    augmentation_schedule_steps: int,
    expected_packed_examples: int,
) -> ContinuationEpochDataConfig:
    """Build a finite new epoch aligned with a restored absolute trainer step."""
    augmented = augment_amino_acids(data, augmentation_schedule_steps)
    values = {
        field.name: getattr(augmented, field.name)
        for field in fields(AminoAcidAugmentedDataConfig)
    }
    return ContinuationEpochDataConfig(
        **values,
        expected_packed_examples=expected_packed_examples,
        start_step=start_step,
        source_augmentation_step=source_augmentation_step,
    )
