"""Two finite global permutations, with augmentation indexed across both epochs.

Packing stays source-local, as in the default model. Each epoch shuffles the
concatenated sources, and the second uses a folded-in epoch key. Concatenating
the two finite permutations before batching avoids duplicating examples to fill
the first epoch's final batch. Only the last batch of the entire run is padded.
The trainer's absolute example offset therefore resumes correctly on either
side of the epoch boundary.
"""

from dataclasses import dataclass, fields

import jax
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import ConcatDataset
from levanter.data.text.datasets import LmDataConfig, NamedLmDataset
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.utils.thread_utils import blocking_wait

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    AminoAcidAugmentedDataConfig,
    AminoAcidAugmentedDataset,
    _validate_contacts_v1_tokenizer,
    augment_amino_acids,
)
from experiments.exp277_models_single_mpnn_pilot.epoch_data import FULL_CORPUS


@dataclass(frozen=True)
class TwoEpochDataConfig(AminoAcidAugmentedDataConfig):
    """Expose exactly two differently shuffled passes of the selected corpus."""

    expected_packed_examples: int = 0

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_contacts_v1_tokenizer(self)
        epochs = {}
        for epoch in range(2):
            datasets = self.train_sets(
                Pos,
                key=key if epoch == 0 else jax.random.fold_in(key, epoch),
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
                    f"Epoch {epoch} coverage changed: {observed} packed examples, "
                    f"expected {self.expected_packed_examples}"
                )
            epochs[str(epoch)] = dataset
        return AminoAcidAugmentedDataset(
            NamedLmDataset(ConcatDataset(epochs), Pos),
            seed=self.augmentation_seed,
            batch_schedule=batch_schedule,
            num_train_steps=self.augmentation_num_train_steps,
        )


def two_epoch_data(
    data: LmDataConfig, *, num_train_steps: int, expected_packed_examples: int
) -> TwoEpochDataConfig:
    """Retain the default augmentation ramp over the requested full duration."""
    augmented = augment_amino_acids(data, num_train_steps)
    values = {
        field.name: getattr(augmented, field.name)
        for field in fields(AminoAcidAugmentedDataConfig)
    }
    return TwoEpochDataConfig(
        **values, expected_packed_examples=expected_packed_examples
    )
