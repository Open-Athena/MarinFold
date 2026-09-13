"""Finite, shuffled coverage of every packed example in the combined corpus.

The standard training mixture cycles sources independently. For one epoch we
instead concatenate all cache-backed sources before shuffling, and expose the
finite result directly to the trainer. This also avoids mixture-block rounding
at the end of the corpus; the loader pads only the final incomplete batch.
"""

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
    _validate_contacts_v1_tokenizer,
    augment_amino_acids,
)

FULL_CORPUS = "input/full-corpus"


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
