"""Coverage and resume checks through the same loader used by training."""

import asyncio
from dataclasses import fields

import jax
import pytest
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule

from experiments.exp277_models_single_mpnn_pilot.test_epoch_data import (
    example_ids,
    served_batches,
    synthetic_config,
    trainer_loader,
)
from experiments.exp364_models_mpnn_only_lr_sweep.epoch_data import TwoEpochDataConfig


def two_epochs(*, expected_examples: int = 50) -> AsyncDataset[LmExample]:
    # The two-statement synthetic examples can legitimately draw an identity
    # permutation. Focus this fixture on traversal; the config comparison
    # separately checks the unchanged production augmentation schedule.
    base = synthetic_config(
        expected_examples=expected_examples, augmentation_steps=1_000_000_000
    )
    return TwoEpochDataConfig(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    ).train_set(Axis("position", 10), BatchSchedule(4), key=jax.random.PRNGKey(1729))


def test_each_epoch_is_complete_reshuffled_and_resume_is_exact() -> None:
    data = two_epochs()
    assert data.is_finite()
    assert asyncio.run(data.async_len()) == 100
    ids = example_ids(asyncio.run(data.get_batch(list(range(100)))))
    assert sorted(ids[:50]) == sorted(ids[50:]) == list(range(500, 550))
    assert ids[:50] != ids[50:]
    assert example_ids(asyncio.run(two_epochs().get_batch([99, 49, 50, 0]))) == [
        ids[index] for index in [99, 49, 50, 0]
    ]
    batches = served_batches(trainer_loader(data), start_step=0)
    assert len(batches) == 25
    assert all(len(batch) == 4 for batch in batches)
    assert [identifier for batch in batches for identifier in batch] == ids
    resumed = served_batches(trainer_loader(two_epochs()), start_step=12)
    assert [identifier for batch in resumed for identifier in batch] == ids[48:]


def test_wrong_packed_cardinality_fails_before_training() -> None:
    with pytest.raises(ValueError, match="coverage changed"):
        two_epochs(expected_examples=51)
