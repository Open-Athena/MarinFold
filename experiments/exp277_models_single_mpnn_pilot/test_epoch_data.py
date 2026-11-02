"""Exercise finite coverage across unequal sources and an incomplete block."""

import asyncio
from collections.abc import Sequence
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from haliax.partitioning import ResourceAxis
from jax.sharding import Mesh
from levanter.data.dataset import AsyncDataset, ListAsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import ConcatDataset
from levanter.data.text.datasets import BlockShuffleConfig, DirectDatasetComponent
from levanter.data.text.examples import GrugLmExample
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule

from experiments.exp277_models_single_mpnn_pilot.config import TOKENIZER
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    STRUCTURE_PROBE_INDEX,
    ContinuationEpochDataConfig,
    OneEpochDataConfig,
)

EXAMPLES = 50
BATCH_SIZE = 4
START_STEP = 3
START_OFFSET = START_STEP * BATCH_SIZE


def synthetic_config(
    *, expected_examples: int, augmentation_steps: int = 10
) -> OneEpochDataConfig:
    """Use unique structure tokens to identify examples after named adaptation.

    A very long `augmentation_steps` keeps the scheduled amino-acid shuffle off
    these synthetic tokens, which are not real contacts-v1 documents.
    """
    examples = [
        GrugLmExample.causal(
            tokens=jnp.array([2, 8, 100, 200, 101, 201, 9, 500 + i, 10, 1]),
            eos_id=1,
            block_cross_document_attention=True,
        )
        for i in range(50)
    ]
    combined = ConcatDataset(
        {
            "small": ListAsyncDataset(examples[:5]),
            "medium": ListAsyncDataset(examples[5:23]),
            "large": ListAsyncDataset(examples[23:]),
        }
    )
    return OneEpochDataConfig(
        tokenizer=TOKENIZER,
        components={FULL_CORPUS: DirectDatasetComponent(datasets={"train": combined})},
        train_weights={FULL_CORPUS: 1.0},
        shuffle=BlockShuffleConfig(
            io_block_size=4, window_blocks=3, perm_type="feistel"
        ),
        augmentation_num_train_steps=augmentation_steps,
        expected_packed_examples=expected_examples,
    )


def test_one_epoch_visits_every_example_once_including_partial_shuffle_block() -> None:
    data = synthetic_config(expected_examples=50).train_set(
        Axis("position", 10), BatchSchedule(128), key=jax.random.PRNGKey(1729)
    )
    assert data.is_finite()
    assert asyncio.run(data.async_len()) == 50
    first = asyncio.run(data.get_batch(list(range(50))))
    ids = [int(np.asarray(example.tokens.array)[7]) for example in first]
    assert sorted(ids) == list(range(500, 550))
    assert ids != list(range(500, 550))
    resumed = asyncio.run(data.get_batch(list(range(37, 50))))
    assert [int(np.asarray(example.tokens.array)[7]) for example in resumed] == ids[37:]


def test_one_epoch_rejects_unexpected_corpus_size() -> None:
    with pytest.raises(ValueError, match="One-epoch coverage changed"):
        synthetic_config(expected_examples=51).train_set(
            Axis("position", 10), BatchSchedule(128), key=jax.random.PRNGKey(1729)
        )


def continuation_config() -> ContinuationEpochDataConfig:
    """Place the synthetic corpus after a restored trainer's absolute offset."""
    base = synthetic_config(expected_examples=EXAMPLES)
    values = {
        field.name: getattr(base, field.name)
        for field in fields(base)
        if field.name
        not in {"augmentation_num_train_steps", "expected_packed_examples"}
    }
    return ContinuationEpochDataConfig(
        **values,
        augmentation_num_train_steps=1_000_000_000,
        expected_packed_examples=EXAMPLES,
        start_step=START_STEP,
        source_augmentation_step=40,
    )


def continuation_train_set(*, seed: int = 1729) -> AsyncDataset[LmExample]:
    return continuation_config().train_set(
        Axis("position", 10), BatchSchedule(BATCH_SIZE), key=jax.random.PRNGKey(seed)
    )


def example_ids(examples: Sequence[LmExample]) -> list[int]:
    """Read back the unique structure token that identifies each example."""
    return [int(np.asarray(example.tokens.array)[7]) for example in examples]


def trainer_loader(dataset: AsyncDataset[LmExample]) -> DataLoader:
    """Build the loader `Trainer.data_loader` builds, on this host's devices."""
    mesh = Mesh(
        np.array(jax.devices()).reshape(1, -1, 1),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        mesh=mesh,
        axis_resources={"batch": ResourceAxis.DATA},
        batch_axis_name="batch",
        max_buffered_batches=0,
    )


def served_batches(loader: DataLoader, *, start_step: int) -> list[list[int]]:
    """Collect each served batch's real ids; levanter pads with a zeroed example."""
    return [
        [int(token) for token in np.asarray(batch.tokens.array)[:, 7] if token != 0]
        for batch in loader.iter_from_step(start_step)
    ]


def test_continuation_epoch_remaps_absolute_offset_and_reshuffles() -> None:
    first = continuation_train_set(seed=1729)
    second = continuation_train_set(seed=1730)
    assert asyncio.run(first.async_len()) == START_OFFSET + EXAMPLES
    epoch = list(range(START_OFFSET, START_OFFSET + EXAMPLES))
    first_ids = example_ids(asyncio.run(first.get_batch(epoch)))
    second_ids = example_ids(asyncio.run(second.get_batch(epoch)))
    assert sorted(first_ids) == list(range(500, 550))
    assert sorted(second_ids) == list(range(500, 550))
    assert first_ids != second_ids


def test_continuation_answers_the_loader_structure_probe_at_index_zero() -> None:
    """`DataLoader.__init__` reads global index 0 before any iteration starts.

    That read happens whatever step the trainer restored, so it lands in the
    unused prefix and is what broke `/bizon/exp277-continue-smoke-a01`.
    """
    data = continuation_train_set()
    probe = asyncio.run(data.getitem_async(STRUCTURE_PROBE_INDEX))
    first = asyncio.run(data.getitem_async(START_OFFSET))
    assert example_ids([probe]) == example_ids([first])


def test_continuation_rejects_a_trainer_that_did_not_restore_its_step() -> None:
    """Only the lone structure probe is served from the prefix."""
    data = continuation_train_set()
    with pytest.raises(IndexError, match="starts at offset 12"):
        asyncio.run(data.get_batch([STRUCTURE_PROBE_INDEX, 1]))
    with pytest.raises(IndexError, match="starts at offset 12"):
        asyncio.run(data.get_batch([11]))


def test_restored_loader_covers_the_whole_new_epoch_exactly_once() -> None:
    """Drive the real loader the way `train_lm` does after a checkpoint restore."""
    batches = served_batches(
        trainer_loader(continuation_train_set()), start_step=START_STEP
    )
    assert len(batches) == -(-EXAMPLES // BATCH_SIZE)
    ids = [identifier for batch in batches for identifier in batch]
    assert sorted(ids) == list(range(500, 550))
    assert ids != list(range(500, 550))


def test_a_locally_indexed_epoch_would_silently_truncate_the_continuation() -> None:
    """Why the absolute offset is required rather than local indices 0..N-1.

    Levanter derives the data offset from the absolute optimizer step, so an
    unshifted finite epoch skips everything before the restored step and then
    runs out of data long before the trainer's step target.
    """
    unshifted = synthetic_config(
        expected_examples=EXAMPLES, augmentation_steps=1_000_000_000
    ).train_set(
        Axis("position", 10), BatchSchedule(BATCH_SIZE), key=jax.random.PRNGKey(1729)
    )
    batches = served_batches(trainer_loader(unshifted), start_step=START_STEP)
    ids = [identifier for batch in batches for identifier in batch]
    assert len(ids) == EXAMPLES - START_OFFSET
    assert len(batches) < -(-EXAMPLES // BATCH_SIZE)
