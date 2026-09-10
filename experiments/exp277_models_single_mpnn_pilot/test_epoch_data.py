"""Exercise finite coverage across unequal sources and an incomplete block."""

import asyncio

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import ConcatDataset
from levanter.data.text.datasets import BlockShuffleConfig, DirectDatasetComponent
from levanter.data.text.examples import GrugLmExample
from levanter.schedule import BatchSchedule

from experiments.exp277_models_single_mpnn_pilot.config import TOKENIZER
from experiments.exp277_models_single_mpnn_pilot.epoch_data import (
    FULL_CORPUS,
    OneEpochDataConfig,
)


def synthetic_config(*, expected_examples: int) -> OneEpochDataConfig:
    """Use unique structure tokens to identify examples after named adaptation."""
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
        augmentation_num_train_steps=10,
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
