"""Check a full-state fork keeps the parent's first-epoch trajectory."""

import asyncio
from dataclasses import asdict, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent
from levanter.data.text.examples import GrugLmExample
from levanter.schedule import BatchSchedule

from experiments.exp277_models_single_mpnn_pilot.epoch_data import OneEpochDataConfig
from experiments.exp277_models_single_mpnn_pilot.test_epoch_data import (
    synthetic_config,
    trainer_loader,
)
from experiments.exp364_models_mpnn_only_lr_sweep.config import (
    EPOCH_PACKED_EXAMPLES,
    PREFIX,
    TRAIN_STEPS,
)
from experiments.exp364_models_mpnn_only_lr_sweep.cooldown_train import (
    EPOCH_STEPS,
    RUN_ID,
    SOURCE_ROOT,
    SOURCE_STEP,
    build_cooldown_run,
)
from experiments.exp364_models_mpnn_only_lr_sweep.epoch_data import (
    FULL_CORPUS,
    TwoEpochDataConfig,
)
from experiments.exp364_models_mpnn_only_lr_sweep.test_train import resolve
from experiments.exp364_models_mpnn_only_lr_sweep.train import build_run


def test_fork_preserves_recipe_and_full_state_resume() -> None:
    parent, _ = resolve(build_run, trial="lr5e-4", smoke=False, nodes=4)
    fork, _ = resolve(build_cooldown_run, smoke=False, nodes=4)
    assert asdict(fork.model) == asdict(parent.model)
    assert asdict(fork.data) == asdict(parent.data)
    assert isinstance(fork.data, OneEpochDataConfig)
    assert isinstance(parent.data, TwoEpochDataConfig)
    assert fork.data.augmentation_num_train_steps == TRAIN_STEPS
    assert fork.data_seed == parent.data_seed
    assert fork.train_seq_len == parent.train_seq_len
    before, after = asdict(parent.optimizer), asdict(fork.optimizer)
    assert after.pop("warmup") == int(before.pop("warmup") * TRAIN_STEPS) == 37_376
    assert after.pop("decay") == int(before.pop("decay") * EPOCH_STEPS) == 37_377
    assert before == after
    for field in ("train_batch_size", "seed", "mp", "steps_per_eval", "mesh"):
        assert getattr(parent.trainer, field) == getattr(fork.trainer, field)
    assert (
        fork.trainer.initialize_from == f"{SOURCE_ROOT}/checkpoints/step-{SOURCE_STEP}"
    )
    assert fork.trainer.allow_partial_checkpoint is False
    assert fork.initialize_from_checkpoint_path is None
    assert fork.initialize_model_from_checkpoint_path is None
    assert fork.trainer.id == RUN_ID != parent.trainer.id
    assert fork.trainer.checkpointer.base_path == f"{PREFIX}/runs/{RUN_ID}/checkpoints"
    assert fork.trainer.tracker.name == RUN_ID
    assert fork.trainer.tracker.replicate_path == f"{PREFIX}/runs/{RUN_ID}"
    assert fork.trainer.num_train_steps == EPOCH_STEPS == 186_885
    assert EPOCH_PACKED_EXAMPLES - (EPOCH_STEPS - 1) * 128 == 37
    assert parent.trainer.num_train_steps == TRAIN_STEPS == 373_769


def test_lr_is_unchanged_until_last_twenty_percent_of_one_epoch() -> None:
    parent, _ = resolve(build_run, trial="lr5e-4", smoke=False, nodes=4)
    fork, _ = resolve(build_cooldown_run, smoke=False, nodes=4)
    original = parent.optimizer.lr_scheduler(TRAIN_STEPS)
    cooldown = fork.optimizer.lr_scheduler(EPOCH_STEPS)
    boundary = 149_508
    for step in (0, 20_000, 37_376, SOURCE_STEP + 1, boundary):
        assert float(cooldown(step)) == pytest.approx(float(original(step)))
    assert float(cooldown(boundary)) == pytest.approx(5e-4)
    assert float(cooldown(boundary + 18_688)) == pytest.approx(
        5e-4 - 4.5e-4 * 18_688 / 37_377
    )
    assert float(cooldown(EPOCH_STEPS)) == pytest.approx(5e-5)
    assert float(original(EPOCH_STEPS)) == pytest.approx(5e-4)


def test_reslicing_is_stable_and_smoke_has_an_isolated_short_cooldown() -> None:
    first, fingerprint = resolve(build_cooldown_run, smoke=False, nodes=1)
    second, other = resolve(build_cooldown_run, smoke=False, nodes=4, attempt=2)
    assert fingerprint == other
    assert first.trainer.id == second.trainer.id
    smoke, _ = resolve(build_cooldown_run, smoke=True, nodes=1)
    assert smoke.trainer.id != first.trainer.id
    assert smoke.trainer.checkpointer.base_path != first.trainer.checkpointer.base_path
    assert smoke.trainer.num_train_steps == SOURCE_STEP + 11
    scheduler = smoke.optimizer.lr_scheduler(smoke.trainer.num_train_steps)
    assert float(scheduler(SOURCE_STEP + 1)) == pytest.approx(5e-4)
    assert float(scheduler(SOURCE_STEP + 11)) == pytest.approx(5e-5)


def test_fork_reuses_augmented_first_epoch_and_masks_only_final_padding() -> None:
    # A short augmentation schedule exercises real token permutation as well as
    # shuffling, so merely preserving the unaugmented example IDs cannot pass.
    base = synthetic_config(expected_examples=50, augmentation_steps=10)
    # Thirty distinguishable statements make accidental identity permutations
    # negligible, unlike the two-statement traversal-only fixture.
    statements = [token for i in range(30) for token in (100 + i, 200 + i)]
    examples = [
        GrugLmExample.causal(
            tokens=jnp.array([2, 8, *statements, 9, 500 + i, 10, 1]),
            eos_id=1,
            block_cross_document_attention=True,
        )
        for i in range(50)
    ]
    base = replace(
        base,
        components={
            FULL_CORPUS: DirectDatasetComponent(
                datasets={"train": ListAsyncDataset(examples)}
            )
        },
    )
    parent = TwoEpochDataConfig(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    args = (Axis("position", 66), BatchSchedule(4))
    key = jax.random.PRNGKey(1729)
    first = base.train_set(*args, key=key)
    both = parent.train_set(*args, key=key)
    left = asyncio.run(first.get_batch(list(range(50))))
    right = asyncio.run(both.get_batch(list(range(50))))
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)
    assert asyncio.run(first.async_len()) == 50
    assert asyncio.run(both.async_len()) == 100
    resumed = list(trainer_loader(first).iter_from_step(9))
    original = list(trainer_loader(both).iter_from_step(9))
    assert len(resumed) == 4
    for a, b in zip(
        jax.tree.leaves(resumed[:3]), jax.tree.leaves(original[:3]), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        resumed[-1].tokens.array[:2], original[3].tokens.array[:2]
    )
    assert not np.asarray(resumed[-1].tokens.array[2:]).any()
    assert not np.asarray(resumed[-1].loss_weight.array[2:]).any()
    assert (
        sum(
            np.any(np.asarray(batch.loss_weight.array), axis=1).sum()
            for batch in resumed
        )
        == 14
    )
