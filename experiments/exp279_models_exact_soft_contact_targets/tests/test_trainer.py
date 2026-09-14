# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Use actual Levanter batching, accumulation and native checkpoint restore."""

import json
from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from levanter.checkpoint import save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.layers.attention import AttentionBackend
from levanter.tracker import current_tracker
from levanter.tracker.tracker import NoopConfig, NoopTracker
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from test_loss import example_and_oracle

from experiments.exp279_models_exact_soft_contact_targets.checkpoints import (
    validate_restore,
)
from experiments.exp279_models_exact_soft_contact_targets.model import (
    ContactQwen3Config,
)
from experiments.exp279_models_exact_soft_contact_targets.recipe import (
    PHASES,
    optimizer_for_phase,
)


def tiny_config():
    return ContactQwen3Config(
        max_seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        use_qk_norm=True,
        attn_backend=AttentionBackend.VANILLA,
        loss_block_size=16,
    )


def trainer_config(tmp_path, *, microbatch=1, resume=None):
    config = TrainerConfig(
        id="exp279-test",
        tracker=NoopConfig(),
        require_accelerator=False,
        mp=jmp.get_policy("f32"),
        train_batch_size=4,
        per_device_parallelism=microbatch,
        per_device_eval_parallelism=1,
        num_train_steps=4,
        log_jaxprs=False,
        log_xla_hlo=False,
        log_dir=tmp_path / "logs",
        load_checkpoint=resume is not None,
        load_checkpoint_path=resume,
    )
    config.initialize()
    return config


def model_loss(model, example, *, key=None):
    return model.compute_next_token_loss(example, key=key)


def assert_state_equal(a, b, *, tolerance=0):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=tolerance, rtol=tolerance)


def test_stock_trainer_resume_restores_rng_optimizer_and_data_position(tmp_path, vocab):
    examples = [
        example_and_oracle(
            vocab, edges=((143, 144),) if i % 2 else ((143, 145), (145, 144))
        )[0]
        for i in range(20)
    ]
    dataset = ListAsyncDataset(examples)
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(4)
    )
    config = trainer_config(tmp_path)
    optimizer = optax.adam(1e-3)
    with Trainer(config, optimizer, model_loss, add_default_hooks=False) as trainer:
        state = trainer.initial_state(jax.random.PRNGKey(5), model=model)
        loader = trainer.data_loader(dataset)
        iterator = loader.iter_from_step(0)
        state = trainer.train_step(state, next(iterator)).state
        checkpoint = str(tmp_path / "step-0")
        save_checkpoint(state.saveable_state, 0, checkpoint, is_temporary=False)
        state_snapshot = jax.tree.map(lambda x: np.array(x), state)
        expected_batch = next(iterator)
        expected = trainer.train_step(state, expected_batch).state
    resumed_config = trainer_config(tmp_path, resume=checkpoint)
    with Trainer(
        resumed_config, optimizer, model_loss, add_default_hooks=False
    ) as trainer:
        resumed = trainer.initial_state(jax.random.PRNGKey(999), model=model)
        assert int(resumed.step) == 1
        assert_state_equal(resumed, state_snapshot)
        loader = trainer.data_loader(dataset)
        resumed_batch = next(loader.iter_from_step(int(resumed.step)))
        assert_state_equal(resumed_batch, expected_batch)
        actual = trainer.train_step(resumed, resumed_batch).state
        assert_state_equal(actual, expected)


def test_accumulation_matches_reference_mean_of_microbatch_means(tmp_path, vocab):
    examples = [
        example_and_oracle(vocab, edges=edges)[0]
        for edges in ((), ((143, 144),), ((143, 144), (143, 145)), ((144, 145),))
    ]
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(4)
    )
    config = trainer_config(tmp_path, microbatch=1)
    with Trainer(
        config, optax.sgd(1e-3), model_loss, add_default_hooks=False
    ) as trainer:
        batch = next(trainer.data_loader(ListAsyncDataset(examples)).iter_from_step(0))
        loss, grad, _ = eqx.filter_jit(trainer._compute_gradients_microbatched)(
            trainer.loss_fn, model, batch
        )
        # The stock loop averages normalized microbatch losses, not a single
        # global scored-token mean when packs contain different padding. Keep
        # microbatch/mesh settings identical across arms; don't redefine CE.
        size = config.microbatch_size or len(examples)
        groups = [
            jax.tree.map(
                lambda *xs: hax.stack(config.TrainBatch.resize(size), xs),
                *examples[start : start + size],
                is_leaf=hax.is_named_array,
            )
            for start in range(0, len(examples), size)
        ]

        def manual(m):
            return sum(model_loss(m, group).array for group in groups) / len(groups)

        expected_loss, expected_grad = eqx.filter_jit(
            eqx.filter_value_and_grad(manual)
        )(model)
        np.testing.assert_allclose(loss, expected_loss, atol=2e-6, rtol=2e-6)
        assert_state_equal(grad, expected_grad, tolerance=3e-6)


def test_recovery_allows_only_new_skip_buffers(tmp_path, vocab):
    model = tiny_config().build(
        hax.Axis("vocab", vocab.size), key=jax.random.PRNGKey(4)
    )
    with current_tracker(NoopTracker()):
        state = TrainerState.init(
            optimizer_for_phase(PHASES["base"]).build(217801),
            model,
            key=jax.random.PRNGKey(5),
        )
        state = replace(state, step=jnp.asarray(217801))
        checkpoint = str(tmp_path / "step-217800")
        save_checkpoint(state.saveable_state, 217800, checkpoint, is_temporary=False)

        def recovery_template():
            return TrainerState.init(
                optimizer_for_phase(PHASES["recovery"]).build(333961),
                model,
                key=jax.random.PRNGKey(5),
            ).saveable_state

        template = eqx.filter_eval_shape(recovery_template)
        validate_restore(template, checkpoint, adding_skip_state=True)
        config = replace(
            trainer_config(tmp_path, resume=checkpoint), allow_partial_checkpoint=True
        )
        with Trainer(
            config,
            optimizer_for_phase(PHASES["recovery"]).build(333961),
            model_loss,
            add_default_hooks=False,
        ) as trainer:
            restored = trainer.initial_state(jax.random.PRNGKey(999), model=model)
            assert_state_equal(restored.model, state.model)
            without_skip = restored.opt_state._replace(
                inner_state=restored.opt_state.inner_state.inner_opt_state
            )
            assert_state_equal(without_skip, state.opt_state)
            np.testing.assert_array_equal(restored.training_key, state.training_key)
            assert int(restored.step) == 217801
            assert int(restored.opt_state.inner_state.count) == 0
        with pytest.raises(ValueError, match="missing"):
            validate_restore(template, checkpoint)
        manifest_path = tmp_path / "step-217800" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["arrays"] = [
            entry
            for entry in manifest["arrays"]
            if not entry["path"].startswith("model/")
        ]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="missing"):
            validate_restore(template, checkpoint, adding_skip_state=True)
