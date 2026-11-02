"""Exercise finite gradients at the exact-epoch padding tail with real Qwen3."""

from dataclasses import asdict, replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

from experiments.exp364_models_mpnn_only_lr_sweep.padding_safe_model import (
    padding_safe_model,
)


def tiny_models():
    config = Qwen3Config(
        num_layers=1,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
    )
    safe_config = padding_safe_model(config)
    assert asdict(safe_config) == asdict(config)
    vocab = hax.Axis("vocab", 32)
    key = jax.random.PRNGKey(13)
    original = Qwen3LMHeadModel.init(vocab, config, key=key)
    safe = safe_config.model_type.init(vocab, safe_config, key=key)
    example = LmExample.causal(hax.named(jnp.arange(8), hax.Axis("position", 8)))
    return original, safe, example


def loss(model, example):
    return model.compute_next_token_loss(example).scalar()


def test_real_examples_preserve_loss_gradients_and_checkpoint_leaves():
    original, safe, example = tiny_models()
    for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(safe), strict=True):
        np.testing.assert_array_equal(a, b)
    ref_loss, ref_grad = eqx.filter_value_and_grad(loss)(original, example)
    new_loss, new_grad = eqx.filter_value_and_grad(loss)(safe, example)
    np.testing.assert_array_equal(ref_loss, new_loss)
    for a, b in zip(jax.tree.leaves(ref_grad), jax.tree.leaves(new_grad), strict=True):
        np.testing.assert_array_equal(a, b)


def test_fully_padded_example_has_zero_loss_and_zero_finite_gradients():
    original, safe, example = tiny_models()
    example = replace(example, loss_weight=hax.zeros_like(example.loss_weight))
    old_loss, old_grad = eqx.filter_value_and_grad(loss)(original, example)
    new_loss, new_grad = eqx.filter_value_and_grad(loss)(safe, example)
    assert float(old_loss) == float(new_loss) == 0
    assert any(not np.isfinite(x).all() for x in jax.tree.leaves(old_grad))
    for gradient in jax.tree.leaves(new_grad):
        assert np.isfinite(gradient).all()
        assert not np.asarray(gradient).any()
