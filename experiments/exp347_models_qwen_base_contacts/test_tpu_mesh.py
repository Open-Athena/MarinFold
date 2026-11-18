"""Checkpoint recovery regression on four emulated CPU devices.

Run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4.
Single-device probes cannot expose an optimizer counter on the wrong mesh.
"""

from pathlib import Path
from unittest import skipUnless

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tpu_checkpoint import replicated_adam_step, restore_arrays, save_arrays
from tpu_model import abstract_model, configuration, convert_parameters, initial_rest
from tpu_optimizer import adamw_update
from tpu_training import document_batch, training_functions
from validate_tpu_math import probe_config
from validate_tpu_runtime import verify_roundtrip


@skipUnless(
    jax.default_backend() == "cpu" and jax.device_count() == 4,
    "Requires four emulated CPU devices; see the module invocation.",
)
def test_four_device_checkpoint_and_adam_update(tmp_path: Path) -> None:
    """Restore the complete mesh state and reproduce its next optimizer update."""
    mesh = Mesh(np.asarray(jax.devices()), ("fsdp",))
    sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    weight = jax.device_put(np.arange(32, dtype=np.float32).reshape(8, 4), sharding)
    state = {
        "params": {"weight": weight},
        "mu": {"weight": jax.device_put(np.full((8, 4), 0.001, np.float32), sharding)},
        "nu": {"weight": jax.device_put(np.full((8, 4), 0.002, np.float32), sharding)},
        "adam_step": replicated_adam_step(59319, mesh),
    }
    uri = str(tmp_path / "state")
    save_arrays(uri, state)
    restored = restore_arrays(uri, state)
    verify_roundtrip(state, restored)
    assert restored["adam_step"].sharding.device_set == set(jax.devices())
    gradient = {"weight": jax.device_put(np.full((8, 4), 0.01, np.float32), sharding)}

    @jax.jit
    def step(current: dict) -> dict:
        params, mu, nu, count, _ = adamw_update(
            current["params"],
            current["mu"],
            current["nu"],
            gradient,
            current["adam_step"],
            1.9e-5,
        )
        return {"params": params, "mu": mu, "nu": nu, "adam_step": count}

    original_next, restored_next = step(state), step(restored)
    verify_roundtrip(original_next, restored_next)
    assert int(restored_next["adam_step"]) == 59320


@skipUnless(
    jax.default_backend() == "cpu" and jax.device_count() == 4,
    "Requires four emulated CPU devices; see the module invocation.",
)
def test_four_device_model_gradient_accumulation_sharding() -> None:
    """Retain FSDP gradients/moments and match an unaccumulated model update."""
    torch.set_num_threads(4)
    torch.manual_seed(347)
    source = probe_config()
    hf = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**source)).eval()
    config = configuration(source, length=128, batch=4, dtype="float32")
    mesh, model = abstract_model(config)
    graph, template, rest = nnx.split(model, nnx.Param, ...)
    rest = initial_rest(rest)
    params = convert_parameters(
        {name: value.detach().numpy() for name, value in hf.named_parameters()},
        source,
        config,
        template,
    ).to_pure_dict()
    layouts = jax.tree.map(lambda array: array.sharding, params)
    rng = np.random.default_rng(914)
    rows = [
        {"contacts_v1": rng.integers(0, 256, length).tolist()}
        for length in (129, 81, 99, 66, 103, 70, 83, 115)
    ]
    count = sum(len(row["contacts_v1"]) - 1 for row in rows)
    batch_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))

    def batch(documents: list[dict]) -> dict:
        return jax.tree.map(
            lambda array: jax.device_put(array, batch_sharding),
            document_batch(documents, "contacts_v1", 128),
        )

    def assert_layouts(values: dict) -> None:
        for array, layout in zip(
            jax.tree.leaves(values), jax.tree.leaves(layouts), strict=True
        ):
            assert array.sharding.is_equivalent_to(layout, array.ndim)

    gradient, accumulate, update = training_functions(graph, template, rest, config)
    loss_a, grads = gradient(params, rest, batch(rows[:4]), count)
    assert_layouts(grads)
    loss_b, grads = accumulate(params, rest, batch(rows[4:]), count, grads)
    assert_layouts(grads)
    full_loss, full_grads = gradient(params, rest, batch(rows), count)
    np.testing.assert_allclose(loss_a + loss_b, full_loss, atol=2e-6, rtol=2e-6)
    for accumulated, full in zip(
        jax.tree.leaves(grads), jax.tree.leaves(full_grads), strict=True
    ):
        np.testing.assert_allclose(accumulated, full, atol=2e-6, rtol=2e-6)
    state = {
        "params": params,
        "mu": jax.tree.map(lambda p: jnp.full_like(p, 0.001), params),
        "nu": jax.tree.map(lambda p: jnp.full_like(p, 0.002), params),
        "adam_step": replicated_adam_step(59319, mesh),
    }
    reference, _ = update(jax.tree.map(jnp.copy, state), full_grads, 1.9e-5)
    actual, _ = update(state, grads, 1.9e-5)
    for key in ("params", "mu", "nu"):
        assert_layouts(actual[key])
    for result, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-6)
    assert int(actual["adam_step"]) == 59320
