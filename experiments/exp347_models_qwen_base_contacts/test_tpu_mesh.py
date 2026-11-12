"""Checkpoint recovery regression on four emulated CPU devices.

Run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4.
Single-device probes cannot expose an optimizer counter on the wrong mesh.
"""

from pathlib import Path
from unittest import skipUnless

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_checkpoint import replicated_adam_step, restore_arrays, save_arrays
from tpu_optimizer import adamw_update
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
