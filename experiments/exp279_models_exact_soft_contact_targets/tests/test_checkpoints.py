# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The boolean restore repair that recovery-phase checkpoints depend on."""

import jax
import jax.numpy as jnp
import levanter.tensorstore_serialization as tensorstore_serialization
import pytest

from experiments.exp279_models_exact_soft_contact_targets.checkpoints import (
    apply_bool_restore_fix,
)


@pytest.fixture(autouse=True)
def _patched():
    apply_bool_restore_fix()


def _reduce(value):
    return tensorstore_serialization._restore_replica_axis(value)


def test_upstream_rejects_bool_without_the_repair():
    """The defect is real: the stock reduction cannot bitcast a bool leaf."""
    upstream = tensorstore_serialization._restore_replica_axis.__closure__[0].cell_contents
    with pytest.raises(TypeError, match="bitcast_convert_type"):
        upstream(jnp.zeros((2, 4), dtype=jnp.bool_))


@pytest.mark.parametrize("contributor", [0, 1, 2])
def test_bool_leaf_restores_the_contributing_replica(contributor):
    """Exactly one replica holds the value; the others hold False."""
    stored = jnp.array([True, False, True, True])
    replicas = jnp.zeros((3, stored.size), dtype=jnp.bool_).at[contributor].set(stored)
    assert jnp.array_equal(_reduce(replicas), stored)


def test_bool_all_false_stays_false():
    replicas = jnp.zeros((4, 5), dtype=jnp.bool_)
    assert not bool(jnp.any(_reduce(replicas)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32, jnp.uint8, jnp.bfloat16])
def test_non_bool_leaves_keep_the_upstream_result(dtype):
    """The repair must not alter the byte-sum path for ordinary leaves."""
    stored = jnp.arange(6, dtype=dtype)
    replicas = jnp.zeros((3, stored.size), dtype=dtype).at[1].set(stored)
    assert jnp.array_equal(_reduce(replicas), stored)


def test_repair_is_idempotent():
    """Re-applying must not stack wrappers or re-enter the broken path."""
    first = tensorstore_serialization._restore_replica_axis
    apply_bool_restore_fix()
    assert tensorstore_serialization._restore_replica_axis is first
    assert jnp.array_equal(
        _reduce(jnp.zeros((2, 3), dtype=jnp.bool_).at[0].set(True)),
        jnp.ones((3,), dtype=jnp.bool_),
    )


def test_skipstep_valid_mask_is_the_boolean_leaf():
    """Guard the premise: the repair matters because this leaf is bool."""
    import optax
    from levanter.optim.skipstep import SkipStepConfig

    wrapped = SkipStepConfig().wrap(optax.sgd(0.1))
    state = wrapped.init(jnp.zeros(3))
    assert state.valid_mask.dtype == jnp.bool_
