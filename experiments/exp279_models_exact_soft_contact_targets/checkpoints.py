# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight full-state restores, especially the SkipStep state transition."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import levanter.tensorstore_serialization as tensorstore_serialization
from levanter.checkpoint_manifest import read_manifest
from levanter.tensorstore_serialization import _flatten_serializable_leaves
from levanter.tracker import current_tracker
from levanter.tracker.tracker import NoopTracker
from levanter.trainer_state import TrainerState

SKIP_BUFFERS = frozenset(
    {
        "_skipstep_losses",
        "_skipstep_grad_norms",
        "_skipstep_valid_mask",
        "_skipstep_current_idx",
        "_skipstep_count",
    }
)


def validate_restore(
    template, checkpoint: str, *, adding_skip_state: bool = False
) -> None:
    """Require every expected array and shape; allow only the five new buffers.

    Uses the pinned serializer's path encoding, also tested against a real
    checkpoint. Reads the small manifest, not the model weights. The subsequent
    stock load still verifies the actual stored arrays.
    """
    manifest = read_manifest(checkpoint)
    if manifest is None:
        raise ValueError("exp279 resumes require a checkpoint with an array manifest")
    # The serializer excludes ShapeDtypeStruct leaves. Zero-stride NumPy views
    # preserve their shapes/dtypes without allocating any parameter-sized data.
    template = jax.tree.map(
        lambda x: (
            np.broadcast_to(np.zeros((), x.dtype), x.shape)
            if isinstance(x, jax.ShapeDtypeStruct)
            else x
        ),
        template,
    )
    paths, leaves = _flatten_serializable_leaves(template)
    expected = dict(zip(paths, leaves, strict=True))
    stored = {entry.path: entry for entry in manifest.arrays}
    allowed = (
        {path for path in expected if path.rsplit("/", 1)[-1] in SKIP_BUFFERS}
        if adding_skip_state
        else set()
    )
    missing = set(expected) - set(stored)
    if missing != allowed or set(stored) - set(expected):
        raise ValueError(
            f"Checkpoint state differs: missing={missing}, allowed={allowed}, extra={set(stored) - set(expected)}"
        )
    for path in set(expected) & set(stored):
        array, entry = expected[path], stored[path]
        if tuple(array.shape) != entry.shape or str(array.dtype) != entry.dtype:
            raise ValueError(f"Checkpoint array shape/dtype changed: {path}")


def validate_training_restore(config, checkpoint: str) -> None:
    """Build only abstract state shapes; validate before the stock loader runs."""

    def template():
        model = config.model.build(hax.Axis("vocab", 2845), key=jax.random.PRNGKey(0))
        optimizer = config.optimizer.build(config.trainer.num_train_steps)
        state = TrainerState.init(
            optimizer, model, key=jax.random.PRNGKey(0), mp=config.trainer.mp
        )
        return state.saveable_state

    with current_tracker(NoopTracker()):
        shapes = eqx.filter_eval_shape(template)
    validate_restore(
        shapes, checkpoint, adding_skip_state=config.trainer.allow_partial_checkpoint
    )


_BOOL_RESTORE_PATCHED = False


def apply_bool_restore_fix() -> None:
    """Make boolean checkpoint leaves restorable under the pinned serializer.

    `levanter.tensorstore_serialization._restore_replica_axis` reduces a leaf's
    replica axis by bitcasting it to uint8 and summing the bytes.
    `lax.bitcast_convert_type` rejects bool operands, so restoring any
    checkpoint that holds a boolean array raises TypeError. SkipStep's
    `SkipStepState.valid_mask` is `jnp.bool_`, and the recovery and final
    phases enable SkipStep, so every checkpoint those phases write is
    unreadable -- including the one the recovery -> final transition must
    restore. The base phase carries no boolean leaf, which is why base-phase
    resumes always worked.

    The original reduction is exact because exactly one replica contributes
    each value and the rest hold zero bytes. For bool that makes a logical OR
    over the replica axis produce the identical result: the contributing
    replica's value, since False is the identity of OR. Non-boolean leaves keep
    the upstream path untouched.

    This repairs a restore-path defect only. It reads no training
    hyperparameter and changes no update, so the recipe under test is
    unaffected. The same defect is present in current upstream levanter, so
    upgrading the pin is not an alternative.
    """
    global _BOOL_RESTORE_PATCHED
    if _BOOL_RESTORE_PATCHED:
        return
    upstream = tensorstore_serialization._restore_replica_axis

    def _restore_replica_axis(value: jax.Array) -> jax.Array:
        if value.dtype == jnp.bool_:
            return jnp.any(value, axis=0)
        return upstream(value)

    tensorstore_serialization._restore_replica_axis = _restore_replica_axis
    # The jitted reducer resolves the module global when it is first built, so
    # drop any entry cached before this patch was applied.
    tensorstore_serialization._replica_reducer.cache_clear()
    _BOOL_RESTORE_PATCHED = True
