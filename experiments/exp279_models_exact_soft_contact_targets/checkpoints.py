# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight full-state restores, especially the SkipStep state transition."""

import equinox as eqx
import haliax as hax
import jax
import numpy as np
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
