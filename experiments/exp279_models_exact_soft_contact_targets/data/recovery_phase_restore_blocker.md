# exp279 blocker: recovery-phase checkpoints cannot be restored

Diagnosed 2026-09-21 after a05's post-preemption retry, a06 and a07 all died
about four minutes into startup with exit 1.

## The exception

Captured by streaming `iris job logs -f` during a07; Iris keeps only a
fixed-size stderr tail, and the per-second JAX coordination warnings flush the
traceback out of it, so this is not recoverable after a job ends.

```
File ".../levanter/tensorstore_serialization.py", line 1058, in _restore_replica_axis
TypeError: lax.bitcast_convert_type does not support bool or complex values unless
the operand and destination types match. Got operand dtype=bool, new_dtype=dtype('uint8').
```

Call path: `trainer.initial_state` -> `load_checkpoint_or_initialize` ->
`load_checkpoint` -> `tree_deserialize_leaves_tensorstore` -> `_restore_ocdbt`
-> `_deserialize_leaves` -> `_finish_leaf` -> `_restore_replica_axis`.

```python
def _restore_replica_axis(value: jax.Array) -> jax.Array:
    bits = jax.lax.bitcast_convert_type(value, jnp.uint8)   # fails on dtype=bool
```

## Why it only bites now

`PHASES` sets `skip_bad_steps=False` for base and `True` for recovery and
final. SkipStep adds five buffers to the optimizer state, one of which,
`SkipStepState.valid_mask`, is `jnp.bool_` (`levanter/optim/skipstep.py:112`).

* Base-phase checkpoints carry no bool leaf, so they restore normally. Every
  successful resume of this run so far, a02 through a05, restored a base-phase
  checkpoint.
* Recovery-phase checkpoints carry `_skipstep_valid_mask`, and restoring one
  hits the bitcast above.

The base -> recovery transition works because `build_config` sets
`allow_partial_checkpoint=True` when `phase_name == "recovery" and start ==
phase.start`: at that one update the five SkipStep buffers are absent from the
base checkpoint and are freshly initialized. That transition is covered by
`checkpoints.validate_restore(adding_skip_state=True)`. Resuming *within* the
recovery phase is a different path and was never exercised, because the
recovery phase had not been interrupted before.

## Scope

Not a corrupt checkpoint. `step-235057` matches the known-good `step-232320`
in total size (16.44 GiB) and has `metadata.json`, `manifest.json` and
`manifest.ocdbt`.

Not fixed upstream: `_restore_replica_axis` is byte-identical in the marin
checkout's levanter at commit fc8a97b4 (2026-09-15), so upgrading does not
help.

Two consequences:

1. Any interruption during the recovery or final phase is unrecoverable
   without discarding every recovery-phase checkpoint and restarting from
   `step-217800`.
2. The recovery -> final transition is blocked outright. `production.main`
   discovers the latest checkpoint and submits the next phase's worker with
   `--resume-latest`; at update 333961 that checkpoint is recovery-written and
   carries the bool leaf, and `allow_partial_checkpoint` is False for `final`.
   So the run cannot reach the final phase even if recovery completes
   uninterrupted.

The recovery phase is 116,160 updates, about 4.7 days, and the final phase
29,040 more. Observed time between terminal failures on this run is 18 to 58
hours.

## Current state

Blocked, awaiting a decision. Recovery-phase checkpoints present:
`step-232320`, `step-235057`. Newest restorable checkpoint: `step-217800`
(permanent, base-phase final, update 217801).
