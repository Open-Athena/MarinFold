# exp232 CoreWeave Continuation Operations

> This document governs the exp232 continuation managed with the
> `run-training-sweep-cw` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- Maintain at most one active writer for each continuation trial's W&B ID and
  checkpoint root. Placement changes preserve both identities.
- Use whole-node GPU gangs at `--priority batch` through the `marin` controller,
  with an exact `--target-cluster` and `--user eczech` on every Iris command.
- Route W&B only to `open-athena/MarinFold`; stop if authenticated writes fail.
- Smoke outputs are temporary and separate from production outputs and identity.
- Use the existing exp232 token caches, exp199 validation cache, and exact
  full-state source checkpoints. Never rewrite source data or checkpoints.

## Sweep Definition

- Entry point and two-source catalog:
  `experiments/exp232_sweep_cv1_decontam/exp232_continue_cw.py`.
- This operation validates both continuation sources through the peak-LR hold and
  cooldown with full-rate augmentation. Production continuation is not launched
  by this operation.
- The original exp232 sweep policy and execution evidence remain in
  `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`; this operation
  has separate W&B identities, checkpoint outputs, Iris roots, and SQLite state.

## Operator Choices

- Stop after both smoke trials finish successfully or on the first blocker.
- GPU limit: 16 H100s for this validation, one node per trial.
- Clusters: `cw-us-east-02a` and `cw-rno2a`, one smoke on each.
- Iris user: `eczech`; priority: `batch` only.
- Durable SQLite:
  `scratch/exp232_continue_cw_smoke/exp232_continue_cw_smoke.sqlite`.
- This tracked file is the sole Operations document for the continuation smoke.

## Operating Policy

- Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and
  `pending_target_limit=1`. Observe more frequently while establishing the short
  smoke runs.
- Treat one-trial failure as isolated until cross-trial evidence suggests a shared
  cause. Stop and investigate rather than blindly retrying correlated failures.
- Let Iris recover ordinary preemptions. Reslice only after stopping and verifying
  the exact active root; preserve the same logical trial identity.
- A smoke trial succeeds only when training finishes, its W&B history proves the
  intended LR shape, and no augmentation invariant fails.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-02a` | H100 | 1 | 8 | eligible | — |
| `cw-rno2a` | H100 | 1 | 8 | eligible | — |

## Change Record

- 2026-08-18 15:44 UTC — The s01 `m2-p06` smoke showed that the stock
  linear-decay endpoint fell one unexecuted update after training. Replaced the
  continuation decay with an inclusive endpoint, invalidated s01, and required
  both sources to pass again as s02 before production.
