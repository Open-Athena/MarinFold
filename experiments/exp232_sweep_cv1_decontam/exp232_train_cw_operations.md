# exp232 CoreWeave Selected Training Operations

> This document governs training of the two selected exp232 sweep survivors with the
> `run-training-sweep-cw` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- Maintain at most one active writer for each training trial's W&B ID and
  checkpoint root. Placement changes preserve both identities.
- Use whole-node GPU gangs at `--priority batch` through the `marin` controller,
  with an exact `--target-cluster` and `--user eczech` on every Iris command.
- Route W&B only to `open-athena/MarinFold`; stop if authenticated writes fail.
- Smoke outputs are temporary and separate from production outputs and identity.
- Use the existing exp232 token caches, exp199 validation cache, and exact
  full-state source checkpoints. Never rewrite source data or checkpoints.
- Continue the original exp232 augmentation schedule from the restored global
  step, reach 100% at step 145199, and remain at 100% thereafter.

## Training Definition

- Entry point and two-source catalog:
  `experiments/exp232_sweep_cv1_decontam/exp232_train_cw.py`.
- The two sources are the selected `m2-p06-aug` and `m1-p02-aug` sweep
  checkpoints immediately before their original cooldowns.
- The earlier s01-s04 operation validated checkpoint restoration and the
  inclusive-zero LR schedule. The continuous augmentation schedule and one-file
  serialization path are validated locally before production.
- The original exp232 sweep policy and execution evidence remain in
  `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`; this operation
  has separate W&B identities, checkpoint outputs, Iris roots, and SQLite state.

## Operator Choices

- The completed validation used 16 H100s, one node per source across
  `cw-us-east-02a` and `cw-rno2a`.
- Production compute scope is H100 only, on `cw-us-east-02a` and `cw-rno2a`.
  GB200 is out of scope unless the operator explicitly broadens it.
- Hard limit of 16 nodes per dispatch (128 H100s). Prefer 16-node gangs;
  use a smaller eligible gang only for a concrete scheduling, recovery, or
  measured wall-clock reason, never merely because capacity is busy.
- Two logical trials, one writer each, so the sweep maximum is two 16-node
  gangs: 32 nodes / 256 H100s. Never duplicate a trial to consume free GPUs.
- Iris user: `eczech`; priority: `batch` only.
- Take utilization snapshots with `--peer cw-us-east-02a --peer cw-rno2a`. The
  default peer set includes the GB200 peer, whose accounting intermittently fails
  validation and aborts the whole snapshot; that peer is ineligible here anyway.
- Production SQLite:
  `scratch/exp232_train_cw/exp232_train_cw.sqlite`.
- Historical validation SQLite:
  `scratch/exp232_continue_cw_smoke/exp232_continue_cw_smoke.sqlite`.
- This tracked file remains the sole Operations document for selected training.

## Operating Policy

- Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and
  `pending_target_limit=1`. Observe more frequently while establishing the
  production runs.
- Treat one-trial failure as isolated until cross-trial evidence suggests a shared
  cause. Stop and investigate rather than blindly retrying correlated failures.
- Let Iris recover ordinary preemptions. Reslice only after stopping and verifying
  the exact active root; preserve the same logical trial identity.
- Training succeeds only when W&B finishes, the final permanent checkpoint is
  verified, and the intended LR and augmentation schedules remain intact.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-08a` | GB200 | 2/4/8/16 | 8/16/32/64 | ineligible | Operator scoped selected training to H100 only |
| `cw-us-east-02a` | H100 | 2/4/8/16 | 16/32/64/128 | eligible | — |
| `cw-rno2a` | H100 | 2/4/8/16 | 16/32/64/128 | eligible | — |

## Change Record

- 2026-08-18 15:44 UTC — The s01 `m2-p06` smoke showed that the stock
  linear-decay endpoint fell one unexecuted update after training. Replaced the
  continuation decay with an inclusive endpoint, invalidated s01, and required
  both sources to pass again as s02 before production.
- 2026-08-18 15:54 UTC — The s02 jobs exposed missing Draccus registration for
  the new schedule while logging W&B configuration. Registered the schedule as
  `linear_inclusive`, invalidated s02 before training, and required a clean s03.
- 2026-08-18 15:59 UTC — The s03 jobs showed that a class created under
  `__main__` still loses its registry identity across the Iris serialization
  boundary. Moved it to an importable module and required a subprocess
  serialization check plus a clean s04.
- 2026-08-18 16:36 UTC — Restored augmentation continuity with the original
  exp232 global-step ramp instead of jumping from about 80% to 100%. Renamed the
  selected-training entry point to `exp232_train_cw.py` and folded the inclusive
  LR schedule into that single script with a canonical serialization identity.
- 2026-08-18 20:55 UTC — Operator restricted selected-training compute to H100 on
  `cw-us-east-02a` and `cw-rno2a` with a hard 16-node/128-GPU per-dispatch limit.
  Marked GB200 `cw-us-east-08a` ineligible and stated the two-gang (32-node/256-H100)
  sweep maximum in Operator Choices. No training semantics changed.
- 2026-08-19 03:30 UTC — Utilization snapshots intermittently exited 2 with
  `peer cw-us-east-08a.backends[0] gb200 accounting disagrees` (held exceeded total),
  producing no capacity reading. Narrowed routine snapshots to the two eligible H100
  peers so a GB200 accounting fault cannot block H100 placement decisions.
