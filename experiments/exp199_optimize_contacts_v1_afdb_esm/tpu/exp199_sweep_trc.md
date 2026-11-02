# exp199 Sweep Operations

> This document governs a training sweep managed with the `run-training-sweep-trc`
> skill. Read it in full at the start of every heartbeat before inspecting code,
> SQLite, W&B, or Iris.

## Invariants

- **Utilization is not capacity.** The `utilization.py` snapshot reports only
  which slices are ready and in use at that instant. It never reveals what the
  fleet holds or could provision, and no command does — submission is the only
  measurement of capacity. A target absent from the snapshot is a target with no
  ready slice right now, not one that is nonexistent, impossible, or
  unavailable. Never say otherwise in a heartbeat, never let the snapshot alone
  drive a reslice, and never let it change target eligibility; only a verified
  terminal `unschedulable` result does that.
- Never copy or move data or checkpoints between regions.
- One live regional run per logical trial; at most one active dispatch per run.
- Stop the current dispatch before submitting its replacement.
- Production checkpoints are isolated only by the regional `MARIN_PREFIX` root;
  never let two live dispatches share a checkpoint path.
- `MARIN_PREFIX` must sit below the region root, enforced by
  `_validate_launch_prefix` and `_regional_prefix_guard`.
- No secrets in this document, in SQLite, or in recorded commands.
- Never change the Iris priority band to solve scarcity.
- Iris never proves training progress; only a new W&B `run_progress` high-water
  mark does.
- W&B `tpu=`/`region=` tags and Iris metadata are stamped at submission and are
  not authoritative for current placement. The SQLite dispatch row is.
- Never parse W&B or Iris names for identity; attempt numbers come from SQLite.

## Sweep Definition

Entry point and trial catalog: `exp199_sweep_trc.py` (`TRIALS`, 24 keys
`m1-p01-base` … `m2-p06-aug`). One invocation launches exactly one trial.

Selection is by environment: `TRIAL`, `REGION`, `TPU`, plus `MARIN_PREFIX`,
`WANDB_ENTITY`, `WANDB_PROJECT` from `~/marin.env`. `--version` is a CalVer whose
numeric suffix becomes the sweep subversion; this sweep uses `2026.08.07.1` →
`s01`. `--run` executes; without it the plan is only lowered.

Regional constraints not evident from code:

- All four candidate regions were verified on 2026-08-07 to hold the AFDB, ESM,
  and validation caches and all six #117 seed checkpoints (36/36 present).
- Seed checkpoints are region-local copies; their names embed the #117 source
  region and carry no meaning for placement here.
- `_validate_placement` chip bounds were widened for this sweep at operator
  direction: 32–512 chips for `v6e`/`v5e`, 16–256 for `v5p`. Per-family region
  policy is unchanged.
- A same-region restart or reslice resumes from the regional checkpoint. A
  cross-region relocation is a separate regional run that starts from step zero.

## Operator Choices

- **Time limit:** two weeks from 2026-08-07.
- **Regional replicas per trial:** 2, but only for a trial whose runs all have
  zero progress. A trial with any progress stays at 1 replica. Relocation after
  `relocate_after` remains allowed and replaces the active region.
- **Global chip cap:** 8192 concurrent chips across the sweep.
- **Approved regions and families:** `v6e` in `europe-west4`, `us-east1`,
  `us-east5`; `v5e` in `europe-west4`, `us-west4`; `v5p` in `us-east5`.
- **Approved chip range:** 32–512 (`v6e`, `v5e`), 16–256 (`v5p`).
- **Exclusions:** none beyond the grid below.
- **Priority band:** the cluster default in effect at submission; unchanged
  without explicit operator approval.
- **Operations document:** this file.
- **Data:** `scratch/exp199_optimize_contacts_v1_afdb_esm/exp199_sweep_trc.sqlite`.

## Operating Policy

- `heartbeat_every` = 30 minutes
- `reslice_after` = 1 hour
- `restart_after` = 3 hours
- `relocate_after` = 3 days
- `pending_target_limit` = 2

Iris job names use `<wandb-id>-<tpu>-aNN`; the attempt number comes from SQLite
and is never recovered by parsing the name.

**Failures.** Classify before retrying. An isolated failure is replaced on the
same region and slice from the regional checkpoint. Failures that recur after
replacement, or cluster across independent trials, regions, or targets, pause
replacement pending investigation of a shared cause. Preemption alone is not a
failure: Iris owns preemption recovery, so never stop or resubmit on a W&B
`crashed` state — act on progress and the timeouts above.

**Unschedulable.** A verified `unschedulable` result marks only its exact target
ineligible and is never generalized to a region or family. On a previously
working target it is an anomaly to investigate, not routine pruning.

**Racing.** A zero-progress trial may hold two regional runs in distinct
validated regions. Add the second replica only in a region actually placing
work, and only onto a target with pending headroom; a replica starts from zero
and transfers nothing. Never add a replica to a trial that already has
progress — that duplicates real work. Let both runs continue while the trial is
incomplete; stop the nonterminal sibling only after the other reaches
`run_progress >= 1` with its checkpoint reachable, and record the sibling as a
race loss. Relocation after `relocate_after` remains available and replaces the
region.

**Completion.** A trial finishes when `run_progress >= 1` and its expected
checkpoint is reachable. W&B `finished` alone is insufficient.

### Target Grid

| Region | Bucket | Slice | Chips | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `europe-west4` | `marin-eu-west4` | `v6e-32` | 32 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v6e-64` | 64 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v6e-128` | 128 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v6e-256` | 256 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v6e-512` | — | `ineligible` | topology undefined in fray |
| `us-east1` | `marin-us-east1` | `v6e-32` | 32 | `eligible` | — |
| `us-east1` | `marin-us-east1` | `v6e-64` | 64 | `eligible` | — |
| `us-east1` | `marin-us-east1` | `v6e-128` | 128 | `eligible` | — |
| `us-east1` | `marin-us-east1` | `v6e-256` | 256 | `eligible` | — |
| `us-east1` | `marin-us-east1` | `v6e-512` | — | `ineligible` | topology undefined in fray |
| `us-east5` | `marin-us-east5` | `v6e-32` | 32 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-64` | 64 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-128` | 128 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-256` | 256 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v6e-512` | — | `ineligible` | topology undefined in fray |
| `europe-west4` | `marin-eu-west4` | `v5litepod-32` | 32 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-64` | 64 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-128` | 128 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-256` | 256 | `eligible` | — |
| `europe-west4` | `marin-eu-west4` | `v5litepod-512` | — | `ineligible` | topology undefined in fray |
| `us-west4` | `marin-us-west4` | `v5litepod-32` | 32 | `eligible` | — |
| `us-west4` | `marin-us-west4` | `v5litepod-64` | 64 | `eligible` | — |
| `us-west4` | `marin-us-west4` | `v5litepod-128` | 128 | `eligible` | — |
| `us-west4` | `marin-us-west4` | `v5litepod-256` | 256 | `eligible` | — |
| `us-west4` | `marin-us-west4` | `v5litepod-512` | — | `ineligible` | topology undefined in fray |
| `us-east5` | `marin-us-east5` | `v5p-32` | 16 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v5p-64` | 32 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v5p-128` | 64 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v5p-256` | 128 | `eligible` | — |
| `us-east5` | `marin-us-east5` | `v5p-512` | 256 | `eligible` | — |

## Change Record

- **2026-08-09T22:14Z — SUSPENDED.** Cause: the operator needed the cluster
  capacity for other work. Change: every live dispatch was stopped — 20 training
  jobs across `us-east1`, `us-east5`, `us-west4`, and `europe-west4`; the nine
  completed trials' jobs had already succeeded. No exp199 job is running and the
  heartbeat loop is disarmed. Effect: 9 of 24 trials are complete and verified;
  the other 15 hold intact regional checkpoints and are resumable. Resumption
  needs no new decisions — resubmit each suspended trial to the region recorded
  in `runs`, which resumes from that region's latest checkpoint; a cross-region
  move would restart it from step zero and is not warranted for any of them.
  The sweep's two-week limit began 2026-08-07 and its remaining budget is
  unaffected by the pause.

- **2026-08-08T10:10Z** — Cause: the sweep's two leading runs were starved by
  self-contention. `us-east1 v6e-128` held four dispatches; the two slices went
  to lower-progress runs (`m1-p06-aug` 21%, `m1-p03-aug` 12%) while
  `m1-p04-base` (42.9%) and `m1-p06-base` (38.9%) queued behind them for 13
  hours. Evicting progressing work is forbidden, and every smaller us-east1
  target already sat at `pending_target_limit = 1`, so no legal move existed.
  Change: operator raised `pending_target_limit` from 1 to 2 and directed that a
  starved leader be moved to a smaller slice rather than left waiting for a
  faster one. Effect: a high-progress run may now be admitted to a target that
  already holds one pending dispatch, trading peak throughput for continuous
  progress. The limit still governs admission only — never evict progressing
  work to satisfy it.

- **2026-08-08T02:05Z** — Cause: measured restart outcomes contradicted the
  12-hour default. Eight same-target restarts showed runs stalled 5.6–8.9h that
  never self-healed, then resumed within 30–100 minutes of a restart (4 of 7
  resolved), while freshly preempted runs consistently self-heal in 30–60
  minutes without help. A 12-hour timeout therefore spent 5–8 hours per stalled
  run inside a dead zone. Change: operator reduced `restart_after` from 12 hours
  to 3 hours. Effect: long stalls become restart-eligible far sooner, while the
  window stays well clear of the interval in which Iris recovers preemptions on
  its own. Note the failures were regional, not temporal — both europe-west4
  restarts failed while us-east5 went 3 for 3 — so a shorter timeout is not
  expected to help europe-west4.

- **2026-08-07T17:40Z** — Cause: after 5.5h only 448 of 8192 approved chips were
  actually training; the cap was never the constraint, but `replicas = 1` capped
  the sweep at 24 dispatches while 13 trials sat at zero progress in regions
  that were not placing work. Change: operator raised regional replicas to 2,
  restricted to zero-progress trials, and the Racing policy above was rewritten
  accordingly. Effect: stranded trials can now be raced into a healthy region
  instead of waiting on a dead one, without duplicating any trial that is
  already making progress.
- **2026-08-07T17:52Z** — Cause: operator caught this sweep's heartbeats
  conflating fleet *utilization* with fleet *capacity*, e.g. reporting targets
  absent from the utilization snapshot as hardware that "does not exist" and
  naming capacity as the sweep's binding constraint. Both claims exceed what any
  available data supports. Change: added the leading "Utilization is not
  capacity" invariant above. Effect: reslice decisions and heartbeat language
  must now rest on submission outcomes and measured `target_rate`, not on
  inferred capacity; prior conclusions drawn that way are downgraded to "no
  ready slice at that moment."
