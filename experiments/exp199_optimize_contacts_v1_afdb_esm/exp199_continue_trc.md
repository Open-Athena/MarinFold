# exp199 continuation

This is the human reference and production operations document for
`exp199_continue_trc.py`. The script continues two completed exp199 m1-p03
models for another 72,600 steps while preserving their model, AdamW state, RNG,
data position, and absolute trainer step. Both continuations use the 50/50
AFDB–ESM mixture and apply amino-acid statement permutation to every training
example.

The active recipe is subversion `s03` (`2026.08.10.3`). It holds the source-final
learning rate for 80% of continuation training and linearly decays it toward zero
over the final 20%. Production is a regional race managed with the
`run-training-sweep` skill; its durable state is
`scratch/exp199_continuation_trc_s03/exp199_continuation.sqlite`.

> This file is authoritative for both the experiment recipe and sweep operations.
> Read it in full at the start of every heartbeat before inspecting code, SQLite,
> W&B, or Iris.

## Experiment recipe

The two logical trials are:

| Source | Trial | Production W&B ID |
| --- | --- | --- |
| `prot-exp199-cv1-s01-m1-p03-aug-us-east1` | `srcaug` | `prot-exp199-cv1-cont-s03-m1-p03-srcaug-aug100-{region}` |
| `prot-exp199-cv1-s01-m1-p03-base-us-east5` | `srcbase` | `prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100-{region}` |

Both runs retain the original m1 mixture, p03 weight decay, QK-normalized model
architecture, 2,845-token tokenizer, global batch 128, 8,192-token sequence
length, packing, Block-Feistel shuffle, and complete validation cache. Training
continues from absolute step 72,600 through 145,199 with 100% augmentation.

Permanent checkpoint roots omit the region suffix and remain distinct by source:

```text
checkpoints/protein/prot-exp199-cv1-cont-s03-m1-p03-srcaug-aug100
checkpoints/protein/prot-exp199-cv1-cont-s03-m1-p03-srcbase-aug100
```

Each region's `MARIN_PREFIX` is its canonical Marin bucket followed by
`protein-structure/MarinFold/exp199_continue_contacts_v1`; artifacts use version
`2026.08.10.3`.

Use these exact roots when submitting a coordinator; do not derive a bucket name
by mechanically expanding the region name:

| Region | Canonical bucket root |
| --- | --- |
| `europe-west4` | `gs://marin-eu-west4` |
| `us-east1` | `gs://marin-us-east1` |
| `us-east5` | `gs://marin-us-east5` |
| `us-west4` | `gs://marin-us-west4` |

## Learning rate and checkpoints

Both source runs finished step 72,599 at W&B learning rate
`0.0003162299981340766`, matching configured peak LR `3.1623e-4`. The new cycle
has no warmup or rewarmup:

- continuation steps 0–58,079: constant `3.1623e-4`;
- continuation steps 58,080–72,599: linear decay toward zero.

The decay begins at absolute step 130,680. The final optimizer update is step
145,199 at approximately `2.18e-8`; the standard Optax schedule is exactly zero
at the terminal step-145,200 boundary.

`TrainerConfig.initialize_from` performs the full-state load. Do not replace it
with a model-only or reset-step initializer. Temporary checkpoints use the
standard ten-minute cadence, permanent checkpoints use the 8,920-step interval
plus the forced final checkpoint, and complete validation runs every 2,230 steps
and at completion.

## Data and source checkpoints

Every selected region must already contain the AFDB, ESM, and full validation
token caches. The experiment never copies or retokenizes data. Both source seed
roots must contain complete `checkpoints/step-72599` and `hf/step-72599`
inventories below:

```text
checkpoints/protein/exp199-continuation-init/{source run ID}/2026.08.09.1
```

The public archive is `open-athena/marinfold-exp199`. Both sources were archived
once and restored independently from Hugging Face into all four supported GCS
regions. `exp199_checkpoint_transfer.py` retains the idempotent inventory checks,
concurrent hf-xet transfer, and regional restore commands if the distribution
ever needs to be inspected or repeated. Never copy a source checkpoint from one
regional GCS bucket to another.

Before a new production submission, verify:

- all three cache ledgers and both source seed inventories in every selected region;
- source metadata at step 72,599;
- a real full-state smoke with the expected LR shape;
- a lowered plan with `use_qk_norm=true`, global batch 128, sequence length 8,192,
  `cycle_length=[72600, 72600]`, `decay=0.2`, zero warmup/rewarmup,
  `lr_schedule=linear`, and `min_lr_ratio=0`.

Any violation is a hard stop before submission.

## Manual preview

Source `~/marin.env`, select one source and region-local prefix, and omit `--run`
to lower the plan without executing it:

```bash
set -a
source /home/exedev/marin.env
set +a

export MARIN_PREFIX=gs://marin-us-east1/protein-structure/MarinFold/exp199_continue_contacts_v1
export REGION=us-east1
export TPU=v6e-64
export SOURCE=aug

uv run --extra tpu --frozen python exp199_continue_trc.py \
  --version 2026.08.10.3
```

Use `marin-dev` only for smoke and development work. Full-scale jobs use the
production `marin` cluster, `--user eczech`, and `--priority interactive`.

## Production invariants

- Use only Iris cluster `marin` for full-scale coordinators and TPU children.
- Every new production coordinator submission must pass `--region` matching the
  regional run so its control-plane metadata stays local to the run's bucket.
- Every production placement must use at least 64 physical TPU chips.
- Never run two live attempts for the same source lineage in the same region;
  they share a W&B ID and checkpoint root. The two source lineages are independent
  and may run concurrently in one region.
- W&B `run_progress` is authoritative for progress. Iris establishes dispatch
  liveness only.
- Use only region-local token caches and seed checkpoints. Never retokenize, copy
  caches, or read a checkpoint across regions.
- Never restart an s01 continuation dispatch; that recipe was operator-cancelled
  with no winner.
- Never restart an s02 continuation dispatch; its cosine-cooldown recipe was
  cancelled before any run reached cooldown.
- Keep commands, SQLite, and this document free of credentials.

## Placement and recovery policy

- Approved regions: `europe-west4`, `us-east1`, `us-east5`, and `us-west4`.
- Approved families: v6e in Europe/east1/east5, v5e in Europe/west4, and v5p in
  east5.
- Approved physical-chip range: 64–256. Iris v5p topology labels contain twice
  the physical chip count, so eligible v5p labels begin at `v5p-128`.
- Global cap: 1,024 physical chips across both trials.
- Favor 64-chip targets with measured progress; use larger eligible targets only
  when current or persisted evidence supports them.
- `heartbeat_every`: 1 hour.
- `reslice_after`: 1 hour without a new W&B high-water mark.
- `restart_after`: 3 hours after a classified retryable failure or persistent
  same-placement stall.
- `pending_target_limit`: one per source lineage and region.
- Time limit: two weeks from first s03 production submission.

Treat a region as non-scheduling once it has failed to register a W&B run across
its distinct eligible accelerator pools, not merely across sizes within one pool.
While a region is non-scheduling, leave each of its regional runs holding one
outstanding request and do not restart or reslice on `restart_after` alone:
resubmission has never changed the outcome and may forfeit whatever scheduling
standing the request has. Replace one only on a terminal dispatch, on new
evidence about that region, or when it would free capacity a productive region
needs. This suspends timer-driven churn only; the runs stay live as insurance and
resume normal placement policy the moment that region registers a run.

No region is non-scheduling as of 2026-08-12 05:53: `europe-west4`, `us-east5`,
and `us-west4` have each registered, so all four approved regions are under
normal placement policy. The hold above remains the standing response should a
region go dark again. Prefer probing an untried pool over an untried size when a
slot must be spent on such a region.

Within `us-east1`, prefer v6e-128 over v6e-64. Dispatch lifetime there is roughly
3 to 4 hours on v6e-128 against about 26 minutes on v6e-64, which ended six
consecutive generations on 2026-08-11 evening. Both shapes still end in terminal
parents, so budget a replacement every few hours rather than treating one as a
fix.

Once a lineage leader is close enough to finish that no sibling can overtake it,
stop spending timer-driven actions on that lineage's other regional runs. A
replacement can only help if that run could still win or could still serve as
fallback, so compare the sibling's remaining work at its own measured rate
against the leader's. Leave such runs live and untouched; act on them only if the
leader is lost. This suspends churn, not the runs.

At every heartbeat, query all eight exact W&B IDs first, then reconcile only the
exact active Iris dispatch IDs stored in SQLite. Persist observations and actions
atomically and check SQLite integrity before returning. Protect recent progress;
stop an active dispatch before replacing it, and never change priority to solve
scarcity.

## Completion

A source trial completes only when one regional W&B run reaches
`run_progress >= 1` and its expected final permanent checkpoint is reachable.
Stop only that source trial's losing regional siblings, record its winner and
checkpoint verification atomically, and continue operating the other source trial.
Remove the hourly heartbeat only after both trials finish or the time limit expires.

## Change record

- 2026-08-12: Added the endgame rule suspending timer-driven actions on regional
  runs that can no longer win or serve as fallback for their lineage.
- 2026-08-12: All four approved regions registered, so the non-scheduling hold is
  inactive and retained only as the standing response. Recorded the us-east1
  shape preference for v6e-128 over v6e-64 on measured dispatch lifetime.
- 2026-08-11: Generalized the non-scheduling-region hold to any region that fails
  across its distinct accelerator pools, and added `us-east5`, which refused
  v6e-64, v6e-128, and v5p-128 over more than nine hours after 00:28.
- 2026-08-11: Suspended timer-driven restarts for `europe-west4` and `us-west4`.
  Both regions had failed to register any W&B run across every eligible shape,
  and a controlled same-target restart of the base Europe run produced no
  different outcome, so hourly resubmission was churn without evidence.
- 2026-08-10: Replaced the cancelled s02 cosine cooldown with the s03 linear
  cooldown and fresh W&B/checkpoint identities; no s02 run reached cooldown.
- 2026-08-10: Consolidated the human guide and operations runbook into this file;
  heartbeat and SQLite references now use this single authoritative document.
- 2026-08-10: Cross-region reporting exposed harmless coordinator path mentions;
  future production coordinators are now pinned to their regional run with
  `--region`, eliminating even the small cross-region executor-lock traffic.
