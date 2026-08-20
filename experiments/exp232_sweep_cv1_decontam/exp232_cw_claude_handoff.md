# exp232 CoreWeave Sweep Handoff for Claude

This handoff is for operating the exp232 CoreWeave GPU sweep to completion. The
training jobs are intentionally left running, but this Codex session stops
scheduling heartbeats after its final handoff heartbeat. The next operator must
start a new W&B-first heartbeat loop.

## Checkout and authoritative state

- MarinFold checkout: `/home/exedev/repos/MarinFold-br/exp232-sweep-cv1-decontam`
- Branch: `exp/232-sweep-cv1-decontam`
- Experiment directory:
  `experiments/exp232_sweep_cv1_decontam`
- Training entry point:
  `experiments/exp232_sweep_cv1_decontam/exp232_sweep.py`
- Authoritative Operations document:
  `experiments/exp232_sweep_cv1_decontam/exp232_cw_operations.md`
- Authoritative dynamic ledger:
  `scratch/exp232_cw_s02/exp232_cw_sweep.sqlite`
- Iris/Marin runtime checkout and environment:
  `/home/exedev/repos/marin-br/main` and its `.venv`
- Sweep skill:
  `/home/exedev/repos/MarinFold-br/exp199-optimize-contacts-v1/.agents/skills/run-training-sweep/SKILL.md`
- Credentials are loaded from `/home/exedev/marin.env`. Never print or persist
  their values.
- W&B group:
  <https://wandb.ai/open-athena/MarinFold/groups/prot-exp232-cw-cv1-decontam-s02>

Read the sweep skill, all of its referenced documents, the complete Operations
document, and the complete training entry point before the first live query and
at every heartbeat. Treat Operations, code, and SQLite as authoritative over this
handoff if they differ.

## Current scope and operator directives

The code declares ten trials, but only eight remain in completion and recovery
scope. Never redispatch either abandoned trial:

- `m1-p01-aug`: operator-abandoned after divergence.
- `m1-p04-aug`: operator-abandoned after divergence on 2026-08-16; its exact
  east n16 root was stopped and verified killed.

The remaining trials are `m1-p02-aug`, `m1-p03-aug`, `m1-p06-aug`,
`m2-p01-aug`, `m2-p02-aug`, `m2-p03-aug`, `m2-p04-aug`, and
`m2-p06-aug`.

Hard constraints:

- W&B routing is always `open-athena/MarinFold`; never fall back to a personal
  entity or project.
- Every Iris submission goes through `--cluster marin`, specifies an approved
  `--target-cluster`, uses `--priority batch`, and includes `--user eczech`.
  Never switch to dev or another priority to gain admission.
- Iris `job list`, `job stop`, and `rpc` do not accept `--user`; use exact
  `/eczech/...` job roots for those commands. The `--user eczech` invariant is
  mandatory on every `job run` submission.
- Approved H100 clusters are `cw-rno2a` and `cw-us-east-02a`. Never use
  `cw-us-west-04a`; do not use the unsupported ARM/GB200 `cw-us-east-08a`.
- Approved production shapes are 2, 4, 8, and 16 nodes: 16, 32, 64, and 128
  H100s respectively. Maximum active compute is 640 GPUs.
- Preserve one active writer per trial/W&B/checkpoint identity. Stop and verify
  the exact old root before submitting a replacement.
- Keep permanent checkpoints on the exp199 schedule and temporary checkpoints
  every 15 minutes; those semantics belong to `exp232_sweep.py` and must not be
  changed while operating the sweep.
- Do not post to PR #233 unless the operator explicitly supplies or requests the
  update.

## Current dispatch layout

The ledger is the source of truth for attempt numbers. At the handoff transition,
the intended active layout is 8 trials and 528 submitted H100s:

| Trial | Exact Iris root | Target | GPUs |
| --- | --- | --- | ---: |
| `m1-p02-aug` | `/eczech/exp232-s02-m1-p02-aug-rno2a-h100-n4-a03` | RNO n4 | 32 |
| `m1-p03-aug` | `/eczech/exp232-s02-m1-p03-aug-use02a-h100-n16-a05` | east n16 | 128 |
| `m1-p06-aug` | `/eczech/exp232-s02-m1-p06-aug-rno2a-h100-n8-a01` | RNO n8 | 64 |
| `m2-p01-aug` | `/eczech/exp232-s02-m2-p01-aug-rno2a-h100-n8-a08` | RNO n8 | 64 |
| `m2-p02-aug` | `/eczech/exp232-s02-m2-p02-aug-use02a-h100-n16-a06` | east n16 | 128 |
| `m2-p03-aug` | `/eczech/exp232-s02-m2-p03-aug-rno2a-h100-n2-a07` | RNO n2 | 16 |
| `m2-p04-aug` | `/eczech/exp232-s02-m2-p04-aug-rno2a-h100-n4-a02` | RNO n4 | 32 |
| `m2-p06-aug` | `/eczech/exp232-s02-m2-p06-aug-rno2a-h100-n8-a05` | RNO n8 | 64 |

At the final Codex heartbeat, observed at 2026-08-16T12:59:59Z, seven in-scope
runs were W&B `running` with fresh progress: `m1-p02=0.29812672`,
`m1-p03=0.36292700`, `m1-p06=0.55738292`, `m2-p01=0.28716942`,
`m2-p03=0.14325758`, `m2-p04=0.28926997`, and `m2-p06=0.31504132`.
The enlarged `m2-p02` attempt a05 had advanced to `0.22107438` and then failed
in isolation. Codex closed a05 and submitted same-target n16 attempt a06 from the
shared checkpoint. At 2026-08-16T13:02:14Z, the a06 root and 128-H100 child were
running, east was full at 0/256 free, and RNO was full at 0/512 free. W&B still
showed the prior `crashed` state because a06 had only just started; confirm W&B
recovery first at takeover.

## Heartbeat contract

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and
`pending_target_limit=1`. At each pass:

1. Reread the skill and its references, Operations, and the full training script.
2. Build the SQLite inventory snapshot.
3. Query all eight in-scope W&B IDs first. Only a new `run_progress` high-water
   mark proves training progress.
4. Check exact Iris roots only for liveness. Use deeper Iris state or capacity
   only for a recorded recovery reason or the explicit capacity-expansion
   directive below.
5. Persist observations and action evidence, rebuild the snapshot, make one
   coordinated fleet decision, act, and replan after every material change.
6. Check SQLite integrity and git synchronization.
7. If the sweep is nonterminal, schedule exactly one time-based next heartbeat
   for 30 minutes later. Do not use a polling daemon.

Snapshot and integrity commands:

```bash
python /home/exedev/repos/MarinFold-br/exp199-optimize-contacts-v1/.agents/skills/run-training-sweep/scripts/persistence.py \
  snapshot scratch/exp232_cw_s02/exp232_cw_sweep.sqlite \
  --reslice-after-hours 1

python /home/exedev/repos/MarinFold-br/exp199-optimize-contacts-v1/.agents/skills/run-training-sweep/scripts/persistence.py \
  check scratch/exp232_cw_s02/exp232_cw_sweep.sqlite
```

## H100 fleet utilization and maximum batch use

The operator explicitly directed the sweep to consume as much available H100
compute as possible at batch priority. The federation backend view used for this
CoreWeave-specific directive is:

```bash
/home/exedev/repos/marin-br/main/.venv/bin/iris --cluster marin \
  rpc controller list-peers 2>/dev/null | \
jq '[.peers[] |
  select(.peer_id=="cw-rno2a" or .peer_id=="cw-us-east-02a") |
  . as $p | .backends[] |
  {peer_id:$p.peer_id,
   reachable:$p.reachable,
   pending_task_count,
   availability:.availability.amounts,
   total_amounts:.availability.total_amounts,
   held_by_band:.availability.held_by_band}]'
```

Interpret `availability.h100` as the currently visible free H100 count, and
`held_by_band` as a point-in-time allocation breakdown. It is a placement hint,
not proof of training: W&B remains the progress authority. Requery after stops and
submissions because releases and gang holds settle asynchronously.

Follow the maximum-use directive without ever changing priority:

- Use free whole-gang headroom to enlarge healthy checkpoint-preserving runs when
  that reduces wall-clock completion time. The operator explicitly allows this
  even for recently progressing dispatches.
- Typical net requirements are 16 GPUs for n2→n4, 32 for n4→n8, and 64 for
  n8→n16. Account for the 640-GPU fleet cap and every pending gang before acting.
- Prefer current SQLite `target_rate`, fresh progress, and successful admission.
  Recent measurements favored n16 over n8 over n4 over n2, but this is evidence,
  not permanent policy.
- Build a single fleet plan. Reserve capacity for each planned gang, stop the
  exact old writer, verify it is no longer running, submit a unique persisted
  attempt, then requery and replan. Never double-count transitional free capacity.
- If a larger gang is freshly pending, allow its normal admission window. If it
  remains unproductive for `reslice_after`, compare waiting against a smaller gang
  that can admit immediately. After three hours without progress, restart or
  reslice from the shared checkpoint.
- Never use `--priority dev`, interactive, or any priority other than batch.

Submission template (fill values from the planned trial and persisted attempt):

```bash
set -a
source /home/exedev/marin.env
set +a

/home/exedev/repos/marin-br/main/.venv/bin/iris --cluster marin job run --no-wait \
  --target-cluster <cw-rno2a-or-cw-us-east-02a> \
  --priority batch --user eczech \
  --job-name <unique-root-name> \
  --enable-extra-resources --cpu 2 --memory 6GB --disk 32GB \
  -e MARIN_PREFIX s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e WANDB_ENTITY open-athena -e WANDB_PROJECT MarinFold \
  -e HF_TOKEN "$HF_TOKEN" \
  -e TRIAL <trial> -e CLUSTER <target-cluster> -e NODES <2|4|8|16> \
  -- python exp232_sweep.py --version 2026.08.14.2 --run
```

Persist a redacted form of every command. Never store secret values.

## Recovery and completion

- Let Iris recover ordinary preemptions. Do not replace a dispatch for a
  preemption alone.
- Observe the entire W&B fleet before classifying failures. Retry an isolated
  failure only after stopping its exact root; pause blind replacements for
  correlated failures.
- A trial completes only when W&B `run_progress >= 1` and its expected checkpoint
  under the shared exp232 S3 namespace is reachable. W&B `finished` alone is not
  sufficient.
- When all eight in-scope trials complete, stop remaining roots, verify every final
  checkpoint and SQLite integrity, and schedule no further heartbeat.

## State at handoff

Codex intentionally schedules no heartbeat after its final handoff pass. All eight
exact active roots are left running, including the newly admitted `m2-p02` a06.
Claude should begin by rereading authoritative state, checking the current git
commit and SQLite integrity, and performing a fresh W&B-first heartbeat.
