# exp232 CoreWeave Cooldown Operations

> This document governs the exp232 cooldown sweep managed with the
> `run-training-sweep-cw` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

- Each trial has one W&B identity, one canonical CWS3 checkpoint root, and at
  most one active writer. Placement changes preserve both identities.
- Restore each requested permanent checkpoint as complete Levanter trainer
  state. Keep source-specific optimizer wrapping, data seed, mixture, and
  augmentation continuity intact.
- Use whole-node CoreWeave GPU gangs at batch priority. Every Iris root uses the
  `marin` controller, an exact target cluster, and `--user eczech`; the training
  child must also submit in Iris's batch band.
- Route W&B only to `open-athena/MarinFold` and write only within the canonical
  exp232 CWS3 prefix.
- Production validation uses the complete contacts-v1 validation cache. Smoke
  runs may use one evaluation batch and isolated temporary checkpoint output.

## Sweep Definition

- Entry point and three-source catalog:
  `experiments/exp232_sweep_cv1_decontam/gpu/exp232_cooldown_cw.py`.
- All three sources have complete permanent checkpoints in canonical CWS3 and
  exact matching `optim/learning_rate` records in their source W&B histories.
- Production starts on 8-node gangs. The entry point accepts 1, 2, 4, 8,
  or 16 nodes on H100 and additionally 32 or 64 nodes on GB200. The 32-node
  GB200 profile is DP128; the 64-node profile is DP128 x TP2. Both exactly
  preserve the fixed global batch and optimizer semantics.

## Operator Choices

- Time limit: 14 days from the first production dispatch.
- Initial compute: three 8-node gangs, placed across the eligible fleet according
  to validated capacity.
- Approved clusters: `cw-us-east-08a`, `cw-us-east-02a`, and `cw-rno2a`.
- Excluded cluster: `cw-us-west-04a`.
- Iris user: `eczech`; priority: `batch` only.
- Operations document: this tracked file.
- Authoritative ledger:
  `scratch/exp232_cooldown_cw/exp232_cooldown_cw.sqlite`.

## Operating Policy

- Use `heartbeat_every=1h`, `reslice_after=1h`, `restart_after=3h`, and
  `pending_target_limit=1` per exact target.
- Query W&B before Iris and fleet utilization at every heartbeat. Let Iris
  recover ordinary preemptions; do not replace a dispatch for preemption alone.
- Treat isolated failures per trial. Pause replacement and investigate when
  failures recur or correlate across trials.
- A trial completes only when W&B `run_progress >= 1` and its expected final
  permanent checkpoint is reachable. Stop any remaining root only after
  reconciling that completion.
- Never restart or reslice an abandoned trial. The TRC-derived `m2-p06-lr005`
  cooldown was abandoned by the operator for overfitting; leave its W&B run and
  checkpoints as historical evidence only.
- Start production only after all three source-specific smoke runs restore full
  state and their logged LR histories show the intended inclusive decay.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-02a` | H100 | 1 | 8 | eligible | smoke and reslice profile |
| `cw-us-east-02a` | H100 | 2 | 16 | eligible | supported reslice profile |
| `cw-us-east-02a` | H100 | 4 | 32 | eligible | supported reslice profile |
| `cw-us-east-02a` | H100 | 8 | 64 | eligible | initial production profile |
| `cw-us-east-02a` | H100 | 16 | 128 | eligible | supported reslice profile |
| `cw-rno2a` | H100 | 1 | 8 | eligible | smoke and reslice profile |
| `cw-rno2a` | H100 | 2 | 16 | eligible | supported reslice profile |
| `cw-rno2a` | H100 | 4 | 32 | eligible | supported reslice profile |
| `cw-rno2a` | H100 | 8 | 64 | eligible | initial production profile |
| `cw-rno2a` | H100 | 16 | 128 | eligible | supported reslice profile |
| `cw-us-east-08a` | GB200 | 1 | 4 | eligible | smoke and reslice profile |
| `cw-us-east-08a` | GB200 | 2 | 8 | eligible | supported reslice profile |
| `cw-us-east-08a` | GB200 | 4 | 16 | eligible | supported reslice profile |
| `cw-us-east-08a` | GB200 | 8 | 32 | eligible | initial production profile |
| `cw-us-east-08a` | GB200 | 16 | 64 | eligible | supported reslice profile |
| `cw-us-east-08a` | GB200 | 32 | 128 | eligible | maximum data-parallel profile at global batch 128 |
| `cw-us-east-08a` | GB200 | 64 | 256 | eligible | full-state DP128 x TP2 smoke succeeded |
| `cw-us-west-04a` | H100 | 1/2/4/8/16 | 8/16/32/64/128 | ineligible | CI cluster |

## Change Record

- 2026-08-24 00:12 UTC: operator changed `heartbeat_every` from 30 minutes
  to 1 hour after stable smaller-gang recovery; retain fleet-utilization checks
  and opportunistic gang enlargement at each heartbeat.
- 2026-08-24 09:58 UTC: newly free GB200 capacity made larger gangs useful.
  Added 32-node DP128 and 64-node DP128 x TP2 GB200 profiles, both preserving
  global batch 128. H100 remains capped at 16 nodes.
- 2026-08-24 10:14 UTC: the 64-node GB200 smoke restored the exact full trainer
  state, completed ten updates with the expected LR history, and measured about
  2.53M tokens/s. Marked the DP128 x TP2 target eligible for production.
- 2026-08-24 19:00 UTC: operator abandoned
  `prot-exp232-cw-cv1-decontam-cooldown-s01-m2-p06-lr005-trc-from363000` for
  overfitting. Its exact active Iris root was killed and must not be restarted.
