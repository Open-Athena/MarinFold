---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T21:48:34Z'
  experiment: exp279_models_exact_soft_contact_targets
  kind: models
  short_description: Exact soft targets on the default decontaminated recipe, 32 H100s,
    required validation
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b02
    entity: open-athena
    project: MarinFold
    run_id: exp279-soft-s0-cw-h100x32-b02
    run_name: exp279-soft-s0-cw-h100x32-b02
  git_sha: b0dd33eca8836c1dddbe6588e4befc3b2dd5c67d
  iris_job_ids:
  - /bizon/exp279-soft-pilot-cw-h100x32-a04
  - /bizon/exp279-soft-production-cw-h100x32-a01
  - /bizon/exp279-soft-production-cw-h100x32-a01/exp279-soft-production-cw-h100x32-a01-base
  - /bizon/exp279-soft-production-cw-h100x32-a02
  - /bizon/exp279-soft-production-cw-h100x32-a02/exp279-soft-production-cw-h100x32-a02-base
  - /bizon/exp279-soft-production-cw-h100x32-a03
  - /bizon/exp279-soft-production-cw-h100x32-a03/exp279-soft-production-cw-h100x32-a03-base
---
# 2026-09-09 · exp279_models_exact_soft_contact_targets · exp279-soft-s0-cw-h100x32-b02

**Launched:** 2026-09-09T21:48:34Z by bizon  
**Kind:** models  
**Experiment:** exp279_models_exact_soft_contact_targets  
**W&B:** [exp279-soft-s0-cw-h100x32-b02](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b02)  
**Git:** `b0dd33ec`  

## Description

Exact soft targets on the default decontaminated recipe, 32 H100s, required validation

## Detailed plan

Fresh exact-soft-target model using exp232 m2/p06's decontaminated AFDB/ESM
mixture, Qwen3 1.47B, context 8192, global batch 128 and prescribed schedule
through update 363000. Four H100 nodes on cw-us-east-02a, batch priority,
per-device batch 1 (microbatch 32, four accumulation steps).

The first 32 updates are a resumable pilot. Continue the same native state and
frozen manifest with the production phase driver after validating the pilot.

## Changes from previous runs

Validation now uses the actual AFDB validation cache and must be nonempty.
The earlier b01 pilot omitted validation and is not a continuation source.

## Provenance and outputs

Frozen manifest: `experiments/exp279_models_exact_soft_contact_targets/data/soft_pilot_cw_launch.json`.
Native and HF checkpoints: `s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-s0-cw-h100x32-b02/`.
The worker independently verifies the runtime source hash, dependency lock and
all three input cache ledgers. No bulk cross-region transfer was performed.

## Pilot result

Completed 32 updates with training loss 6.3659, ordinary validation CE 6.3678,
and 3.464 seconds/update. Verified permanent native step-31 state and HF export
with tokenizer. All four Iris tasks succeeded. Production resumes this state.

## Production resume

At 2026-09-09 21:59:58 UTC the base-phase workers completed update 32 after
restoring the pilot native checkpoint, with finite loss 6.3391. W&B reports the
same run as running. The driver stays alive and waits for this phase, then
continues the two prescribed transitions. Full training and accuracy evaluation
are not yet complete.

## Restart 1

Worker 3 exited 139 at logged step 55520 on 2026-09-12 04:51:15 UTC; the other
three workers ended as coscheduled siblings and the production driver propagated
the failure. Iris newest complete recovery checkpoint is step 55265. Relaunch
driver `/bizon/exp279-soft-production-cw-h100x32-a02` and its base-phase child
were submitted on 2026-09-13 with the original frozen source, input manifests,
W&B identity, 32-H100 geometry, and batch priority. The child initially entered
the normal Kueue capacity gate awaiting 32 GPU/RDMA quota units.

## Restart 2

The a02 base-phase gang failed on 2026-09-15 when replica 2 was OOM-killed
during the temporary step-109962 checkpoint save; the other three replicas were
stopped as coscheduled siblings. That directory lacks `metadata.json` and is not
a committed resume source. Driver
`/bizon/exp279-soft-production-cw-h100x32-a03` was submitted from the original
frozen source and restored the latest complete checkpoint, step 109720, at
update 109721. Its child is
`/bizon/exp279-soft-production-cw-h100x32-a03/exp279-soft-production-cw-h100x32-a03-base`.
The data manifest, 32-H100 geometry, base-phase recipe, checkpoint root, and W&B
run identity are unchanged.
At 2026-09-15 12:47:18 UTC both a03 jobs were running and W&B had advanced past
the failed attempt to observed step 109985, with finite loss 2.9193 and 3.497
seconds/update.

## Restart 3

The a03 base-phase gang failed on 2026-09-16: task 0 exited 139 (SIGSEGV) after
17 hours 43 minutes, and the other three replicas were stopped as coscheduled
siblings. Iris reports `preemptions=0`, so this was not a preemption, and the
retained diagnostic holds only the faulthandler extension-module tail with no
native frame, so the fault site remains unidentified. This is the second exit
139 of this run, after logged step 55520 on a01.

The last logged W&B step was 125935, but that is not a committed checkpoint.
`step-125936` and `step-125680` both stop short of `metadata.json`, exactly as
a02's OOM-killed `step-109962` did, so all three terminal failures of this run
landed on or near a checkpoint save. Levanter's `discover_latest_checkpoint`
admits only directories with a readable `metadata.json`, so `--resume-latest`
cannot select either incomplete directory.

Driver `/bizon/exp279-soft-production-cw-h100x32-a04` was submitted at
2026-09-16 13:36:51 UTC from the original frozen source
(`b0dd33eca8836c1dddbe6588e4befc3b2dd5c67d`, code SHA-256 `d42c8bab...1ac1`),
restoring the latest complete checkpoint, temporary step 125489, at update
125490. That discards 446 updates relative to the last logged step. The data
manifest, 32-H100 geometry, batch priority, base-phase recipe, checkpoint root
and W&B run identity are unchanged.

At 2026-09-16 14:14:32 UTC both a04 jobs were running, W&B had returned to
running, and the run had passed the failed attempt's high-water step 125935 to
observed step 126023, with finite loss 2.8922 and 3.518 seconds/update.

Iris retains no logs for this job, so the resume point could not be read from
the driver log, and W&B cannot show it directly either: a resumed run drops
steps at or below its previous maximum, so the first visible a04 row is 125936
rather than 125490. Timing confirms the re-run of the dropped range. The
worker's train-step compile artifact landed at 13:42:37 UTC and the first
visible step at 14:09:06 UTC, a 26.5 minute gap that is about 454 updates at
3.5 seconds each, against the 446 predicted by resuming at update 125490. The
same arithmetic reproduces a03's first visible step from its own resume point.

## Restart 4

The a04 base-phase gang ran 2 days 10 hours 36 minutes, from update 125490 to
logged step 183541, and then task 0 was OOM-killed (exit 137) with
`preemptions=0`. The diagnostic carries a finelog client send failure
(DEADLINE_EXCEEDED, 19 rows, retryable) immediately before the kill. This is the
second OOM kill after a02, against exit 139 on a01 and a03; the two modes
alternate, but all four terminal failures have left a checkpoint directory that
stops short of `metadata.json`, so all four landed on or near a checkpoint save.
a04's 58 hours against a03's 18 shows the hazard is not a fixed per-step
probability.

`step-183543` is the incomplete directory; the latest complete checkpoint is
temporary `step-183289`. Driver
`/bizon/exp279-soft-production-cw-h100x32-a05` was submitted at 2026-09-19
00:33 UTC to restore it at update 183290, discarding 252 updates.

Iris rejected the first submission attempt: the experiment venv's `marin-iris`
build is 2026-09-01, the controller runs 2026-09-17, and the freshness window is
14 days. The job was submitted instead with the marin checkout's iris client
(`/home/bizon/git/marin/lib/iris/src`, last iris commit 2026-09-15) shadowing
the venv package on `PYTHONPATH`. That client resolves its date from git in a
checkout, so this is a genuinely fresh client rather than an override of the
gate. The four guarded `runtime_packages` versions are read from the venv's
dist-info and are unchanged, and the emitted manifest is identical to the
pilot's, so the frozen guard passed on its own terms. Workers are exempt from
the gate and install from the staged frozen `uv.lock`, so the worker
environment, training code and configuration are unaffected.

At 2026-09-19 00:52 UTC both a05 jobs were running, W&B had returned to running,
and the run had passed a04's high-water step 183541 to observed step 183607,
with finite loss 2.9063 and 3.502 seconds/update. The resume point was again
established by timing: the worker's compile artifact landed at 00:31:30 UTC and
the first visible step 183542 at 00:46:47 UTC, a 15m17s gap that is about 262
updates at 3.5 seconds each, against the 252 predicted by resuming at update
183290.

## Base phase complete

a05 ran the base phase to its end without a restart: 34,510 updates from
183290 to 217800 in about 34 hours, against a03's 18 hours and a04's 58 before
they failed. The base-phase child
`/bizon/exp279-soft-production-cw-h100x32-a05-base` reached state `succeeded`
at 2026-09-20 10:49 UTC with W&B run progress 1.0, training loss 2.8969 and
ordinary validation CE 3.0463.

Its final checkpoint `step-217800` is complete and permanent
(`is_temporary: false`), the first clean phase boundary of the run. Every prior
terminal failure left a checkpoint directory stopping short of `metadata.json`;
this one did not.

The production driver stayed alive across the transition and spawned the
recovery-phase child
`/bizon/exp279-soft-production-cw-h100x32-a05-recovery` at 10:48:14 UTC, which
covers updates 217801 through 333960 at constant learning rate 1e-3. The final
phase, updates 333961 through 363000 with the learning rate decaying to 5e-5,
follows after it. No relaunch or intervention was needed for this transition.

## Restart 5: the recovery-phase restore blocker, and the re-freeze

a05's recovery child ran about 17.5 hours from update 217801, then a replica was
preempted and the retry died in under four minutes. a06 and a07 died the same
way. All three resumed `step-235057`; the one attempt that worked, a05's first,
had resumed the base-phase `step-217800`.

Streaming `iris job logs -f` during a07 captured the traceback that Iris's
fixed-size stderr tail had been discarding behind per-second JAX coordination
warnings:

```
File ".../levanter/tensorstore_serialization.py", line 1058, in _restore_replica_axis
TypeError: lax.bitcast_convert_type does not support bool or complex values ...
Got operand dtype=bool, new_dtype=dtype('uint8').
```

The recovery and final phases set `skip_bad_steps=True`, and SkipStep's
`valid_mask` is `jnp.bool_`, so every checkpoint those phases write carried a
boolean leaf the serializer could not read back. Base-phase checkpoints carry
no such leaf, which is why every earlier resume worked. This also blocked the
recovery -> final transition outright, so the run could not have finished even
uninterrupted. The defect is byte-identical in current upstream levanter.
Details in `data/recovery_phase_restore_blocker.md`.

With the user's authorization the deserialization path was repaired in commit
`90f0d577`: boolean leaves reduce by logical OR over the replica axis, which
reproduces the upstream byte-sum exactly given that one replica contributes and
the rest hold zero. Eleven tests cover it, and the full suite shows the same
eight pre-existing failures before and after. No hyperparameter, data,
optimizer, schedule or model change.

The experiment was re-frozen on that commit. The manifest differs from the
pilot's in exactly `source.git_sha` and `source.code_sha256`; runtime packages,
`uv_lock_sha256`, `reference_sha`, tokenizer and all three input ledgers are
unchanged, and `data/soft_refrozen_v2_reference.json` re-arms the launcher guard
against the new source. The checkpoint root's `experiment.json` was updated in
the same two fields, with the original preserved as
`experiment.json.pre-refreeze-b0dd33e`, because the worker's resume check
compares an identity that embeds the whole manifest.

Driver `/bizon/exp279-soft-production-cw-h100x32-a08` was submitted at
2026-09-21 14:04:33 UTC and restored `step-235057` at update 235058. All four
ranks logged the load at 14:08:2x with no traceback, against three attempts that
had died on that same checkpoint, and by 14:13:58 the run had passed the failed
attempts' high-water step 235104. Roughly 14 hours of cluster time were lost to
the blocker.

## Restart 6: a silent stall

a08 ran the repaired restore for about six hours and then stalled at logged step
240891 on 2026-09-21. This is the first stall of the run rather than a crash:
no exit code, no traceback, and Iris reported both jobs `running` throughout.
Three independent signals agreed that nothing was progressing — the worker
produced zero log lines for 25 minutes, W&B went to `crashed` with the
heartbeat stuck for 15 minutes, and no checkpoint was written after
`step-240841` at 20:23:29 UTC despite the cadence being due. Iris state alone
would never have surfaced it; the absence of object-store writes is the
reliable signal when a task zombies.

The a08 root was cancelled, which killed it and its descendant. That is a
job-level action on this experiment's own job, not a change to the shared
cluster. Driver `/bizon/exp279-soft-production-cw-h100x32-a09` was submitted at
2026-09-21 20:46 UTC from the re-frozen source `90f0d577` and restored the
complete `step-240841` at update 240842, discarding 50 updates. By 21:26 the run
had passed a08's high-water step 240891 with a live heartbeat.

The run has now shown four distinct failure modes: exit 139 (SIGSEGV), exit 137
(OOM kill), the boolean-restore blocker, and this stall.
