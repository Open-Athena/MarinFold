---
marinfold_run:
  user: bizon
  launched_at: '2026-09-16T15:13:24Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: Reshuffled second full-corpus epoch continued from step-213072
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B
    run_name: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B
  git_sha: 822caeaa78532b8cf1e4ec4c4136d2937bfd809d
  iris_job_ids:
  - /bizon/exp277-continue-a01
  - /bizon/exp277-continue-a02
  - /bizon/exp277-continue-a02/exp277-train-4ff31f60
  - /bizon/exp277-continue-a02/exp277-train-0e644343
---

# 2026-09-16 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B

**Launched:** 2026-09-16T15:13:24Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B)  
**Git:** `822caeaa`  

## Description

Reshuffled second full-corpus epoch continued from step-213072

## Detailed plan

Restore the complete trainer state (model, optimizer, RNG, step) from
`step-213072` of `contacts-v1-exp277-m2-p06-full-epoch-1.5B` -- the last save
before the first epoch's cooldown -- and add one more full epoch over all
34,092,146 packed examples in a fresh data-seed-1 permutation. 266,345
updates, absolute steps 213,073-479,417, WSD at peak 1e-3 for 80% then a
linear cooldown to 1e-4 over the final 20%. 16 nodes / 128 H100s at batch
priority on CoreWeave US-EAST-02A.

## Changes from previous runs

- Second epoch over the same corpus rather than a first pass: data seed 1
  instead of 0, so every packed example is seen once more in a new order.
- First MarinFold run to restore full trainer state onto a *new* finite
  dataset. That needed `ContinuationEpochDataset`, which addresses the new
  epoch at the restored absolute data offset (213,073 x 128 = 27,273,344)
  because levanter derives the data offset from the absolute optimizer step.
- Gated on `...-smoke-a02`, after the first restore smoke failed on
  `DataLoader.__init__`'s index-0 structure probe.

## Notes

**Completed 2026-09-19T15:23Z at step 479,417 — all 266,345 added updates.**
Final validation loss **2.98099**, against the first epoch's final 2.98441:
a whole second epoch bought **0.0034 nats**. The plateau was flat throughout
(~3.052-3.057 vs ~3.058 for epoch 1's plateau); essentially all of the
apparent gain is the cooldown, which is worth ~0.07 nats on its own. Treat
0.0034 nats as within the range where contact accuracy can move either way
(see #169: matched loss does not imply matched accuracy), so the contact
eval decides whether this checkpoint is worth promoting.

Final learning rate 1.0002e-4, the configured 0.1x WSD floor. Wall clock
2 days 23:37, 0.968 s/step including all overhead, 1.209M tokens/s,
MFU ~13.0%, 502,706,208,768 tokens. Zero tracebacks in the final gang.

Final native checkpoint `checkpoints/step-479417` is 119 objects /
17,657,263,371 bytes; `hf/step-479417` is 6 files / 5.89 GB including
`tokenizer.json` and `tokenizer_config.json`. Permanent checkpoints landed
every 26,634 steps: 239706, 266340, 292974, 319608, 346242, 372876, 399510,
426144, 452778, 479412.

### Recovery history

- `/bizon/exp277-continue-a01` (15:01Z 09-16) was admitted with all sixteen
  nodes and died in under a minute: worker task 2 hit a stream error part-way
  through its 480 MB `nvidia-cudnn-cu13` download and gang co-scheduling
  failed the other fifteen. The production phases now submit with
  `--max-retries 3` because of this.
- `/bizon/exp277-continue-a02` (15:13Z 09-16) ran the epoch. Three
  preemptions. Two were absorbed in place, resuming at steps 229,272 and
  231,597. The third, at 00:48:02Z on 09-19, SIGTERMed task 0; it was then
  SIGKILLed (exit 137) and 29 s later all fifteen siblings aborted together
  on `the JAX distributed service detected fatal errors`. Because the child's
  `max_task_failures` is 0 this failed the child job, and the driver
  relaunched a fresh gang `exp277-train-0e644343` one second later, resuming
  from temporary checkpoint `step-424901`. Cost: ~25 minutes.

### Two traps worth remembering

- **Temporary checkpoints are not in the run directory.** They go to
  region-local TTL storage,
  `s3://marin-us-east-02a/tmp/ttl=14d/checkpoints-temp/.../checkpoints/step-N`.
  The run directory holds only the permanent `keep` checkpoints, so a gap
  there is expected rather than lost work.
- **The driver's lock heartbeat can die silently.** At 18:07:54Z on 09-18
  `marin.execution.step_status._heartbeat` took a `botocore ClientError
  (SlowDown)` from CoreWeave object storage; it catches only `LeaseLostError`,
  so the thread died and `.executor_status.lock` went stale for the rest of
  the run. Harmless here -- `write_status()` is an unconditional write and
  `release()` is idempotent -- but while the lease is stale the lock will not
  block a duplicate submission to the same output path.
