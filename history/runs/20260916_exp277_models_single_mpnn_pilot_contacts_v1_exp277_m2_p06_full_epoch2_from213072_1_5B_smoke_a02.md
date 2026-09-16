---
marinfold_run:
  user: bizon
  launched_at: '2026-09-16T13:46:18Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: Repaired ten-step restore smoke for the reshuffled second epoch
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02
    run_name: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02
  git_sha: 3babe97e54f05d29408ef852e37cc08574e8a61d
  iris_job_ids:
  - /bizon/exp277-continue-smoke-a02
  - /bizon/exp277-continue-smoke-a02/exp277-train-4ffc905f
---

# 2026-09-16 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02

**Launched:** 2026-09-16T13:46:18Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke-a02)  
**Git:** `3babe97e`  

## Description

Repaired ten-step restore smoke for the reshuffled second epoch

## Detailed plan

One node / eight H100s at batch priority on CoreWeave US-EAST-02A. Restores
the full trainer state (model, optimizer, RNG, step) from `step-213072` of
`contacts-v1-exp277-m2-p06-full-epoch-1.5B`, then runs ten updates from
absolute step 213,073 over a fresh data-seed-1 permutation, plus a two-batch
validation and one checkpoint. Production is gated on this.

## Changes from previous runs

Replaces the failed
`contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke`, which raised
`IndexError: continuation data starts at offset 27273344, got 0` before its
first update. Two changes:

- The continuation dataset now serves `DataLoader.__init__`'s one-off global
  index-0 structure probe from the first example of the new epoch, while
  still rejecting every other index below the restored absolute offset. The
  offset itself is kept: levanter derives the data offset from the absolute
  optimizer step, so a locally-indexed epoch would have silently trained
  only the last ~20% of the corpus.
- Smoke identities now carry their attempt number. The failed smoke left
  `SUCCESS` in `.executor_status` at the old path, and a `SUCCESS` status
  skips the step outright, so a rerun there would have been cache-served.

## Notes

Submitted 2026-09-16T13:46:18Z. The GPU child was `SchedulingGated` for about
an hour: CoreWeave US-EAST-02A was at 255/256 H100s in use (exp278 `scale-v1`
215 single-GPU workers, exp279 `soft-production` 32, an 11-day dev pod 8),
with no node holding eight free GPUs. It started at 14:49:29Z, initialized
W&B at 14:50:05Z, and logged `Overriding data seed with 1` at 14:50:12Z.
