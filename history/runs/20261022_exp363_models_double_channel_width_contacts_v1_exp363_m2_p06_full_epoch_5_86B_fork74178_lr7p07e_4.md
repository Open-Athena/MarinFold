---
marinfold_run:
  user: bizon
  launched_at: '2026-10-22T21:08:34Z'
  experiment: exp363_models_double_channel_width
  kind: models
  short_description: Full-state LR fork from step 74178; initial startup failed before
    training
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
    run_name: contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
  git_sha: 9d0d088e18f9cd7393293e9ac490a4f44b6beeae
  iris_job_ids:
  - /bizon/exp363-fork-train-a02
  - /bizon/exp363-fork-train-a02/exp363-train-755515df
---

# 2026-10-22 · exp363_models_double_channel_width · contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4

**Launched:** 2026-10-22T21:08:34Z by bizon  
**Kind:** models  
**Experiment:** exp363_models_double_channel_width  
**W&B:** [contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4)  
**Git:** `9d0d088e`  

## Description

Full-state LR fork from step 74178; initial startup failed before training

## Detailed plan

Restore complete training state from checkpoint 74,178 and lower the 0.001 LR
linearly to 0.000707 over 2,114 updates. Retain the original final cooldown at
step 213,076 and 266,345-update horizon, alongside the original control run.

## Changes from previous runs

Independent W&B/checkpoint paths; full Adam state, RNG and data cursor retained.
Only the plateau LR changes; final cooldown retains its original timing and
10% floor ratio.

## Notes

October 23 inspection: startup failed before any training metrics. TensorStore's
default AWS credential chain could not source credentials, and the first data
cache metadata read failed with 403 AccessDenied. The child exited zero and
W&B says finished, but the driver's final-checkpoint guard correctly failed.

Runtime now explicitly passes AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from the
CoreWeave credentials. All 13 local tests pass. Recovery awaits refreshed local
credentials: the configured cw key is rejected with InvalidAccessKeyId by the
public CoreWeave endpoint. No retry has been submitted with the rejected key.
