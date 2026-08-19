---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T18:56:03Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: Stock next-token 4x8 H100 bs256 10% training pilot with hourly
    rolling and 1%-cadence archive checkpoints.
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r60-4x8-bs256-10pct
    entity: open-athena
    project: MarinFold
    run_id: exp177-next-token-cw-r60-4x8-bs256-10pct
    run_name: exp177-next-token-cw-r60-4x8-bs256-10pct
  git_sha: 3adbb46c83d914051edebc4092901068786ed84a
  iris_job_ids:
  - /zack/exp177-next-token-cw-r60-4x8-bs256-10pct-driver
  - /zack/exp177-next-token-cw-r60-4x8-bs256-10pct-driver/exp177-next-token-cw-r60-4x8-bs256-10pct
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-next-token-cw-r60-4x8-bs256-10pct

**Launched:** 2026-08-19T18:56:03Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-next-token-cw-r60-4x8-bs256-10pct](https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r60-4x8-bs256-10pct)  
**Git:** `3adbb46c`  

## Description

Stock next-token 4x8 H100 bs256 10% training pilot with hourly rolling and 1%-cadence archive checkpoints.

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Running. First archive checkpoint succeeded at step 357:

`s3://marin-us-east-02a/MarinFold/exp177_soft_target_loss_h2h_cw/checkpoints/exp177-next-token-cw-r60-4x8-bs256-10pct/2026.08.19.r60/checkpoints/step-357`

Eval at step 357: `contacts-v1-val loss = 4.133`. Steady-state training rate is about 5.3 s/step, with roughly 4.5 hours remaining after the first checkpoint/eval.
