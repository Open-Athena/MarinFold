---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T18:56:02Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: Sparse soft-target 4x8 H100 bs128 10% pilot attempt; stopped
    after slow warmup projection.
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct
    run_name: exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct
  git_sha: 3adbb46c83d914051edebc4092901068786ed84a
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct-driver
  - /zack/exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct-driver/exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct

**Launched:** 2026-08-19T18:56:02Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r59-sparse-4x8-bs128-10pct)  
**Git:** `3adbb46c`  

## Description

Sparse soft-target 4x8 H100 bs128 10% pilot attempt; stopped after slow warmup projection.

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Stopped manually after warmup because the 4x8 bs128 sparse run projected roughly 112 hours for the 10% slice. Replaced by `exp177-soft-target-cw-r61-sparse-4x8-bs256-10pct`, which also proved too slow.
