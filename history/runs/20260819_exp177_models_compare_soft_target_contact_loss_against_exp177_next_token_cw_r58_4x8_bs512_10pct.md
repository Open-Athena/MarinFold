---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T18:56:02Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: Stock next-token 4x8 H100 bs512 10% pilot attempt; stopped after
    GPU OOM.
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r58-4x8-bs512-10pct
    entity: open-athena
    project: MarinFold
    run_id: exp177-next-token-cw-r58-4x8-bs512-10pct
    run_name: exp177-next-token-cw-r58-4x8-bs512-10pct
  git_sha: 3adbb46c83d914051edebc4092901068786ed84a
  iris_job_ids:
  - /zack/exp177-next-token-cw-r58-4x8-bs512-10pct-driver
  - /zack/exp177-next-token-cw-r58-4x8-bs512-10pct-driver/exp177-next-token-cw-r58-4x8-bs512-10pct
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-next-token-cw-r58-4x8-bs512-10pct

**Launched:** 2026-08-19T18:56:02Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-next-token-cw-r58-4x8-bs512-10pct](https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r58-4x8-bs512-10pct)  
**Git:** `3adbb46c`  

## Description

Stock next-token 4x8 H100 bs512 10% pilot attempt; stopped after GPU OOM.

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Stopped manually after GPU OOM warnings around step 9 (`31.12GiB` allocation failures during train hooks). Replaced by `exp177-next-token-cw-r60-4x8-bs256-10pct`.
