---
marinfold_run:
  user: zack
  launched_at: '2026-08-17T16:31:32Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 v2 compact soft-target smoke, bs24, 20 steps
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20
    run_name: exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20
  git_sha: d99a138a7c65f45cb29c84010e808228702952fc
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20-driver
  - /zack/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20-driver/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20
---

# 2026-08-17 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20

**Launched:** 2026-08-17T16:31:32Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20)  
**Git:** `d99a138a`  

## Description

8xH100 v2 compact soft-target smoke, bs24, 20 steps

## Detailed plan

Smoke-test whether a larger global batch improves exp177 soft-target throughput after the v2 compact precomputed dataset fixed the loader path. Use 1 node × 8 H100, tensor parallelism 8, global batch size 24, direct precomputed reader (`EXP177_PRECOMPUTED_MP=0`), `jax_flash` attention, and 20 training steps with no retries.

## Changes from previous runs

- Same v2 compact S3 dataset and compact `segment_ids` / `attention_blocks` path as r45.
- Increased global batch size from 16 to 24 while keeping 8 H100 and TP=8.
- Kept dataloader prefetch small (`EXP177_DATALOADER_PREFETCH_SIZE=2`) to avoid the old loader-buffer cliff.

## Notes

- Iris driver and child both succeeded.
- First batch loaded in 0.1s, confirming the v2 compact precomputed data path remains fast.
- First train step completed in 111.0s including compilation.
- Training progress reached 20/20 at about 78.8s/step averaged over the short smoke, slower than r45's bs16 smoke (~51.5s/step).
- Final eval loss was 19.363 on `tokenized/contacts-v1-val`.
- Final checkpoint saved at `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r47-v2compactmask-8gpu-bs24-smoke20/2026.08.17.r47/checkpoints/step-19`.
- PJRT `WatchTasksAsync` disconnect warnings appeared after W&B finalization and checkpoint save; Iris marked the run succeeded.
- Follow-up bs32 attempts r46 and r48 both dispatched but stayed running with zero child logs; both were stopped as CoreWeave pre-container/logless startup stalls, not training results.
