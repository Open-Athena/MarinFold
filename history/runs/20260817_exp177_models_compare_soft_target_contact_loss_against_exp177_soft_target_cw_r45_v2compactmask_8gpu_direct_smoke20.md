---
marinfold_run:
  user: zack
  launched_at: '2026-08-17T13:59:16Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 v2 compact soft-target smoke, bs16, 20 steps
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20
    run_name: exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20
  git_sha: 625d64d9a91263bb448af30b696ab0ee73452844
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20-driver
  - /zack/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20-driver/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20
---

# 2026-08-17 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20

**Launched:** 2026-08-17T13:59:16Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20)  
**Git:** `625d64d9`  

## Description

8xH100 v2 compact soft-target smoke, bs16, 20 steps

## Detailed plan

Smoke-test exp177 soft-target training with the v2 precomputed compact dataset on CoreWeave after H100 capacity returned. Use 1 node × 8 H100, tensor parallelism 8, global batch size 16, direct precomputed reader (`EXP177_PRECOMPUTED_MP=0`), `jax_flash` attention, and 20 training steps with no retries.

## Changes from previous runs

- Uses v2 precomputed rows from `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/preprocessed/soft_target_compact_v2/2026.08.15.1/`.
- Uses compact `segment_ids` / `attention_blocks` rather than dataset-carried dense attention masks.
- Runs from pushed branch commit `625d64d`, so pod-installed `marinfold-models` includes the compact batch schema.

## Notes

- Iris driver and child both succeeded.
- First batch loaded in 0.1s; the previous dense-mask path had first-batch latency around tens of seconds.
- First train step completed in 84.6s including compilation. Training progress reached 20/20 at roughly 51.5s/step averaged over the short smoke.
- Final eval loss was 17.453 on `tokenized/contacts-v1-val`.
- Final checkpoint saved at `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r45-v2compactmask-8gpu-direct-smoke20/2026.08.15.r45/checkpoints/step-19`.
- PJRT `WatchTasksAsync` disconnect warnings appeared after W&B finalization and checkpoint save; Iris marked the run succeeded.
