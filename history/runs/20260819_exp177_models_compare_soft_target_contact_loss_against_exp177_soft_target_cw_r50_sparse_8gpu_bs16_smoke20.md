---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T16:08:34Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 sparse soft-target smoke, bs16, 20 steps
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20
    run_name: exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20
  git_sha: 34f4ce7236e713c7343b54e117b3151f0f96762a
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20-driver
  - /zack/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20-driver/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20

**Launched:** 2026-08-19T16:08:34Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20)  
**Git:** `34f4ce77`  

## Description

8xH100 sparse soft-target smoke, bs16, 20 steps

## Detailed plan

Smoke-test the sparse soft-target batch/loss path without recomputing or materializing new data-side artifacts. Use the exp139 analyzed S3 shards directly, construct sparse second-endpoint supports on the fly with `EXP177_PRECOMPUTED_MP=0`, `EXP177_SOFT_TARGET_BATCH=sparse`, `EXP177_MAX_SPARSE_CONTACTS=2048`, and `EXP177_MAX_SPARSE_DEGREE=32`, then train for 20 steps on 1 node × 8 H100 with tensor parallelism 8 and global batch size 16.

## Changes from previous runs

- Replaced the v2 compact padded soft-target loss path with the sparse soft-target loss path from `b862e20`.
- Updated the experiment lockfile to include the sparse MarinFold packages at `34f4ce7` after r49 failed because the remote env still resolved old packages.
- Kept global batch size 16, 8 H100, direct precomputed reader, and JAX flash attention comparable to r45.

## Notes

- Iris driver and child both succeeded.
- First batch loaded in 0.1s.
- First train step completed in 43.6s including tracing/lowering/compilation.
- Training progress reached 17/20 at 9.2s/step in the tqdm rate readout; using wall time from first-step completion (16:11:10) to checkpoint start (16:14:13), the post-compile training phase averaged about 9.6s/step. Including first-step compile from first batch to checkpoint start averaged about 11.4s/step over 20 steps.
- This is a large improvement over the compact padded bs16 smoke r45 (~51.5s/step) and compact padded bs24 smoke r47 (~78.8s/step).
- Final eval loss was 17.172 on `tokenized/contacts-v1-val`.
- Final checkpoint saved at `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20/2026.08.19.r50/checkpoints/step-19`.
- A non-fatal draccus config-encoding exception appeared for `SparsePrecomputedSoftTargetContactsDataset` during startup; training continued and Iris marked the run succeeded.
- PJRT `WatchTasksAsync` disconnect warnings appeared after W&B finalization and checkpoint save; Iris marked the run succeeded.
