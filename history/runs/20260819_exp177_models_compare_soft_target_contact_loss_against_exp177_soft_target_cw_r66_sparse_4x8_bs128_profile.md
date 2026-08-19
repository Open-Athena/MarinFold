---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T23:23:35Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: Sparse soft-target 4x8 H100 bs128 one-step JAX profile with S3
    trace persistence.
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r66-sparse-4x8-bs128-profile
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r66-sparse-4x8-bs128-profile
    run_name: exp177-soft-target-cw-r66-sparse-4x8-bs128-profile
  git_sha: 9ce25c6aa73b8d3c3c18cff553dae3bef7eba7c5
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r66-sparse-4x8-bs128-profile-driver
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r66-sparse-4x8-bs128-profile

**Launched:** 2026-08-19T23:23:35Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r66-sparse-4x8-bs128-profile](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r66-sparse-4x8-bs128-profile)  
**Git:** `9ce25c6a`  

## Description

Sparse soft-target 4x8 H100 bs128 one-step JAX profile with S3 trace persistence.

## Detailed plan

Retry sparse 4x8 H100 bs128 profiling after moving Levanter's profiler output to a local pod path and uploading artifacts to S3 at process exit.

## Changes from previous runs

- Uses commit `9ce25c6`, which sets profiler `log_dir` to `/tmp/exp177-levanter-logs` and uploads local traces to S3.
- Profiles one train step: `EXP177_CW_PROFILER_START_STEP=8`, `EXP177_CW_PROFILER_NUM_STEPS=1`.

## Notes

- Iris job succeeded; W&B run: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r66-sparse-4x8-bs128-profile
- Trace artifacts uploaded under `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r66-sparse-4x8-bs128-profile/2026.08.19.r66/profile-logs/`.
- Levanter's separate XProf-to-GCS upload still emitted GCS auth tracebacks, but the job completed and the local artifacts were uploaded to S3.
- Profile summary showed a 5M-event trace cap, ~265k `command_buffer::execute` events per host, ~197k NCCL send/recv kernels, ~69k all-reduce kernels, ~67k all-gather kernels, and ~49k `all-to-all.8` events in a single profiled step.
