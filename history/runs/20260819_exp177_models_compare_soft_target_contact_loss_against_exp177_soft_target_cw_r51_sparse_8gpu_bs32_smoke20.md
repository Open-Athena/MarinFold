---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T16:25:47Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 sparse soft-target batch-size smoke, bs32, 20 steps
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20
    run_name: exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20
  git_sha: e6655efcc94de3f4e139eed302356cf0f8cd2556
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20-driver
  - /zack/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20-driver/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20

**Launched:** 2026-08-19T16:25:47Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20)  
**Git:** `e6655efc`  

## Description

8xH100 sparse soft-target batch-size smoke, bs32, 20 steps

## Notes

- Iris driver and child both succeeded.
- Global batch size 32, 1 node × 8 H100, TP=8, sparse precomputed soft-target path.
- First batch loaded in 0.2s.
- First train step completed in 53.0s including compilation.
- Tqdm reached 19/20 at 18.1s/step. Wall time from first-step completion (16:27:52) to checkpoint start (16:33:46) gives ~18.6s/step for post-compile steps.
- Approximate token throughput: ~14.1-14.5k tok/s.
- Final eval loss was 16.236 on `tokenized/contacts-v1-val`.
- Final checkpoint saved at `s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r51-sparse-8gpu-bs32-smoke20/2026.08.19.r51/checkpoints/step-19`.
