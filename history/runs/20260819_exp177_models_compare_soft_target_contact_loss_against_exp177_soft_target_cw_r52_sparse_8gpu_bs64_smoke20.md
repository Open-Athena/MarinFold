---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T16:42:15Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 sparse soft-target batch-size smoke, bs64, stopped after timing signal
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20
    run_name: exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20
  git_sha: e6655efcc94de3f4e139eed302356cf0f8cd2556
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20-driver
  - /zack/exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20-driver/exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20

**Launched:** 2026-08-19T16:42:15Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r52-sparse-8gpu-bs64-smoke20)  
**Git:** `e6655efc`  

## Description

8xH100 sparse soft-target batch-size smoke, bs64, stopped after timing signal

## Notes

- Global batch size 64, 1 node × 8 H100, TP=8, sparse precomputed soft-target path.
- First batch loaded in 0.4s.
- First train step completed in 73.2s including compilation.
- Reached 15/20 at 36.2s/step in the tqdm rate readout. Wall time from first-step completion (16:44:39) to the 15/20 progress line (16:53:14) gives ~36.8s/step for post-compile steps.
- Approximate token throughput: ~14.2-14.5k tok/s, essentially tied with bs32 and not better than bs32.
- No OOM was observed. The run was intentionally stopped during an interval checkpoint after enough timing signal was collected.
