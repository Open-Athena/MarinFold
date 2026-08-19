---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T16:59:47Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 sparse soft-target batch-size smoke, bs128, stopped after poor scaling signal
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8
    entity: open-athena
    project: MarinFold
    run_id: exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8
    run_name: exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8
  git_sha: e6655efcc94de3f4e139eed302356cf0f8cd2556
  iris_job_ids:
  - /zack/exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8-driver
  - /zack/exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8-driver/exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8

**Launched:** 2026-08-19T16:59:47Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8](https://wandb.ai/open-athena/MarinFold/runs/exp177-soft-target-cw-r53-sparse-8gpu-bs128-smoke8)  
**Git:** `e6655efc`  

## Description

8xH100 sparse soft-target batch-size smoke, bs128, stopped after poor scaling signal

## Notes

- Global batch size 128, 1 node × 8 H100, TP=8, sparse precomputed soft-target path.
- First batch loaded in 0.5s.
- First train step completed in 115.6s including compilation.
- Reached 3/8 at 94.7s/step in the tqdm rate readout. Wall time from first-step completion (17:02:56) to the 3/8 progress line (17:05:33) gives a less compile-biased early estimate of ~78.5s/step across the following two steps.
- Approximate token throughput: ~11-13k tok/s, worse than bs32/bs64.
- No immediate OOM was observed. The run was intentionally stopped after confirming poor batch-size scaling.
