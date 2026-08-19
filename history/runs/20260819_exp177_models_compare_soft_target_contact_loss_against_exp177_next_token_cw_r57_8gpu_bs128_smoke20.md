---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T17:58:57Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: 8xH100 matched next-token stock smoke, bs128, 20 steps
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r57-8gpu-bs128-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-next-token-cw-r57-8gpu-bs128-smoke20
    run_name: exp177-next-token-cw-r57-8gpu-bs128-smoke20
  git_sha: 11462b0a4c5a13f0fa1f1ab8ab61369a6c850f84
  iris_job_ids:
  - /zack/exp177-next-token-cw-r57-8gpu-bs128-smoke20-driver
  - /zack/exp177-next-token-cw-r57-8gpu-bs128-smoke20-driver/exp177-next-token-cw-r57-8gpu-bs128-smoke20
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-next-token-cw-r57-8gpu-bs128-smoke20

**Launched:** 2026-08-19T17:58:57Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-next-token-cw-r57-8gpu-bs128-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r57-8gpu-bs128-smoke20)  
**Git:** `11462b0a`  

## Description

8xH100 matched next-token stock smoke, bs128, 20 steps

## Notes

- Iris driver and child both succeeded.
- Global batch size 128, 1 node × 8 H100, stock next-token CE, exp177 Qwen3 1.47B config.
- First batch loaded in 13.1s; the cache path works after making the train component a flat train cache and keeping `contacts-v1-val` as the only validation cache-backed eval component.
- First train step completed in 49.6s including compilation.
- Tqdm reached 15/20 at 10.5s/step. Wall time from first-step completion (18:01:48) to checkpoint start (18:05:09) gives ~10.6s/step for post-compile steps.
- Approximate token throughput: ~99k tok/s (`128 * 8192 / 10.6`). This is the best current matched stock baseline for comparing sparse soft-target throughput on the same 1×8 H100 shape.
- Final eval loss was 6.179 on `contacts-v1-val`.
- Final checkpoint saved at `s3://marin-us-east-02a/MarinFold/exp177_soft_target_loss_h2h_cw/checkpoints/exp177-next-token-cw-r57-8gpu-bs128-smoke20/2026.08.19.r57/checkpoints/step-19` with HF export at `.../hf/step-19`.
