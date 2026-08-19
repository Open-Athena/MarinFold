---
marinfold_run:
  user: zack
  launched_at: '2026-08-19T17:55:58Z'
  experiment: exp177_models_compare_soft_target_contact_loss_against
  kind: models
  short_description: failed next-token stock smoke attempt before train due missing validation cache
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r56-8gpu-bs128-smoke20
    entity: open-athena
    project: MarinFold
    run_id: exp177-next-token-cw-r56-8gpu-bs128-smoke20
    run_name: exp177-next-token-cw-r56-8gpu-bs128-smoke20
  git_sha: 51dcf4a8d7f58f0e6e764d790f509043453b8bb5
  iris_job_ids:
  - /zack/exp177-next-token-cw-r56-8gpu-bs128-smoke20-driver
  - /zack/exp177-next-token-cw-r56-8gpu-bs128-smoke20-driver/exp177-next-token-cw-r56-8gpu-bs128-smoke20
---

# 2026-08-19 · exp177_models_compare_soft_target_contact_loss_against · exp177-next-token-cw-r56-8gpu-bs128-smoke20

**Launched:** 2026-08-19T17:55:58Z by zack  
**Kind:** models  
**Experiment:** exp177_models_compare_soft_target_contact_loss_against  
**W&B:** [exp177-next-token-cw-r56-8gpu-bs128-smoke20](https://wandb.ai/open-athena/MarinFold/runs/exp177-next-token-cw-r56-8gpu-bs128-smoke20)  
**Git:** `51dcf4a8`  

## Description

Failed next-token stock smoke attempt before train due missing validation cache.

## Notes

- The run initialized W&B but did not train.
- Failure: the train `contacts-v1` component pointed at the cache root with `split="train"`; Levanter still tried to validate that component at `contacts-v1/validation`, which does not exist in the exp108 S3 cache.
- Fixed in `11462b0` by making the train component a flat cache at `contacts-v1/train`, so validation skips it and evaluates only `contacts-v1-val`.
