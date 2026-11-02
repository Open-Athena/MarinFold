---
marinfold_run:
  user: zack
  launched_at: '2026-07-30T15:20:53Z'
  experiment: exp124_models_contacts_v1_think_loss_masked
  kind: models
  short_description: Train contacts-v1 think-token masked-loss Qwen3 1.5B from scratch
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3
    entity: open-athena
    project: MarinFold
    run_id: exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3
    run_name: exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3
  git_sha: c1e87142e902449e61164859c09614772577800d
  iris_job_ids:
  - /zack/exp124-train-full-20260731-1413-resume-r5
  - /zack/exp124-train-full-20260731-1744-resume-auto
---

# 2026-07-30 · exp124_models_contacts_v1_think_loss_masked · exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3

**Launched:** 2026-07-30T15:20:53Z by zack  
**Kind:** models  
**Experiment:** exp124_models_contacts_v1_think_loss_masked  
**W&B:** [exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3](https://wandb.ai/open-athena/MarinFold/runs/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3)  
**Git:** `c1e87142`  

## Description

Train contacts-v1 think-token masked-loss Qwen3 1.5B from scratch

## Detailed plan

Train the exp177/exp117-style Qwen3 1.5B contacts-v1 recipe from scratch on the think-token transformed cache, with `<think>` causal targets masked out of the loss. Global batch 256, sequence length 8192, v5p-128 in `us-east5-a`.

## Changes from previous runs

- Uses `contacts-v1-think-masked` cache at `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2`.
- Trains on masked think-token examples while also evaluating `tokenized/contacts-v1-val` with zero train weight.

## Notes

- Final driver `/zack/exp124-train-full-20260731-1744-resume-auto` succeeded after earlier resume attempts.
- Output artifact status is `SUCCESS` at `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/checkpoints/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3/2026.07.30.4`.
- Final checkpoint/HF export: `step-35680`.
- Final W&B summary: `eval/tokenized/contacts-v1-val/loss = 3.131303071975708`, `eval/contacts-v1-think-masked/loss = 3.0855870246887207`, `train/loss = 3.0132369995117188`, `global_step = 35680`.
