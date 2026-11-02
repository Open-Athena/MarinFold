---
marinfold_run:
  user: zack
  launched_at: '2026-08-30T16:47:09Z'
  experiment: exp157_models_fixed_position_embeddings
  kind: models
  short_description: Qwen3 1.5B contacts-v1 control-matched RoPE plus learned position-delta
    run
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8
    entity: open-athena
    project: MarinFold
    run_id: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8
    run_name: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8
  git_sha: 5b3920c913060b8da8fc78ec3ea2b799b8a1517b
  iris_job_ids:
  - /zack/exp157-rope-delta-qwen3-controlmatch-r1-east02-driver
  - /zack/exp157-rope-delta-qwen3-controlmatch-r1-east02-driver/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8
---

# 2026-08-30 · exp157_models_fixed_position_embeddings · exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8

**Launched:** 2026-08-30T16:47:09Z by zack  
**Kind:** models  
**Experiment:** exp157_models_fixed_position_embeddings  
**W&B:** [exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8](https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs128-qwen3-rope_delta-position-controlmatch-r1-east02-h100x8)  
**Git:** `5b3920c9`  

## Description

Qwen3 1.5B contacts-v1 control-matched RoPE plus learned position-delta run

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Cancelled on 2026-08-30 after the request to switch the matched rope-delta run
to 8x4 GB200 placement. It reached W&B/global-step logging but was stopped before
the first full-validation point.
