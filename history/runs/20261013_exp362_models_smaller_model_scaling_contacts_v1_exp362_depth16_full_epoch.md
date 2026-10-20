---
marinfold_run:
  user: bizon
  launched_at: '2026-10-13T14:59:08Z'
  experiment: exp362_models_smaller_model_scaling
  kind: models
  short_description: '985M depth-only Qwen3: complete default corpus, 266345 updates
    on 64 H100s'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth16-full-epoch
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp362-depth16-full-epoch
    run_name: contacts-v1-exp362-depth16-full-epoch
  git_sha: 5c036e962efe4b8f4156b1d8d89623c968bd1ab2
  iris_job_ids:
  - /bizon/exp362-depth16-train-a01
  - /bizon/exp362-depth16-train-a01/exp362-train-8f65895e
  - /bizon/exp362-depth16-train-a01/exp362-train-a45b279a
---
# 2026-10-13 · exp362_models_smaller_model_scaling · contacts-v1-exp362-depth16-full-epoch

**Launched:** 2026-10-13T14:59:08Z by bizon  
**Kind:** models  
**Experiment:** exp362_models_smaller_model_scaling  
**W&B:** [contacts-v1-exp362-depth16-full-epoch](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth16-full-epoch)  
**Git:** `5c036e96`  

## Description

985M depth-only Qwen3: complete default corpus, 266345 updates on 64 H100s

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

At 2026-10-14T20:33Z, W&B retained a high water of 67,623 completed updates and marked the interrupted process crashed; its last heartbeat was 17:23:39Z. The exact child reported two preemptions, zero failures, and all eight tasks gated on Kueue admission. Preserve the queued dispatch while the eligible east02 cluster has no free H100s. Native step-53268 metadata, manifests and OCDBT data are reachable; the state was not loaded, and an HF export at that intermediate step is absent. Updates after that native checkpoint may need to repeat on resume.
