---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T18:54:45Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: Ten-step 128-H100 startup validation for the native/MPNN pilot
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke
    run_name: contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke
  git_sha: 656214dc8ad1ee507c2d832c345f0ae6b45f397c
  iris_job_ids:
  - /bizon/exp277-train-smoke-a01
  - /bizon/exp277-train-smoke-a01/exp277-train-eeab3847
---

# 2026-09-09 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke

**Launched:** 2026-09-09T18:54:45Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-native-mpnn-1.5B-smoke)  
**Git:** `656214dc`  

## Description

Ten-step 128-H100 startup validation for the native/MPNN pilot

## Detailed plan

Validate the production 16-node / 128-H100 gang for ten updates, using the audited small redesign caches, native training caches, and two validation batches. Fixed exp232 m2-p06 optimizer and model configuration.

## Changes from previous runs

Uses the native/MPNN four-way mixture and Marin 0.2.86, the nearest runtime release accepted by the current Iris controller. Smoke and production have separate W&B and checkpoint identities.

## Notes

Succeeded on 2026-09-09. W&B finished with run_progress=1 and global_step=9. Final train loss 6.43476, two-batch eval loss 6.31338. The final measured step took 0.86565 seconds (1.211M tokens/s). Both Levanter step-9 and HF step-9 outputs were verified in the experiment S3 runs directory. The HF export contains both safetensors shards, tokenizer.json, and tokenizer_config.json. This is startup validation, not a model-quality result.
