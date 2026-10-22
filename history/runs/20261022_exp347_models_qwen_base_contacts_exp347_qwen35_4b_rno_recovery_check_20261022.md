---
marinfold_run:
  user: bizon
  launched_at: '2026-10-22T20:31:06Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Bounded RNO2A full-state recovery and regional checkpoint validation
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-rno-recovery-check-20261022
    entity: open-athena
    project: MarinFold
    run_id: exp347-qwen35-4b-rno-recovery-check-20261022
    run_name: exp347-qwen35-4b-rno-recovery-check-20261022
  git_sha: 990e36414f05dbb48cefbd6dc417d2dfa8bbba6c
  iris_job_ids:
  - /timodonnell/exp347-qwen35-4b-rno-recovery-check-20261022-a02
---

# 2026-10-22 · exp347_models_qwen_base_contacts · exp347-qwen35-4b-rno-recovery-check-20261022

**Launched:** 2026-10-22T20:31:06Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-qwen35-4b-rno-recovery-check-20261022](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-rno-recovery-check-20261022)  
**Git:** `990e3641`  

## Description

Bounded RNO2A full-state recovery and regional checkpoint validation

## Detailed plan

Restore raw production step 5229 / 574,648,070 tokens on eight RNO2A H100s under a separate identity. Run two full optimizer updates with the unchanged 100B schedule, save both, then exit. This validation does not advance the production run.

## Changes from previous runs

Commit 990e3641 resolves outputs to the compute-local bucket and adds one-time full-state relocation. CoreWeave reports BucketRegion=RNO2A for marin-rno2a. Source corpus URIs and seeded data order stay unchanged.

## Notes

First update completed at step 5230. Independently verified all 20 checkpoint objects (50,489,547,802 bytes), model/tokenizer, eight optimizer shards, and matching local completion/resume markers. Startup metadata finished in 23 ms. The first rejected submission never created an Iris root; a02 is the actual GPU validation.

Both updates completed; Iris succeeded with exit 0 and no preemptions or failures. The final local step-5231 checkpoint contains 574,901,468 tokens and all 20 objects (50,489,547,802 bytes). The preceding local checkpoint was deleted only after its successor committed. No R-precision evaluation was performed for this diagnostic.
