---
marinfold_run:
  user: bizon
  launched_at: '2026-11-17T10:34:37Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Bounded TPU validation; fused accumulation still exceeds HBM
    before an optimizer update
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08
    entity: open-athena
    project: MarinFold
    run_id: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08
    run_name: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08
  git_sha: f5094a72
  iris_job_ids:
  - /timodonnell/exp347-tpu-contacts_v1-validate-a08
---

# 2026-11-17 · exp347_models_qwen_base_contacts · debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08

**Launched:** 2026-11-17T10:34:37Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08](https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a08)  
**Git:** `f5094a72`  

## Description

Bounded TPU validation; fused accumulation still exceeds HBM before an optimizer update

## Detailed plan

Two optimizer updates and full recovery validation on one v5p-8 in us-east5-a,
with four-document microbatches, global batch 32, full 16K context and frozen
contacts-v1 parent step 59319. Output storage remains in us-east5.

## Changes from previous runs

Same fused accumulation code as a06; this retry followed its source-bundle
download failure. a07 was rejected before creation by the client-date gate.
Only local client build metadata changed before this submission; the CPU
canary succeeded. Training code is from f5094a72, with updated result artifacts
in the submission bundle.

## Notes

Failed November 17 at 10:39 UTC with zero preemptions. Full FP32/eight-shard
Adam conversion completed in 509.5 seconds and exact converted-checkpoint
comparison passed. First microbatch completed; loading `jit_accumulate` for the
second required 70.05G with 68.30G available (`RuntimeProgramAllocationFailure`).
No optimizer update or validation checkpoint completed. W&B registered at
10:34:37 UTC; this is a failed bounded validation, not a production fork.
Unchanged retries are paused pending memory/sharding diagnosis.
