---
marinfold_run:
  user: bizon
  launched_at: '2026-11-09T16:38:47Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Bounded Qwen TPU full-state validation; OOM during second microbatch
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03
    entity: open-athena
    project: MarinFold
    run_id: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03
    run_name: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03
  git_sha: b38c2352
  iris_job_ids:
  - /timodonnell/exp347-tpu-contacts_v1-validate-a03
---

# 2026-11-09 · exp347_models_qwen_base_contacts · debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03

**Launched:** 2026-11-09T16:38:47Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03](https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a03)  
**Git:** `b38c2352`  

## Description

Bounded Qwen TPU full-state validation; OOM during second microbatch

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

_(Anything else worth remembering — preemptions, midway tweaks, etc.)_

November10 observation: all426 FP32 parameters and eight optimizer shards loaded; conversion took840.6s. Converted regional checkpoint saved. First of four microbatches completed; second failed with HBM exhaustion (90MB requested,6.46MB free). No optimizer update completed. Retry a04 uses four documents per microbatch with eight accumulation steps, preserving the global32-document update.
