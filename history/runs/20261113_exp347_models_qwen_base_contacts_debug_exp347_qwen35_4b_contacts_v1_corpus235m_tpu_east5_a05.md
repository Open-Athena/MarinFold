---
marinfold_run:
  user: bizon
  launched_at: '2026-11-13T04:50:39Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Bounded TPU validation; full-state restore passed, second microbatch
    exhausted HBM
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05
    entity: open-athena
    project: MarinFold
    run_id: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05
    run_name: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05
  git_sha: dce66418
  iris_job_ids:
  - /timodonnell/exp347-tpu-contacts_v1-validate-a05
---

# 2026-11-13 · exp347_models_qwen_base_contacts · debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05

**Launched:** 2026-11-13T04:50:39Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05](https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a05)  
**Git:** `dce66418`  

## Description

Bounded TPU validation; full-state restore passed, second microbatch exhausted HBM

## Detailed plan

Validate two optimizer updates and exact checkpoint recovery from the frozen
contacts-v1 parent step 59319, using one v5p-8 in us-east5-a. Full 16K context,
four-document microbatches, and a 32-document global batch preserve the GPU
training objective. All checkpoint I/O remains in us-east5.

## Changes from previous runs

The Adam step counter is explicitly replicated across the model mesh, fixing
the previous attempt's incompatible-device restore comparison.

## Notes

Full FP32 parameter/eight-shard Adam conversion completed in 768.9 seconds.
Converted-state restoration and exact comparison passed. The first microbatch
completed; the second gradient call exhausted HBM (120 MiB requested, 73.28 MiB
free). No optimizer update completed. Iris records one application failure and
one preemption. This is a failed bounded validation, not production training.

Follow-up a06 fuses gradient computation and accumulation with accumulator
donation. CPU full-batch Adam, different microbatch sizes, and exact resumed-update
checks pass at unchanged tolerances; TPU memory feasibility remains unproven.
