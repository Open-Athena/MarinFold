---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T21:48:34Z'
  experiment: exp279_models_exact_soft_contact_targets
  kind: models
  short_description: Exact soft targets on the default decontaminated recipe, 32 H100s,
    required validation
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b02
    entity: open-athena
    project: MarinFold
    run_id: exp279-soft-s0-cw-h100x32-b02
    run_name: exp279-soft-s0-cw-h100x32-b02
  git_sha: b0dd33eca8836c1dddbe6588e4befc3b2dd5c67d
  iris_job_ids:
  - /bizon/exp279-soft-pilot-cw-h100x32-a04
  - /bizon/exp279-soft-production-cw-h100x32-a01
  - /bizon/exp279-soft-production-cw-h100x32-a01/exp279-soft-production-cw-h100x32-a01-base
---
# 2026-09-09 · exp279_models_exact_soft_contact_targets · exp279-soft-s0-cw-h100x32-b02

**Launched:** 2026-09-09T21:48:34Z by bizon  
**Kind:** models  
**Experiment:** exp279_models_exact_soft_contact_targets  
**W&B:** [exp279-soft-s0-cw-h100x32-b02](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b02)  
**Git:** `b0dd33ec`  

## Description

Exact soft targets on the default decontaminated recipe, 32 H100s, required validation

## Detailed plan

Fresh exact-soft-target model using exp232 m2/p06's decontaminated AFDB/ESM
mixture, Qwen3 1.47B, context 8192, global batch 128 and prescribed schedule
through update 363000. Four H100 nodes on cw-us-east-02a, batch priority,
per-device batch 1 (microbatch 32, four accumulation steps).

The first 32 updates are a resumable pilot. Continue the same native state and
frozen manifest with the production phase driver after validating the pilot.

## Changes from previous runs

Validation now uses the actual AFDB validation cache and must be nonempty.
The earlier b01 pilot omitted validation and is not a continuation source.

## Provenance and outputs

Frozen manifest: `experiments/exp279_models_exact_soft_contact_targets/data/soft_pilot_cw_launch.json`.
Native and HF checkpoints: `s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-s0-cw-h100x32-b02/`.
The worker independently verifies the runtime source hash, dependency lock and
all three input cache ledgers. No bulk cross-region transfer was performed.

## Pilot result

Completed 32 updates with training loss 6.3659, ordinary validation CE 6.3678,
and 3.464 seconds/update. Verified permanent native step-31 state and HF export
with tokenizer. All four Iris tasks succeeded. Production resumes this state.

## Production resume

At 2026-09-09 21:59:58 UTC the base-phase workers completed update 32 after
restoring the pilot native checkpoint, with finite loss 6.3391. W&B reports the
same run as running. The driver stays alive and waits for this phase, then
continues the two prescribed transitions. Full training and accuracy evaluation
are not yet complete.
