---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T21:35:35Z'
  experiment: exp279_models_exact_soft_contact_targets
  kind: models
  short_description: Full-model soft-target pilot on the default decontaminated mixture,
    32 H100s
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b01
    entity: open-athena
    project: MarinFold
    run_id: exp279-soft-s0-cw-h100x32-b01
    run_name: exp279-soft-s0-cw-h100x32-b01
  git_sha: b80b42d4
  iris_job_ids:
  - /bizon/exp279-soft-pilot-cw-h100x32-a03
---

# 2026-09-09 · exp279_models_exact_soft_contact_targets · exp279-soft-s0-cw-h100x32-b01

**Launched:** 2026-09-09T21:35:35Z by bizon  
**Kind:** models  
**Experiment:** exp279_models_exact_soft_contact_targets  
**W&B:** [exp279-soft-s0-cw-h100x32-b01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-s0-cw-h100x32-b01)  
**Git:** `b80b42d4`  

## Description

Full-model soft-target pilot on the default decontaminated mixture, 32 H100s

## Result

Completed 32 updates (step-31) on four H100 nodes, with finite final training
loss 6.3657 and roughly 3.49 seconds/update (300,487 nominal tokens/second).
Native state and HF weights/tokenizer were verified in the region-local S3
prefix `MarinFold/exp279/checkpoints/exp279-soft-s0-cw-h100x32-b01/`.

Validation was silently absent because flat caches only supply training in the
pinned Levanter loader. This run is a throughput/startup pilot only and will not
be resumed for the scientific production experiment. A new run starts after the
validation fix. No scientific accuracy conclusions follow from this pilot.
