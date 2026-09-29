---
marinfold_run:
  user: zack
  launched_at: '2026-09-22T16:57:22Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Exp277-scale delta-stream V2 cache/training smoke (a02)
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a02
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-full-epoch-1_5b-smoke-a02
    run_name: delta-v2-exp277-full-epoch-1_5b-smoke-a02
  git_sha: c69d717086709f0e1db7c1856785c04cadf28770
  iris_job_ids:
  - /zack/exp299-v2-exp277-train-smoke-driver-a02
  - /zack/exp299-v2-exp277-train-smoke-driver-a02/delta-v2-exp277-full-epoch-1_5b-smoke-a02
---

# 2026-09-22 · exp299_models_delta_stream_contacts · delta-v2-exp277-full-epoch-1_5b-smoke-a02

**Launched:** 2026-09-22T16:57:22Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-full-epoch-1_5b-smoke-a02](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a02)  
**Git:** `c69d7170`  

## Description

Exp277-scale delta-stream V2 cache/training smoke (a02)

## Detailed plan

Retry the ten-update exp277-scale smoke with the one-split token cache loaded as
a flat Levanter cache.

## Changes from previous runs

- Avoids validation-split lookup by using `flat_cache=True`.

## Notes

Failed before training because `PrebuiltLmDatasetFormat` means the cache rows
are already fixed-length model examples; Levanter therefore did not pack the
variable-length documents and rejected a 4,573-token row for an 8,192-token
model axis. The next attempt uses the packable token-ID format adapter.
