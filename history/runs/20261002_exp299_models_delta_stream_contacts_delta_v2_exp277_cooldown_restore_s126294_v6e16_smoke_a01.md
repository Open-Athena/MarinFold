---
marinfold_run:
  user: zack
  launched_at: '2026-10-02T17:19:06Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Full-state GPU-to-TPU restore smoke from exp299 step 126294 on
    v6e-16
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01
    run_name: delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01
  git_sha: fa695ba5d37c935122c8dafe0dd1f2879ccfedbe
  iris_job_ids:
  - /zack/delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01
---

# 2026-10-02 · exp299_models_delta_stream_contacts · delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01

**Launched:** 2026-10-02T17:19:06Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown-restore-s126294-v6e16-smoke-a01)  
**Git:** `fa695ba5`  

## Description

Full-state GPU-to-TPU restore smoke from exp299 step 126294 on v6e-16

## Detailed plan

Verify that the complete step-126,294 GB200 checkpoint can restore model,
optimizer, training RNG, global step, and finite-loader position on TPU, then
run two updates, a two-batch validation, and a complete checkpoint write.

## Changes from previous runs

- Source checkpoint was mirrored from rhoarnet S3 to the co-located
  `marin-us-east5` GCS bucket (28 objects, 17,714,954,468 bytes).
- Backend changed from 16 GB200 GPUs to one `v6e-16` slice while preserving
  global batch 128, sequence length 8,192, corpus order, and full training state.
- Smoke end step was 126,297, so its compressed 40%-decay scheduler is not a
  scientific continuation result.

## Notes

Succeeded. Levanter restored 16.50 GiB across four hosts and explicitly resumed
from step 126,295; each host read 4.12 GiB in 3.6–4.5 seconds. Updates 126,295
and 126,296 completed, with final train loss 0.8561. Two-batch validation loss
was 1.06911, consistent with the source run near this checkpoint. The complete
step-126,296 output checkpoint committed successfully. Steady-state was 134,729
tokens/s (7.783 seconds/update, 12.56% instantaneous MFU).
