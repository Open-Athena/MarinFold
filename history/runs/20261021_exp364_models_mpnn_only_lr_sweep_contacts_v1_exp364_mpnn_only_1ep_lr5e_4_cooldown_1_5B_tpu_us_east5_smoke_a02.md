---
marinfold_run:
  user: bizon
  launched_at: '2026-10-21T16:12:49Z'
  experiment: exp364_models_mpnn_only_lr_sweep
  kind: models
  short_description: Ten-update full-state TPU restore and cooldown smoke on v5p-8;
    separate from GPU and production identities.
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02
    run_name: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02
  git_sha: ea6239b8
  iris_job_ids:
  - /bizon/exp364-tpu-cooldown-smoke-a02
  - /bizon/exp364-tpu-cooldown-smoke-a02/exp364-tpu-cooldown-f331bc07
---

# 2026-10-21 · exp364_models_mpnn_only_lr_sweep · contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02

**Launched:** 2026-10-21T16:12:49Z by bizon  
**Kind:** models  
**Experiment:** exp364_models_mpnn_only_lr_sweep  
**W&B:** [contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5-smoke-a02)  
**Git:** `ea6239b8`  

## Description

Ten-update full-state TPU restore and cooldown smoke on v5p-8; separate from GPU and production identities.

## Detailed plan

Restore the immutable parent checkpoint 139053, preserving optimizer/RNG/data state; execute steps 139054–139063 with an accelerated ten-update linear cooldown. Verify finite metrics, native checkpoint counters and HF export plus tokenizer before launching the independent production copy.

## Changes from previous runs

Uses four-chip v5p-8 in us-east5-a, regional GCS inputs, Splash attention and microbatch one with global batch 128. The preceding unregistered v5p-64 smoke was cancelled while its pool was in provisioning backoff. Neither existing GPU trajectory was changed.

## Notes

Registered in W&B at 2026-10-21 16:15 UTC. Validation is capped at two batches for this pipeline check and must not be compared with full production validation.
