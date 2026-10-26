---
marinfold_run:
  user: bizon
  launched_at: '2026-10-21T16:51:58Z'
  experiment: exp364_models_mpnn_only_lr_sweep
  kind: models
  short_description: Independent v5p-64 one-epoch 5e-4 cooldown, restored from full
    state at step 139053
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5
    run_name: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5
  git_sha: 33cc077f
  iris_job_ids:
  - /bizon/exp364-tpu-cooldown-train-a01
  - /bizon/exp364-tpu-cooldown-train-a01/exp364-tpu-cooldown-5e68abef
  - /bizon/exp364-tpu-cooldown-train-a02
---
# 2026-10-21 · exp364_models_mpnn_only_lr_sweep · contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5

**Launched:** 2026-10-21T16:51:58Z by bizon  
**Kind:** models  
**Experiment:** exp364_models_mpnn_only_lr_sweep  
**W&B:** [contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B-tpu-us-east5)  
**Git:** `33cc077f`  

## Description

Independent v5p-64 one-epoch 5e-4 cooldown, restored from full state at step 139053

## Detailed plan

Run an independent copy of the GPU one-epoch cooldown trajectory on v5p-64 in us-east5-a. Restore the complete original state at step 139053, begin at update 139054, start cooldown at 149508, and finish at step 186884 after exactly one redesign-corpus epoch. Preserve batch 128, data permutation, optimizer state, and the original augmentation ramp.

## Changes from previous runs

TPU Splash attention and a separately locked TPU runtime replace the GPU execution backend. The scientific configuration is unchanged. The GPU fork and two-epoch parent retain independent identities and continue running. Inputs and outputs use gs://marin-us-east5/protein-structure/MarinFold/exp364_models_mpnn_only_lr_sweep/.

## Notes

The approved 267.73 GB transfer was independently verified before the ten-update v5p-8 smoke. Production starts from the original frozen source, not the accelerated smoke endpoint. W&B initialized at 2026-10-21T16:51:58Z. At 17:50 UTC production had reached step 139702, with full validation loss 3.103912830352783 and 244184.40967320377 packed token slots/s. These are preliminary measurements before cooldown.
