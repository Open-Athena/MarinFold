---
marinfold_run:
  user: bizon
  launched_at: '2026-10-19T13:32:42Z'
  experiment: exp364_models_mpnn_only_lr_sweep
  kind: models
  short_description: Full-state fork of the 5e-4 redesign-only run with cooldown ending
    at exactly one epoch
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B
    run_name: contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B
  git_sha: 1e5136a2
  iris_job_ids:
  - /bizon/exp364-lr5e-4-cooldown-a01
  - /bizon/exp364-lr5e-4-cooldown-a01/exp364-train-4f1ab581
  - /bizon/exp364-lr5e-4-cooldown-a01/exp364-train-6ae81a1a
  - /bizon/exp364-lr5e-4-cooldown-a01/exp364-train-38093fa8
  - /bizon/exp364-lr5e-4-cooldown-a02
---
# 2026-10-19 · exp364_models_mpnn_only_lr_sweep · contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B

**Launched:** 2026-10-19T13:32:42Z by bizon  
**Kind:** models  
**Experiment:** exp364_models_mpnn_only_lr_sweep  
**W&B:** [contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B)  
**Git:** `1e5136a2`  

## Description

Full-state fork of the 5e-4 redesign-only run with cooldown ending at exactly one epoch

## Detailed plan

Restore the complete trainer state from the immutable copy of the two-epoch
5e-4 run's checkpoint 139,053. Continue its original data order from optimizer
step 139,054. End after exactly one redesign epoch: 186,885 total updates,
final checkpoint 186,884, with 37 real examples in the last batch.

## Changes from previous runs

Keep the parent model, optimizer moments, RNG state, data/augmentation seeds,
augmentation duration, validation, and global batch 128. Preserve the historical
37,376-step warmup, then hold LR 5e-4 until step 149,508 and linearly decay toward
5e-5 over the final 37,377 updates. The parent two-epoch run continues independently.

## Notes

One 8-H100 node on cw-us-east-02a at batch priority. W&B initialized at
2026-10-19T13:45:00Z after queue admission. At 14:28 UTC the fork had advanced to
step 139,261, with finite training loss and approximately 100,430 token slots/s.
The live saved configuration was checked against the intended source checkpoint,
one-epoch target, cooldown schedule, and unchanged augmentation duration.

Native checkpoints are under
`s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep/runs/contacts-v1-exp364-mpnn-only-1ep-lr5e-4-cooldown-1.5B/checkpoints/`.
Source copy provenance is recorded in the experiment's `data/cooldown_source.json`.
The initial W&B progress includes the inherited prefix; it is not all new compute.

Verified complete on October 29: final checkpoint 186884 was written at
2026-10-29T02:19:20 UTC; trainer, optimizer and LR counters are all 186885.
The recovery root succeeded at 02:21:29 UTC. Final full-validation loss is
3.03571987. A complete finite-weight scan of the evaluation export passed.
Native and export evidence is in
`experiments/exp364_models_mpnn_only_lr_sweep/data/gpu_cooldown_final_verification.json`.
