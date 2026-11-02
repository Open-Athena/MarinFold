---
marinfold_run:
  user: bizon
  launched_at: '2026-10-13T15:10:50Z'
  experiment: exp364_models_mpnn_only_lr_sweep
  kind: models
  short_description: ProteinMPNN-only two-epoch 1.47B training at peak LR 0.002
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B
    run_name: contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B
  git_sha: 9accbfa72c2fd66159478fd5f2a8a16a6eb13b77
  iris_job_ids:
  - /bizon/exp364-lr2e-3-train-a01
  - /bizon/exp364-lr2e-3-train-a02
  - /bizon/exp364-lr2e-3-train-a03
  - /bizon/exp364-lr2e-3-train-a03/exp364-train-2176995f
  - /bizon/exp364-lr2e-3-train-a03/exp364-train-54370502
  - /bizon/exp364-lr2e-3-train-a04
---
# 2026-10-13 · exp364_models_mpnn_only_lr_sweep · contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B

**Launched:** 2026-10-13T15:10:50Z by bizon  
**Kind:** models  
**Experiment:** exp364_models_mpnn_only_lr_sweep  
**W&B:** [contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp364-mpnn-only-2ep-lr2e-3-1.5B)  
**Git:** `9accbfa7`  

## Description

ProteinMPNN-only two-epoch 1.47B training at peak LR 0.002

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Stopped early at the user's request on 2026-10-30T15:44:10.039Z because of
training instability. Iris root `/bizon/exp364-lr2e-3-train-a04` and all eight
tasks in its latest child `exp364-train-c84a0ccf` are verified killed, releasing
64 H100s. Last logged step: 255772 (approximately 1.369 redesign epochs);
latest validation loss: 3.67563. This trial did not complete two epochs.
Checkpoints and evaluation artifacts are retained. Do not restart without
explicit user authorization. The independent 5e-4 and 1e-3 runs continue.
