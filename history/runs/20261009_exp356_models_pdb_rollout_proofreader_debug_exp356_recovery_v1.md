---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T21:17:56Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: One-H100 proofreading smoke with pause at step 6 and resume to
    step 12
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-recovery-v1
    entity: open-athena
    project: MarinFold
    run_id: debug-exp356-recovery-v1
    run_name: debug-exp356-recovery-v1
  git_sha: 05e009885731ae3bb146bb5f055fe02cc00684e7
  iris_job_ids:
  - /bizon/exp356-train-smoke-a01
  - /bizon/exp356-train-smoke-a02
---
# 2026-10-09 · exp356_models_pdb_rollout_proofreader · debug-exp356-recovery-v1

**Launched:** 2026-10-09T21:17:56Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [debug-exp356-recovery-v1](https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-recovery-v1)  
**Git:** `05e00988`  

## Description

One-H100 proofreading smoke with pause at step 6 and resume to step 12

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

The initial job used clean commit `05e00988` (Iris tree `7f0a9429`). The recovery job used base commit `4aafce9e` and source-manifest SHA256 `3fccd291433da9f6f83779f9518b3ad08156ef841a72caf6cb588ac3b97f5f9e`. It reloaded step 6, reproduced all validation metrics exactly, and completed step 12. This run verifies execution and recovery only.
