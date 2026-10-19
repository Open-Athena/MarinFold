---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T21:26:03Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: 'Eight-H100 DDP smoke: global batch 64, variable rollout prefixes,
    checkpoint and validation'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-eight-h100-v1
    entity: open-athena
    project: MarinFold
    run_id: debug-exp356-eight-h100-v1
    run_name: debug-exp356-eight-h100-v1
  git_sha: 4aafce9e778c4d040609b10c90dd4218cab69433
  iris_job_ids:
  - /bizon/exp356-train-eight-gpu-smoke-a01
---

# 2026-10-09 · exp356_models_pdb_rollout_proofreader · debug-exp356-eight-h100-v1

**Launched:** 2026-10-09T21:26:03Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [debug-exp356-eight-h100-v1](https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-eight-h100-v1)  
**Git:** `4aafce9e`  

## Description

Eight-H100 DDP smoke: global batch 64, variable rollout prefixes, checkpoint and validation

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Completed four optimizer steps with eight H100s and global batch 64. Source-manifest SHA256: `3fccd291433da9f6f83779f9518b3ad08156ef841a72caf6cb588ac3b97f5f9e`. Validation loss fell from 0.913 to 0.684; this execution smoke does not establish production quality. The saved step-4 model was subsequently used to verify the full evaluation/reporting pipeline.
