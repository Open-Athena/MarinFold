---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T22:44:02Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: Eight-H100 frozen causal backbone and bidirectional contact encoder
    smoke with optimizer recovery
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-frozen-contact-v2
    entity: open-athena
    project: MarinFold
    run_id: debug-exp356-frozen-contact-v2
    run_name: debug-exp356-frozen-contact-v2
  git_sha: ce02fa292f5e56cad615534bc2b53192071b6467
  iris_job_ids:
  - /bizon/exp356-train-frozen-smoke-a01
  - /bizon/exp356-train-frozen-smoke-a02
---
# 2026-10-09 · exp356_models_pdb_rollout_proofreader · debug-exp356-frozen-contact-v2

**Launched:** 2026-10-09T22:44:02Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [debug-exp356-frozen-contact-v2](https://wandb.ai/open-athena/MarinFold/runs/debug-exp356-frozen-contact-v2)  
**Git:** `ce02fa29`  

## Description

Eight-H100 frozen causal backbone and bidirectional contact encoder smoke with optimizer recovery

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

The eight-H100 smoke paused at step 2 and resumed in a separate job. All six validation metrics reproduced exactly (loss 0.5982204149477184), then training completed step 4. This verifies execution, frozen-backbone behavior and optimizer recovery; it does not establish model quality.
