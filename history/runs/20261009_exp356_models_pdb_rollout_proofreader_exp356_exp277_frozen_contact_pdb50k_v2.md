---
marinfold_run:
  user: bizon
  launched_at: '2026-10-09T22:49:56Z'
  experiment: exp356_models_pdb_rollout_proofreader
  kind: models
  short_description: Frozen exp277 causal features with a bidirectional contact encoder
    on the 50000-structure PDB corpus
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-frozen-contact-pdb50k-v2
    entity: open-athena
    project: MarinFold
    run_id: exp356-exp277-frozen-contact-pdb50k-v2
    run_name: exp356-exp277-frozen-contact-pdb50k-v2
  git_sha: 8bc77612ff769edf54254b95efa459062bf7a1fb
  iris_job_ids:
  - /bizon/exp356-train-frozen-prod-a01
---

# 2026-10-09 · exp356_models_pdb_rollout_proofreader · exp356-exp277-frozen-contact-pdb50k-v2

**Launched:** 2026-10-09T22:49:56Z by bizon  
**Kind:** models  
**Experiment:** exp356_models_pdb_rollout_proofreader  
**W&B:** [exp356-exp277-frozen-contact-pdb50k-v2](https://wandb.ai/open-athena/MarinFold/runs/exp356-exp277-frozen-contact-pdb50k-v2)  
**Git:** `8bc77612`  

## Description

Frozen exp277 causal features with a bidirectional contact encoder on the 50000-structure PDB corpus

## Detailed plan

Compare against the fully bidirectional backbone on the identical frozen corpus and prefix sampler. Keep exp277 causal features frozen and train a four-layer 512-wide bidirectional encoder over contact triples (22,181,378 trainable parameters; 1,487,731,202 total). Global batch 64 on eight H100s. The three-epoch learning-rate schedule has 18,714 steps, with an intentional pause at step 2,000 for full validation before continuation.

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Started 2026-10-09T22:48:13Z. The prerequisite eight-H100 smoke paused at step 2 and resumed in a separate job, reproducing all validation metrics exactly. Early training is about 0.27 seconds per optimizer step. Quality selection uses validation only; the new PDB test split remains reserved for the selected model.
