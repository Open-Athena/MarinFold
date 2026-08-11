---
marinfold_run:
  user: eczech
  launched_at: '2026-08-06T14:17:50Z'
  experiment: exp199_models_optimize_contacts_v1_afdb_esm
  kind: models
  short_description: Ten-step package-only DCLM nano smoke on a europe-west4 v6e-4
  wandb:
    url: https://wandb.ai/eric-czech/marin/runs/exp199-dclm-nano-v6e4-smoke-streaming
    entity: eric-czech
    project: marin
    run_id: exp199-dclm-nano-v6e4-smoke-streaming
    run_name: exp199-dclm-nano-v6e4-smoke-streaming
  git_sha: 2a797d7
  iris_job_ids:
  - /eczech/exp199-dclm-nano-v6e4-smoke-streaming
  - /eczech/exp199-dclm-nano-v6e4-smoke-streaming/run_levanter_train_lm-60342421
---

# 2026-08-06 · exp199_models_optimize_contacts_v1_afdb_esm · exp199-dclm-nano-v6e4-smoke-streaming

**Launched:** 2026-08-06T14:17:50Z by eczech  
**Kind:** models  
**Experiment:** exp199_models_optimize_contacts_v1_afdb_esm  
**W&B:** [exp199-dclm-nano-v6e4-smoke-streaming](https://wandb.ai/eric-czech/marin/runs/exp199-dclm-nano-v6e4-smoke-streaming)  
**Git:** `2a797d7`  

## Description

Ten-step package-only DCLM nano smoke on a europe-west4 v6e-4

## Detailed plan

Validate MarinFold's published-package Marin path with a ten-step nano Llama run on one `v6e-4` in `europe-west4`. Consume the completed regional DCLM Llama-3 token cache through `ArtifactStep.adopt`, force cache auto-build off, and stream its train split without constructing any tokenization step.

## Changes from previous runs

- Read `WANDB_ENTITY` and `WANDB_PROJECT` from `/home/exedev/marin.env`.
- Used a continuous token stream instead of eager document packing; the latter attempted to index all 2.9 billion cache rows before step 0.
- Used a fresh run ID and checkpoint output path after the two preflight attempts.

## Notes

Succeeded with no failures or preemptions. The first batch loaded in 3.6 seconds, all ten steps completed through global step 9, final loss was 11.9435, and the run processed 163,840 tokens at 579,847 tokens/s. Iris and W&B both confirmed one host with four TPU v6 lite devices. Step-9 Levanter and HF-compatible checkpoints were written to the experiment's `marin-eu-west4` prefix.
