---
marinfold_run:
  user: jder
  launched_at: '2026-07-25T00:19:32Z'
  experiment: exp147_models_on_the_fly_contacts_v1_pilot
  kind: models
  short_description: Loss-only any-permissible contacts ablation; failed during TPU
    program load before step 1
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8
    entity: open-athena
    project: MarinFold
    run_id: exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8
    run_name: exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8
  git_sha: 1234e26458c7721266c9598c521eeed9292b9664
  iris_job_ids:
  - /jder/iris-run-train-20260725-001931
---

# 2026-07-25 · exp147_models_on_the_fly_contacts_v1_pilot · exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8

**Launched:** 2026-07-25T00:19:32Z by jder  
**Kind:** models  
**Experiment:** exp147_models_on_the_fly_contacts_v1_pilot  
**W&B:** [exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8](https://wandb.ai/open-athena/MarinFold/runs/exp147-otf-contacts-v1-any-permissible-1_5b-steps35680-bs256-v6e8)  
**Git:** `1234e264`  

## Description

Loss-only any-permissible contacts ablation; failed during TPU program load before step 1

## Detailed plan

Run the schedule-matched exp147 configuration as a loss-only ablation:
v6e-8 in `us-east5-b`, 35,680 steps, global batch 256, sequence length
8,192, and evaluation every 1,115 steps. Keep serialized documents, causal
attention, packing, corpus, model, optimizer, and validation unchanged.

## Changes from previous runs

- At each contact endpoint slot, score the incidence-weighted distribution over
  remaining permissible contacts rather than only the serialized next token.

## Notes

- [Iris job](https://iris.oa.dev/#/job/%2Fjder%2Firis-run-train-20260725-001931)
  failed before step 1 with no preemptions.
- TPU program loading requested another 85.85 MiB when only 78.13 MiB was
  free, producing `RESOURCE_EXHAUSTED: RuntimeProgramAllocationFailure`.
- W&B ended at history step 0; there are no training or validation metrics from
  this attempt.
