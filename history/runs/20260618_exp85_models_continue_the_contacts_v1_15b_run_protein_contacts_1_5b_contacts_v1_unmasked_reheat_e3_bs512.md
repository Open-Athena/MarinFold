---
marinfold_run:
  user: tim
  launched_at: '2026-06-18T15:07:12Z'
  experiment: exp85_models_continue_the_contacts_v1_15b_run
  kind: models
  short_description: 'Warm-restart #67 contacts-v1 1.5B from step-11999 with v5p-32
    batch512 LR reheat'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512-5fc77c
    entity: open-athena
    project: MarinFold
    run_id: protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512-5fc77c
    run_name: protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512
  git_sha: 98f57ea35f4adaf9ec4e52ace562006f6e4199c0
  iris_job_ids:
  - /tim/iris-run-job-20260618-150348
  - /tim/iris-run-job-20260618-150348/checkpoints-protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512_726d1794-19515e99
  - /tim/iris-run-job-20260618-151053
  - /tim/iris-run-job-20260618-151053/checkpoints-protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512_726d1794-c8cac57f
---
# 2026-06-18 · exp85_models_continue_the_contacts_v1_15b_run · protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512

**Launched:** 2026-06-18T15:07:12Z by tim  
**Kind:** models  
**Experiment:** exp85_models_continue_the_contacts_v1_15b_run  
**W&B:** [protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512](https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-contacts-v1-unmasked-reheat-e3-bs512-5fc77c)  
**Git:** `98f57ea3`  

## Description

Warm-restart #67 contacts-v1 1.5B from step-11999 with v5p-32 batch512 LR reheat

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

- `/tim/iris-run-job-20260618-150348` confirmed the original cache-ledger
  `input_ids/0` failure was fixed, then failed in validation because
  `PrebuiltLmDatasetFormat` bypassed packed-example construction and produced
  variable-length raw-document examples.
- `/tim/iris-run-job-20260618-151053` relaunched from the dirty-tree refined
  `ArrayExemplarTextLmDatasetFormat` fix. That change keeps the
  `TextLmDatasetFormat` / `PackedTokenDataset` path while making the cache
  exemplar derive the existing ledger field `input_ids`. The code was committed
  immediately after launch for reproducibility.
