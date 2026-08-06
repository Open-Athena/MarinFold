---
marinfold_run:
  user: eczech
  launched_at: '2026-08-06T16:44:44Z'
  experiment: exp199_models_optimize_contacts_v1_afdb_esm
  kind: models
  short_description: Two-step 50/50 AFDB and ESM existing-cache smoke on a us-east5
    v6e-4
  wandb:
    url: https://wandb.ai/eric-czech/marin/runs/prot-exp199-smoke-cache-validation-us-east5-v6e-4
    entity: eric-czech
    project: marin
    run_id: prot-exp199-smoke-cache-validation-us-east5-v6e-4
    run_name: prot-exp199-smoke-cache-validation-us-east5-v6e-4
  git_sha: 74d9ccffb15561786a206b1c6e0584850a55bbdb
  iris_job_ids:
  - /eczech/prot-exp199-smoke-cache-validation-us-east5-v6e-4
  - /eczech/prot-exp199-smoke-cache-validation-us-east5-v6e-4/run_levanter_train_lm-7799dd57
---

# 2026-08-06 · exp199_models_optimize_contacts_v1_afdb_esm · prot-exp199-smoke-cache-validation-us-east5-v6e-4

**Launched:** 2026-08-06T16:44:44Z by eczech
**Kind:** models
**Experiment:** exp199_models_optimize_contacts_v1_afdb_esm
**W&B:** [prot-exp199-smoke-cache-validation-us-east5-v6e-4](https://wandb.ai/eric-czech/marin/runs/prot-exp199-smoke-cache-validation-us-east5-v6e-4)
**Git:** `74d9ccff`

## Description

Two-step 50/50 AFDB and ESM existing-cache smoke on a us-east5 v6e-4

## Detailed plan

Validate that the current packaged Marin/Levanter APIs can train directly from
the existing AFDB and ESM contacts-v1 token caches without copying or
retokenizing them. Run the exp166-style 1.5B configuration for two steps on a
single `v6e-4`, then evaluate the complete `tokenized/contacts-v1-val` cache.

## Changes from previous runs

- Used a 50/50 AFDB/ESM training mixture and the 3,848-token
  `timodonnell/contacts-and-crops-v1-tokenizer` contract.
- Adopted all three caches as path-only regional artifacts and disabled
  Levanter cache auto-building so missing data fails instead of rebuilding.
- Preserved exp166's model, optimizer, packing, block shuffle, and
  training-only amino-acid statement augmentation.

## Notes

Completed successfully with zero failures and zero preemptions. Both training
steps and all 104 validation batches completed; final validation loss was
`23.72143` (`2.95559` BPB). Cache adoption took under a second and no
retokenization or cache copy ran. Native and HF-compatible step-1 checkpoints
were saved beneath
`gs://marin-us-east5/protein-structure/MarinFold/exp199_models_optimize_contacts_v1_afdb_esm/users/exedev/checkpoints/prot-exp199-smoke-cache-validation-us-east5-v6e-4/dev/`.
