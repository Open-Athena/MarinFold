---
marinfold_run:
  user: bizon
  launched_at: '2026-06-10T12:54:46Z'
  experiment: exp67_models_contacts_v1_1_5b
  kind: models
  short_description: 1.5B Llama on contacts-v1, unmasked next-token loss, shuffled,
    12k steps (~2.7 epochs), LR 3.5e-4, v5p-8 @ us-east5-a
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2
    entity: open-athena
    project: MarinFold
    run_id: protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2
    run_name: protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked
  git_sha: 5f925614076f9f3594a314182267b31841ce6064
  iris_job_ids:
  - /bizon/iris-run-job-20260610-124627
---

# 2026-06-10 · exp67_models_contacts_v1_1_5b · protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked

**Launched:** 2026-06-10T12:54:46Z by bizon  
**Kind:** models  
**Experiment:** exp67_models_contacts_v1_1_5b  
**W&B:** [protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked](https://wandb.ai/open-athena/MarinFold/runs/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2)  
**Git:** `5f925614`  

## Description

1.5B Llama on contacts-v1, unmasked next-token loss, shuffled, 12k steps (~2.7 epochs), LR 3.5e-4, v5p-8 @ us-east5-a

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

- First MarinFold run to do a *fresh* tokenize→train on `marin-latest`
  (`0.99.dev20260529`); surfaced 7 distinct infra issues, all documented in the
  experiment README "Implementation notes".
- **Temporary shim:** the trainer can't read its own token cache on this marin
  build (bug #6008, fixed upstream by #6014 but not yet in a published wheel).
  Worked around with `experiments/exp67_models_contacts_v1_1_5b/sitecustomize.py`
  (patches `BatchTokenizer.output_exemplar` to return numpy leaves), injected via
  `PYTHONPATH=/app` in the pod env. **Remove once on a marin build with #6014.**
- Launched from branch `exp/67-contacts-v1-1_5b` (PR #70); also carries a
  repo-wide fix to `models/marinfold_models/defaults.py` (marin import drift).
- Healthy start: loss 8.29→6.21 over the first ~9 steps, MFU ~40%.
