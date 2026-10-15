---
marinfold_run:
  user: bizon
  launched_at: '2026-10-13T15:04:10Z'
  experiment: exp362_models_smaller_model_scaling
  kind: models
  short_description: 255M depth-only full-corpus scaling trial; 266345 updates
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth4-full-epoch
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp362-depth4-full-epoch
    run_name: contacts-v1-exp362-depth4-full-epoch
  git_sha: 6e2df4371275d11287aa9974fe72ae7d2aff0753
  iris_job_ids:
  - /bizon/exp362-depth4-train-a01
  - /bizon/exp362-depth4-train-a02
  - /bizon/exp362-depth4-train-a02/exp362-train-275539ff
  - /bizon/exp362-depth4-train-a01/exp362-train-0a9a5492
---
# 2026-10-13 · exp362_models_smaller_model_scaling · contacts-v1-exp362-depth4-full-epoch

**Launched:** 2026-10-13T15:04:10Z by bizon  
**Kind:** models  
**Experiment:** exp362_models_smaller_model_scaling  
**W&B:** [contacts-v1-exp362-depth4-full-epoch](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp362-depth4-full-epoch)  
**Git:** `6e2df437`  

## Description

255M depth-only full-corpus scaling trial; 266345 updates

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

At 2026-10-15 14:37 UTC, W&B had reached 173,896 updates (65.3%) but last reported at 02:05 UTC. The original child `/bizon/exp362-depth4-train-a01/exp362-train-0a9a5492` reported three preemptions, zero failures, and all four workers waiting for Kueue admission. Native checkpoint step 159804 has completed metadata, both manifests, and 20 OCDBT data objects; inspection did not load the trainer state.

After approximately 12.5 hours without progress, the old root and child were cancelled and verified killed. Attempt 2 was submitted at 2026-10-15T14:40:00.233Z on one node/eight H100s at batch priority, using source `14b2a33243bcd1eea576444b61747d91ab45a7e3`. The larger depth8 smoke's per-device batch eight supports this placement's memory feasibility. Global batch 128, the complete 266,345-update recipe, W&B identity, and checkpoint root are unchanged. Approximately 14,091 updates beyond step 159804 may repeat. Child `/bizon/exp362-depth4-train-a02/exp362-train-275539ff` is submitted; admission and trainer restoration await verification.

At 14:43 UTC the replacement had initialized JAX and resumed the W&B identity, but Iris reported one startup preemption, zero failures, and its single worker waiting at the admission gate again. No new optimizer updates or successful trainer restoration were verified. Its startup found temporary checkpoint step 172859; metadata, manifests, and 23 data objects are reachable under `s3://marin-us-east-02a/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/MarinFold/exp362_models_smaller_model_scaling/runs/contacts-v1-exp362-depth4-full-epoch/checkpoints/step-172859`. Successful restoration there would reduce repeated updates to approximately 1,036.
