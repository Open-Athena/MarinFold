---
marinfold_run:
  user: bizon
  launched_at: '2026-10-16T17:05:41Z'
  experiment: exp363_models_double_channel_width
  kind: models
  short_description: '5.862B TPU hardware trial: 20 native-only updates on 32 v5p
    chips at Interactive priority'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03
    run_name: contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03
  git_sha: 2bc4cf065a95da2bd4244afe04bb050216238328
  iris_job_ids:
  - /bizon/exp363-tpu-smoke-v5p-64-a03
  - /bizon/exp363-tpu-smoke-v5p-64-a03/exp363-train-3f5081a0
---

# 2026-10-16 · exp363_models_double_channel_width · contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03

**Launched:** 2026-10-16T17:05:41Z by bizon
**Kind:** models
**Experiment:** exp363_models_double_channel_width
**W&B:** [contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03)
**Git:** `2bc4cf06`

## Description

5.862B TPU hardware trial: 20 native-only updates on 32 v5p chips at Interactive priority

## Detailed plan

Validate the exact 5,861,984,256-parameter model on v5p-64 (32 chips, eight VMs), global batch 128, sequence length 8192 and per-device microbatch one. Run 20 fresh-initialization updates on existing regional native AFDB/ESM caches with Splash attention. Both CPU driver and TPU child request Interactive priority; all I/O remains in us-east5.

## Changes from previous runs

Replaces the queued Batch-priority v5p trial. This is a hardware smoke with native-only data and a compressed schedule, not continuation of the full native/MPNN production epoch.

## Notes

Succeeded October 16, 17:18 UTC, after starting at 17:04 UTC. All eight workers completed without failures or preemptions. Final step 19: train loss 6.64660, two-batch validation loss 6.56125. Median update time over steps 5–19 is 7.399 s. Native step-19 completion metadata and the 23.448 GB HF export (including tokenizer) were verified October 19. Output: gs://marin-us-east5/protein-structure/MarinFold/exp363_models_double_channel_width/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-tpu-smoke-v5p-64-a03/. Full evidence is in the experiment data/progress_20261019.json and data/checkpoint_verification_20261019.json. History entry backfilled on first observed W&B completion, October 19.
