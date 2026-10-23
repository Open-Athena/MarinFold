---
marinfold_run:
  user: bizon
  launched_at: '2026-10-13T15:15:36Z'
  experiment: exp363_models_double_channel_width
  kind: models
  short_description: '5.862B double-width exp277 variant: one full epoch on 32 H100s'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp363-m2-p06-full-epoch-5.86B
    run_name: contacts-v1-exp363-m2-p06-full-epoch-5.86B
  git_sha: 56f10733f684fa0975692eb42f6a5d252a55bec3
  iris_job_ids:
  - /bizon/exp363-train-a02
  - /bizon/exp363-train-a02/exp363-train-4fae9eab
  - /bizon/exp363-train-a03
  - /bizon/exp363-train-a03/exp363-train-0cdc9fb5
---
# 2026-10-13 · exp363_models_double_channel_width · contacts-v1-exp363-m2-p06-full-epoch-5.86B

**Launched:** 2026-10-13T15:15:36Z by bizon

**Kind:** models

**Experiment:** exp363_models_double_channel_width

**W&B:** [contacts-v1-exp363-m2-p06-full-epoch-5.86B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B)

**Git:** `56f10733`


## Description

5.862B double-width exp277 variant: one full epoch on 32 H100s

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

2026-10-16: reached step 8438 (3.17%), train loss 3.15777 and last eval loss 3.46485. Checkpoint barrier timeout stopped training despite Iris success. Last complete checkpoint: step 8120; step 8279 lacks completion metadata. Restart a03 submitted from commit 241b26c9 with the same run ID and data/optimizer schedule; runtime now requires the final checkpoint before success.

2026-10-19: a03 is running after two preemptions and no recorded failures. Latest observed W&B step 48,451 (18.19%), train loss 2.89288 and last full eval loss 3.12334. Complete rolling checkpoint step 48,452 and permanent step 26,634 verified. Optimizer time 5.612 s/update on 32 H100s. The successful 32-chip v5p smoke measured 7.399 s/update; production remains on GPUs.

## October 23 recovery blocked

Last W&B update: step 75,122 at 2026-10-22 18:02 EDT, LR 0.001. Latest full
validation at step 73,990: 3.0982568264. The a03 child is worker_failed: its
old submission lacks the CoreWeave credentials now required by Iris. The driver
is failed. The current launcher forwards the named CW credentials and the
standard AWS variables needed by TensorStore.

Relaunch is blocked: the local cw profile is rejected by the public CoreWeave
storage endpoint with InvalidAccessKeyId. No replacement job has been submitted.
The latest complete checkpoint cannot be reverified until credentials are
refreshed. Recovery command and evidence: experiment data/recovery_20261023.json.
