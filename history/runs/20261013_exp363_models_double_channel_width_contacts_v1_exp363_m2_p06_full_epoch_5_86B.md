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
  - /bizon/exp363-train-a04
  - /bizon/exp363-train-a04/exp363-train-6d5c8a62
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

## October 23 credentials refreshed

At 10:33 EDT the refreshed key passed fsspec and TensorStore data reads.
The latest complete checkpoint is step 75,031; direct scalar reads verified
trainer, Adam and scheduler count 75,032 and LR 0.001. Driver a04 was submitted
at 10:34 EDT from commit 563579f2. Its child confirmed resumption from 75,032
at 10:42 EDT and entered first-step compilation. The original schedule is unchanged.

Recovery verified: first resumed update 75,032 completed at 10:43:58 EDT;
fresh training losses continued afterward. W&B temporarily retained the
pre-restart high-water mark while earlier steps were replayed.

At 10:54 EDT W&B also confirmed fresh progress beyond the old high-water mark:
step 75,126, train loss 2.8462910652, LR 0.001.

## October 26: running but diverged

At 11:55 EDT the a04 driver and child remain running. W&B last reported
step 117,186 (44.00%), LR 0.001, train loss 16.43802 at 11:51:56 EDT.
Training loss first exceeds 4 at step 77,229 in the inspected 76,000–78,299
window. Validation rises from 3.09974 at 76,104 to 11.88947 at 78,218;
it peaks at 42.43565 at 80,332 and remains 17.96501 at 116,270.
The lower-LR fork is stable. At shared step 114,156 the original/fork/default
losses are 17.649031 / 3.066603 / 3.071639. Evidence is in experiment
data/progress_20261026.json and data/original_loss_onset_20261026.csv.
The original remains active per the user's instruction; stopping this
divergent control is recommended. No schedule change or cancellation made.
