---
marinfold_run:
  user: bizon
  launched_at: '2026-10-22T21:08:34Z'
  experiment: exp363_models_double_channel_width
  kind: models
  short_description: Full-state LR fork from step 74178 to 0.000707; stable and ahead
    of the default at 19 matched evaluations
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
    run_name: contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4
  git_sha: 9d0d088e18f9cd7393293e9ac490a4f44b6beeae
  iris_job_ids:
  - /bizon/exp363-fork-train-a02
  - /bizon/exp363-fork-train-a02/exp363-train-755515df
  - /bizon/exp363-fork-train-a03
  - /bizon/exp363-fork-train-a03/exp363-train-fb907ed2
  - /bizon/exp363-fork-train-a04
  - /bizon/exp363-fork-train-a04/exp363-train-3fc9d00b
---
# 2026-10-22 · exp363_models_double_channel_width · contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4

**Launched:** 2026-10-22T21:08:34Z by bizon  
**Kind:** models  
**Experiment:** exp363_models_double_channel_width  
**W&B:** [contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp363-m2-p06-full-epoch-5.86B-fork74178-lr7p07e-4)  
**Git:** `9d0d088e`  

## Description

Full-state LR fork from step 74178 to 0.000707; stable and ahead of the default
at 19 matched evaluations as of October 26.

## Detailed plan

Restore complete training state from checkpoint 74,178 and lower the 0.001 LR
linearly to 0.000707 over 2,114 updates. Retain the original final cooldown at
step 213,076 and 266,345-update horizon, alongside the original control run.

## Changes from previous runs

Independent W&B/checkpoint paths; full Adam state, RNG and data cursor retained.
Only the plateau LR changes; final cooldown retains its original timing and
10% floor ratio.

## Notes

October 23 inspection: startup failed before any training metrics. TensorStore's
default AWS credential chain could not source credentials, and the first data
cache metadata read failed with 403 AccessDenied. The child exited zero and
W&B says finished, but the driver's final-checkpoint guard correctly failed.

Runtime now explicitly passes AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from the
CoreWeave credentials. All 13 local tests pass. Recovery awaits refreshed local
credentials: the configured cw key is rejected with InvalidAccessKeyId by the
public CoreWeave endpoint. No retry has been submitted with the rejected key.

## October 23 credentials refreshed and serialization fixed

The refreshed key passed fsspec and TensorStore reads. The preserved source
checkpoint 74,178 still has trainer/Adam/scheduler count 74,179 and LR 0.001.
Attempt a03 passed authentication but could not serialize the optimizer class
loaded through the CLI as __main__.ForkAdamConfig. It was cancelled before
training. The registered class now lives in importable fork_optimizer.py; all
14 tests pass, including subprocess serialization and choice-registry decoding.

Attempt a04 was submitted at 10:41 EDT from commit d923758f with unchanged
optimizer calculations, full-state checkpoint structure, data and schedule.

Recovery verified: first update 74,179 completed at 10:51:32 EDT. At 10:52 EDT,
W&B reports step 74,190, LR 0.0009984754724 and 5.59 seconds/update. The first
nine recorded LRs match the requested schedule within 1.14e-10 absolute error.
The first loss exactly matches the original at step 74,179: 2.90014386177063.
There is no new full-validation comparison yet.

## October 26: stable at target LR

At 11:55 EDT the a04 driver and child remain running. W&B last reported
step 116,027 (43.56%), LR 0.000707, train loss 2.84628 at 11:54:33 EDT.
The fork has lower full-validation loss than exp277 at all 19 exact-step
post-fork evaluations. At the latest shared step 114,156: fork 3.066603,
default 3.071639, original high-LR wide run 17.649031. The advantage over
the default is 0.005036 nats/token, or 0.50% lower perplexity. The original
has diverged; this fork remains stable. Same final cooldown at 213,076 and
266,345-update horizon. This is an interim single-seed language-model result,
with no wider-model contact evaluation yet. Evidence and figure inputs:
experiment data/progress_20261026.json and data/lr_fork_validation_20261026.csv.
