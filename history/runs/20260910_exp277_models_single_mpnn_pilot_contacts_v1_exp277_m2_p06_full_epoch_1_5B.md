---
marinfold_run:
  user: bizon
  launched_at: '2026-09-10T13:56:19Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: One finite epoch over all 232M native and MPNN documents, 1.5B
    model
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-full-epoch-1.5B
    run_name: contacts-v1-exp277-m2-p06-full-epoch-1.5B
  git_sha: 9783b3848045c54a725daf31e59a14522b6e78d5
  iris_job_ids:
  - /bizon/exp277-train-a02
  - /bizon/exp277-train-a02/exp277-train-8c393c8d
  - /bizon/exp277-train-a03
  - /bizon/exp277-train-a03/exp277-train-78333950
---
# 2026-09-10 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-full-epoch-1.5B

**Launched:** 2026-09-10T13:56:19Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-full-epoch-1.5B](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B)  
**Git:** `9783b384`  

## Description

One finite epoch over all 232M native and MPNN documents, 1.5B model

## Detailed plan

Train a scratch Qwen3 1.5B for one epoch over the complete native and ProteinMPNN-redesigned AFDB/ESM corpus: 232,090,905 documents, 248,583,762,834 raw tokens, 34,092,146 packed examples. Exactly 266,345 updates at batch 128 / sequence 8192; final batch has 114 real examples and 14 zero-padding examples. Finite concatenation followed by block shuffling ensures every packed example is visited once. Retain m2-p06 LR 0.001, weight decay 0.2, WSD warmup 10% / decay 20% / minimum LR ratio 0.1, blocked attention, and scheduled amino-acid order augmentation.

## Changes from previous runs

Replaces the original 50:50 weighted-mixture pilot at the user’s request. The original run was intentionally stopped near step 69,556; this run starts from scratch with a separate W&B/checkpoint identity. Sampling proportions now follow the complete corpus rather than preset source weights. The exact packing audit retains every document; the existing 8192-token limit trims one trailing token from each of 862 documents.

## Notes

Submitted 2026-09-10 at 13:54:56 UTC on cw-us-east-02a, 16 nodes / 128 H100s at batch priority. Source commit 9783b384. Four local configuration/coverage tests passed and the full-corpus ten-step GPU smoke succeeded, including native/HF checkpoint export with tokenizer files. All production workers are running and W&B initialized. Startup verified at 14:06:54 UTC: step 65 / 266,345, finite train loss 5.90443, step duration 0.85815 seconds, throughput 1.222M tokens/s. The full packed-example cardinality check passed; the log confirms scratch initialization. Packing and distributed startup took about nine minutes. Continuous monitoring remains active.

Checkpoint base: `s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B/checkpoints/step-<N>/`. Temporary recovery saves every 15 minutes use the standard same-region 14-day checkpoint mirror. Permanent saves every 26,634 steps and at completion; final expected native/HF checkpoint step is 266,344. HF exports use the sibling `hf/step-<N>/` directory. No restart so far.

2026-09-10 14:19 UTC: step 863 / 266,345, train loss 3.95884, step duration 0.87900 seconds. First recovery checkpoint completed at step 366 at 14:11:39 UTC under the same-region temporary checkpoint mirror. No restart.

First full LM validation completed on 2026-09-10 at 14:38:19 UTC: step 2,114, loss 3.90597725. Training resumed and reached step 2,300 by 14:41 UTC, with train loss 3.72747 and 0.86330 seconds/step. Recovery checkpoint rotation passed at step 1,355. No restart.

2026-09-10 21:03 UTC: step 27,003 / 266,345 (10.14%), train loss 2.98891, step duration 0.86685 seconds. All twelve full LM validation passes improved, reaching 3.18098283 at step 25,368. Warmup is complete. First permanent checkpoint `step-26634` committed at 20:58:21 UTC under the checkpoint base above; temporary recovery checkpoint `step-26591` also saved. No failures or restarts. Validation history is recorded in `data/epoch_validation_progress.csv`.

2026-09-11 03:54 UTC: step 53,571 / 266,345 (20.11%), train loss 2.86532, step duration 0.86567 seconds. All 25 full LM validation passes improved, reaching 3.10472560 at step 52,850. Permanent checkpoint `step-53268` committed at 03:49:34 UTC; temporary recovery checkpoint `step-52927` also saved. No failures or restarts. Brief data-loading delays cleared without intervention. Validation history is recorded in `data/epoch_validation_progress.csv`.

2026-09-11 09:10 UTC: first production failure after step 73,702 (27.67%). JAX distributed shutdown barrier timed out with worker 6 first at the barrier; the underlying initiating error was not recovered before pod cleanup. All workers exited 139. Checkpoint step-73703 has a manifest but no completion metadata; step-72744 has both and committed at 08:50:11 UTC. The driver reached FAILED. Recovery attempt /bizon/exp277-train-a03 was submitted at 09:17:36 UTC with the unchanged training recipe, same W&B/checkpoint identity, and 128 H100s at batch priority. Source commit 5ae12a8f (documentation changes only since original launch). All 16 replacement workers started; restore verification pending. Validation loss improved across all 34 evaluations to 3.08621430 at step 71,876.

Recovery verified at 09:28 UTC: latest-checkpoint discovery skipped incomplete step-73703 and restored step-72744. The trainer resumed from step 72,745 at 09:27:21 UTC, completed its first update at 09:27:35 UTC, and advanced to about 72,800 with finite loss 2.88 and 1.1 updates/second. W&B suppresses replayed metrics until its existing step-73703 high-water mark; logs are the progress source during catch-up. One recovery so far.

2026-09-11 09:34:44 UTC: first post-restart checkpoint step-73199 completed its distributed commit and rotated out step-72744. Training reached step 73,246 by 09:35:27 UTC. Both training and checkpointing have recovered.

2026-09-11 09:46 UTC: W&B caught up and reports fresh step 73,958 / 266,345 (27.77%), train loss 2.87417 and 0.87684 seconds/update. Iris reports RUNNING without errors. The resumed run has passed its pre-failure high-water mark.
