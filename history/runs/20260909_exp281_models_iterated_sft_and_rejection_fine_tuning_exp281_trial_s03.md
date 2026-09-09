---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T20:24:03Z'
  experiment: exp281_models_iterated_sft_and_rejection_fine_tuning
  kind: models
  short_description: 256-step multi-contact format SFT trial on 2023 training and
    25 held-out proteins
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp281-trial-s03
    entity: open-athena
    project: MarinFold
    run_id: exp281-trial-s03
    run_name: exp281-trial-s03
  git_sha: bab3f50a
  iris_job_ids:
  - /bizon/exp281-trial-s03
  - /bizon/exp281-trial-s03-r1
  - /bizon/exp281-trial-s03-eval-natural
  - /bizon/exp281-trial-s03-eval-forced
  - /bizon/exp281-trial-s03-score-natural
  - /bizon/exp281-trial-s03-score-forced
  - /bizon/exp281-trial-s03-report
---
# 2026-09-09 · exp281_models_iterated_sft_and_rejection_fine_tuning · exp281-trial-s03

**Launched:** 2026-09-09T20:24:03Z by bizon\
**Kind:** models\
**Experiment:** exp281_models_iterated_sft_and_rejection_fine_tuning\
**W&B:** [exp281-trial-s03](https://wandb.ai/open-athena/MarinFold/runs/exp281-trial-s03)\
**Git:** `bab3f50a`\

## Description

256-step multi-contact format SFT trial on 2023 training and 25 held-out proteins

## Detailed plan

Test full-model format acquisition before synthesis-weighted SFT. Current exp232
m2-p06 step 363000, extended to vocabulary 2847; 16 bootstrap drafts, hypothesis
weight 1, 50% plain rehearsal, global batch 32, LR 1e-4 with 25-step warm-up.
One 8×H100 node on cw-us-east-02a; large I/O stays in its S3 bucket.

The fixed step-256 format gate uses 25 held-out proteins × 8 completions per mode:
99% validity separately for natural/forced and 90% natural multiple nonempty
hypotheses. Natural seed 281; forced seed 282, budgets 0/256/1024/2048.

## Changes from previous runs

First complete format-learning trial after the full-model eight-step profile.
The profile checkpoint does not initialize this run. Uses all 2,048 balanced
targets (2,023 train / 25 validation); accepts and audits empty contact sets.

## Notes

Completed all 256 steps; format gate failed: natural 0/200 valid, forced 14/200
valid. Multiple raw sections appear in 188/200 natural outputs, but the natural
final marker appears in only 8/200. Forced samples restart hypotheses after the
final marker in 185/200 cases. No synthesis/rejection round was launched.

Validation loss: 2.7812 at step 32 → 2.2445 at 256. Final minibatch loss 1.8686.
Peak allocated memory 35.45 GB/GPU; median optimizer-step time 1.049 s.

The first attempt stalled publishing step-128 optimizer state, then a waiting
rank aborted. Resume from complete step 64, unchanged source/configuration,
successfully saved 128/192/256. The abandoned optimizer multipart was aborted.
Checkpoint retry hardening remains necessary before long runs.

W&B explicit step numbers suppress replayed metrics after rollback. The committed
trial_s03_training.csv replaces replayed steps with the successful resumed
console's values; the raw W&B history is retained separately. Eight-GPU replay
was not numerically identical to the lost attempt.

Final checkpoint with tokenizer:
`s3://marin-us-east-02a/protein-structure/MarinFold/exp281/trial-s03/checkpoints/exp281-trial-s03/step-256`.

The experiment README records the full results and public report checksum.
Prediction timings and all 400 candidate diagnostics are committed alongside it.
