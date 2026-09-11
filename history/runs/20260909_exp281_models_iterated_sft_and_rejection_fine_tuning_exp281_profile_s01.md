---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T20:14:19Z'
  experiment: exp281_models_iterated_sft_and_rejection_fine_tuning
  kind: models
  short_description: Eight-step full-model 8xH100 memory and throughput profile on
    64 validated examples
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp281-profile-s01
    entity: open-athena
    project: MarinFold
    run_id: exp281-profile-s01
    run_name: exp281-profile-s01
  git_sha: 179ab337
  iris_job_ids:
  - /bizon/exp281-profile-s01
  - /bizon/exp281-profile-report
  - /bizon/exp281-profile-report-r2
---
# 2026-09-09 · exp281_models_iterated_sft_and_rejection_fine_tuning · exp281-profile-s01

**Launched:** 2026-09-09T20:14:19Z by bizon  
**Kind:** models  
**Experiment:** exp281_models_iterated_sft_and_rejection_fine_tuning  
**W&B:** [exp281-profile-s01](https://wandb.ai/open-athena/MarinFold/runs/exp281-profile-s01)  
**Git:** `179ab337`  

## Description

Eight-step full-model 8xH100 memory and throughput profile on 64 validated examples

## Detailed plan

Profile the actual exp232-derived 1.5B checkpoint with the new vocabulary on one
8×H100 node in `cw-us-east-02a`. Eight optimizer steps, global batch 8, context
8,192; 64 validated bootstrap examples with hypothesis weight 1 and no rehearsal.
The profile checkpoint is not used to initialize the format trial.

## Changes from previous runs

Uses the full model and eight GPUs, extending the earlier tiny-Qwen smoke checks.
Records per-step time and maximum allocated GPU memory across all ranks.

## Notes

All eight steps and checkpoint publication succeeded. Peak allocated memory was
35.44 GB/GPU; median steady step time was 0.450 seconds and steady throughput was
58,480 tokens/second across the node. Worker duration including setup, validation,
and saving was 182.9 seconds. Detailed data are in exp281's
`data/profile_s01.csv` and `data/profile_s01.json`.

Checkpoint:
`s3://marin-us-east-02a/protein-structure/MarinFold/exp281/profile-s01/checkpoints/exp281-profile-s01/step-8/`.

The separate report job initially failed because an empty history directory was
not preserved in the worker bundle. Its retry succeeded after explicitly creating
the directory. Model training itself succeeded on the first attempt.
