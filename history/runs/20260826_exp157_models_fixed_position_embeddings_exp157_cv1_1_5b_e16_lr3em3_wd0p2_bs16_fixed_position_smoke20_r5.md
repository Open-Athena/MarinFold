---
marinfold_run:
  user: zack
  launched_at: '2026-08-26T21:09:34Z'
  experiment: exp157_models_fixed_position_embeddings
  kind: models
  short_description: 20-step CoreWeave contacts-v1 fixed-position input-embedding
    training smoke
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5
    entity: open-athena
    project: MarinFold
    run_id: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5
    run_name: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5
  git_sha: e5d2bb9bd77a2e7a9d9b5d3c4f889dd5db5c0175
  iris_job_ids:
  - /zack/exp157-fixed-position-training-smoke-r5-driver
  - /zack/exp157-fixed-position-training-smoke-r5-driver/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5
---

# 2026-08-26 · exp157_models_fixed_position_embeddings · exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5

**Launched:** 2026-08-26T21:09:34Z by zack  
**Kind:** models  
**Experiment:** exp157_models_fixed_position_embeddings  
**W&B:** [exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5](https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r5)  
**Git:** `e5d2bb9b`  

## Description

20-step CoreWeave contacts-v1 fixed-position input-embedding training smoke

## Detailed plan

20-step regular next-token training smoke on CoreWeave H100, using the contacts-v1 corpus and fixed residue-position input embeddings for `<p0>` ... `<p1999>`.

## Changes from previous runs

Reuses the exp108 contacts-v1 token cache instead of launching a fresh cache build under the exp157 prefix.

## Notes

Started successfully and created W&B run, but the first worker attempt was preempted before completing a train step. As of launch monitoring, the replacement 8xH100 task is waiting in Kueue scheduling gates.
