---
marinfold_run:
  user: zack
  launched_at: '2026-08-26T20:51:20Z'
  experiment: exp157_models_fixed_position_embeddings
  kind: models
  short_description: 20-step CoreWeave contacts-v1 fixed-position input-embedding
    training smoke
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4
    entity: open-athena
    project: MarinFold
    run_id: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4
    run_name: exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4
  git_sha: 64f4a78829f6bfe836a41c02928af81a5751d443
  iris_job_ids:
  - /zack/exp157-fixed-position-training-smoke-r4-driver
  - /zack/exp157-fixed-position-training-smoke-r4-driver/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4
---

# 2026-08-26 · exp157_models_fixed_position_embeddings · exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4

**Launched:** 2026-08-26T20:51:20Z by zack  
**Kind:** models  
**Experiment:** exp157_models_fixed_position_embeddings  
**W&B:** [exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4](https://wandb.ai/open-athena/MarinFold/runs/exp157-cv1-1_5b-e16-lr3em3-wd0p2-bs16-fixed-position-smoke20-r4)  
**Git:** `64f4a788`  

## Description

20-step CoreWeave contacts-v1 fixed-position input-embedding training smoke

## Detailed plan

20-step regular next-token training smoke on CoreWeave H100, using the contacts-v1 corpus and fixed residue-position input embeddings for `<p0>` ... `<p1999>`.

## Changes from previous runs

First full Levanter training-path smoke for exp157 after the unit-level CoreWeave pytest smoke.

## Notes

Failed before training because the run tried to build a fresh token cache via Zephyr, and cache-build workers repeatedly hit dependency-index 429s. Superseded by r5, which reuses the already-built exp108 contacts-v1 token cache.
