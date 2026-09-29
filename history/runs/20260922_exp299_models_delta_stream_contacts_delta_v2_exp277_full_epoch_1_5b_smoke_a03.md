---
marinfold_run:
  user: zack
  launched_at: '2026-09-22T16:57:22Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Exp277-scale delta-stream V2 cache/training smoke (a03)
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a03
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-full-epoch-1_5b-smoke-a03
    run_name: delta-v2-exp277-full-epoch-1_5b-smoke-a03
  git_sha: c69d717086709f0e1db7c1856785c04cadf28770
  iris_job_ids:
  - /zack/exp299-v2-exp277-train-smoke-driver-a03
  - /zack/exp299-v2-exp277-train-smoke-driver-a03/delta-v2-exp277-full-epoch-1_5b-smoke-a03
---

# 2026-09-22 · exp299_models_delta_stream_contacts · delta-v2-exp277-full-epoch-1_5b-smoke-a03

**Launched:** 2026-09-22T16:57:22Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-full-epoch-1_5b-smoke-a03](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a03)  
**Git:** `c69d7170`  

## Description

Exp277-scale delta-stream V2 cache/training smoke (a03)

## Detailed plan

Verify that converted exp277 V2 documents can be packed with document-aware
attention and train exp277's 1.5B architecture end-to-end for ten updates on one
8×H100 node.

## Changes from previous runs

- Uses `PackableTokenIdsFormat` so variable-length pretokenized rows select
  Levanter's `PackedTokenDataset`.
- Preserves segment IDs and blocks cross-document attention.

## Notes

Succeeded. The input smoke cache has 1,000 documents and 4,812,542 tokens.
Training reached step 9 with final loss 4.8388. Excluding compilation, W&B
reported 101,603 tokens/s and 10.32 seconds/update for batch 128 × 8,192 tokens.
The first update took 155 seconds including tracing/compilation.
