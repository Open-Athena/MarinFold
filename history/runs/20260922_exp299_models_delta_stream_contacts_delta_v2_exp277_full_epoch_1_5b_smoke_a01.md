---
marinfold_run:
  user: zack
  launched_at: '2026-09-22T16:57:22Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Exp277-scale delta-stream V2 cache/training smoke (a01)
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-full-epoch-1_5b-smoke-a01
    run_name: delta-v2-exp277-full-epoch-1_5b-smoke-a01
  git_sha: c69d717086709f0e1db7c1856785c04cadf28770
  iris_job_ids:
  - /zack/exp299-v2-exp277-train-smoke-driver-a01
  - /zack/exp299-v2-exp277-train-smoke-driver-a01/delta-v2-exp277-full-epoch-1_5b-smoke-a01
---

# 2026-09-22 · exp299_models_delta_stream_contacts · delta-v2-exp277-full-epoch-1_5b-smoke-a01

**Launched:** 2026-09-22T16:57:22Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-full-epoch-1_5b-smoke-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-smoke-a01)  
**Git:** `c69d7170`  

## Description

Exp277-scale delta-stream V2 cache/training smoke (a01)

## Detailed plan

Run ten updates of the exp277 1.5B recipe against a 1,000-document converted
V2 cache smoke in `cw-us-east-02a`.

## Changes from previous runs

- Uses the append-only 4,030-token V2 vocabulary.
- Uses converted exp277 ProteinMPNN-AFDB documents rather than the original
  exp299 corpus.
- Enables blocked cross-document attention.

## Notes

Failed before training because Levanter attempted to open a nonexistent
`validation/shard_ledger.json`. The cache contains only a `train` split; the
next attempt loads it as a flat cache.
