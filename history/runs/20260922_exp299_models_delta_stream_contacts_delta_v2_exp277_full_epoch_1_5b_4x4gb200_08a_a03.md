---
marinfold_run:
  user: zack
  launched_at: '2026-09-22T20:21:26Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: One finite epoch of delta-stream V2 over the full exp277 native
    plus ProteinMPNN corpus
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
    run_name: delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
  git_sha: eea93a5efeadf06a48e370dc4ae5da727f9ec2ae
  iris_job_ids:
  - /zack/exp299-v2-exp277-mirror-cache-08a-a01
  - /zack/exp299-v2-exp277-full-epoch-driver-4x4gb200-08a-a03
  - /zack/exp299-v2-exp277-full-epoch-driver-4x4gb200-08a-a03/delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
---

# 2026-09-22 · exp299_models_delta_stream_contacts · delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03

**Launched:** 2026-09-22T20:21:26Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03)  
**Git:** `eea93a5e`  

## Description

One finite epoch of delta-stream V2 over the full exp277 native plus ProteinMPNN corpus

## Detailed plan

Train the 1.476B-parameter Qwen model from scratch for exactly one finite pass
over all 232,090,905 exp277 native and ProteinMPNN documents in canonical V2
format. The audited corpus contains 26,942,937 packed examples and 220.72B
packed tokens, giving 210,492 updates at global batch 128. Use exp277's WSD
recipe: LR 1e-3, weight decay 0.2, 10% warmup, 70% stable, and 20% linear
decay to 1e-4.

## Changes from previous runs

- Uses the append-only 4,030-token V2 vocabulary and sequence-prefix delta
  stream instead of contacts-v1.
- Uses all four exp277 first-epoch corpora with no sequence-orientation
  augmentation.
- Packs whole documents with segment IDs and blocked cross-document attention.
- Runs on 4×4 GB200 in `cw-us-east-08a`; cache and checkpoints are in the
  co-located `rhoarnet-us-east-08a` bucket.

## Notes

The 159,232,199,956-byte cache was mirrored and byte-count/object-set verified
before launch: 32,146 objects copied from `marin-us-east-02a` to
`rhoarnet-us-east-08a` by `/zack/exp299-v2-exp277-mirror-cache-08a-a01`.

Startup succeeded on all four nodes. Step 0 completed after compilation at
20:18 UTC. Early steady-state median through step 89 was 2.5805 seconds/update
(about 406k tokens/s), projecting roughly 6.3 days for 210,492 updates before
checkpoint/evaluation overhead. The config-artifact YAML logger emitted a
nonfatal warning because the custom packable format was not registered with
Draccus; training and W&B metrics were unaffected, and commit `981f3781`
registers it for future restarts.
