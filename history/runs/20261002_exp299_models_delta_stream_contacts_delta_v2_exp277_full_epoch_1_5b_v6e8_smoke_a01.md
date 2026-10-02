---
marinfold_run:
  user: zack
  launched_at: '2026-10-02T15:55:54Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: TPU compatibility smoke for exp277-scale delta-stream V2 on v6e-8
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01
    run_name: delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01
  git_sha: 51660aac0476376940bee5f93000a68f8a201386
  iris_job_ids:
  - /zack/delta-v2-exp277-full-epoch-1_5b-v5p128-smoke-a01
  - /zack/delta-v2-exp277-full-epoch-1_5b-v5p8-smoke-a01
  - /zack/delta-v2-exp277-full-epoch-1_5b-v5p8-smoke-a02
  - /zack/delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01
---

# 2026-10-02 · exp299_models_delta_stream_contacts · delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01

**Launched:** 2026-10-02T15:55:54Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01)  
**Git:** `51660aac`  

## Description

TPU compatibility smoke for exp277-scale delta-stream V2 on v6e-8

## Detailed plan

Verify that the exp277-scale variable-length delta-stream cache, whole-document
online packing, blocked cross-document attention, 1.476B Qwen model, Splash
attention, optimizer state, and native checkpoint writer all work on a TPU
against the region-local `marin-us-east5` mirror. Run ten updates from scratch
at global batch 128 and sequence length 8,192.

## Changes from previous runs

- Runs on one preemptible `v6e-8` in `us-east5-b` rather than CoreWeave GB200s.
- Reads the GCS mirror of the V2 `mpnn-afdb` cache and writes checkpoints to the
  co-located `marin-us-east5` bucket.
- Uses Splash attention and a per-chip microbatch of 16 across eight TPU chips.
- Uses JAX/JAXlib 0.11.1 and libtpu 0.0.46.1 from the Marin TPU extra.

## Notes

The run completed all ten updates without OOMs or malformed data. The first
step, including compilation, took 36.6 seconds. At step 9, W&B reported train
loss 4.5486, 75,775 tokens/s, 13.84 seconds/update, and 14.13% MFU. A complete
16.50 GiB native checkpoint committed at
`gs://marin-us-east5/protein-structure/MarinFold/exp299_contacts_delta_stream_v2_sequence_prefix/checkpoints/delta-v2-exp277-full-epoch-1_5b-v6e8-smoke-a01/checkpoints/step-9`.

The preceding `v5p-128` request never triggered a scale-up. Two `v5p-8`
attempts were also cancelled before execution: the first requested more disk
than TPU workers expose as allocatable, and the corrected request remained
queued while `us-east5-a` provisioning repeatedly failed. An already-ready,
region-local `v6e-8` placed immediately. The cache metadata warning is the same
nonfatal preprocessor-metadata difference seen with the converted cache; all
batches loaded and trained successfully.
