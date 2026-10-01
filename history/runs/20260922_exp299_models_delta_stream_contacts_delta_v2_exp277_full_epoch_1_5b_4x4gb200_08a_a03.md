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
  - /zack/exp299-v2-exp277-full-epoch-driver-2x4gb200-08a-a04
  - /zack/exp299-v2-exp277-full-epoch-driver-2x4gb200-08a-a04/delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
  - /zack/exp299-v2-exp277-full-epoch-driver-4x4gb200-08a-a05
  - /zack/exp299-v2-exp277-full-epoch-driver-4x4gb200-08a-a05/delta-v2-exp277-full-epoch-1_5b-4x4gb200-08a-a03
  - /zack/exp299-s3-gcs-cv1-native-afdb-a01
  - /zack/exp299-s3-gcs-cv1-native-esm-a01
  - /zack/exp299-s3-gcs-cv1-mpnn-afdb-a01
  - /zack/exp299-s3-gcs-cv1-mpnn-afdb-a02
  - /zack/exp299-s3-gcs-cv1-mpnn-esm-a01
  - /zack/exp299-s3-gcs-cv1-validation-a01
  - /zack/exp299-s3-gcs-v2-cache-a01
  - /zack/exp299-s3-gcs-v2-cache-a02
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

By step 19,960 the 4×4 gang had suffered 12 capacity preemptions and was again
scheduling-gated, with no progress for several hours. It was cancelled and
resubmitted as the planned 2×4 fallback under driver
`/zack/exp299-v2-exp277-full-epoch-driver-2x4gb200-08a-a04`, retaining the same
run ID and output path. The smaller gang placed immediately, restored the full
training state from temporary checkpoint step 19,784, and resumed at about 5.1
seconds/update. The latest pre-fallback validation loss was 1.1501 at step
19,026.

On 2026-09-29, after the 2×4 run reached step 112,345, newly available GB200
capacity justified retrying the 4×4 configuration. The 2×4 driver was stopped
only after its step-112,345 temporary checkpoint committed. The replacement
4×4 driver `/zack/exp299-v2-exp277-full-epoch-driver-4x4gb200-08a-a05` was
initially scheduling-gated, placed after about 35 minutes, restored the complete
state from step 112,345, and resumed at approximately 2.6 seconds/update. It
retains the same W&B run ID and output path.

Canonical 100-rollout evaluation of retained checkpoints 63,147, 84,196, and
126,294 used the same 670 proteins and seeds. Step 126,294 reached `legacy_554`
R/all 0.5742 and R/long 0.5166, `eval-val` 0.4428 and 0.4134, and
`eval-denovo` 0.6283 and 0.5685. Relative to step 84,196, the paired all-670
improvement was +0.01498 R/all (95% CI [0.00885, 0.02123]) and +0.01514 R/long
([0.00785, 0.02262]). All 67,000 generations parsed successfully. Results are
stored beneath `exp277_full_epoch_rollout_votes/results-2026-10-01-step126294/`
in the 08a bucket and summarized in the experiment data CSVs.

On 2026-10-01, both exp277-scale token-cache formats were mirrored to the
TPU-local `marin-us-east5` GCS bucket in preparation for a possible
`us-east5-a` v5p continuation. The contacts-v1 cache contains 32,622 objects
and 351,901,003,059 bytes across the four train corpora plus validation. The
contacts-v2 cache contains 32,146 objects and 159,232,199,956 bytes. GCS
listings exactly matched both source object counts and aggregate byte totals;
each uploaded object was also checked against its source size. Both tokenizers
were staged alongside the caches. The common destination is
`gs://marin-us-east5/protein-structure/MarinFold/exp299_contacts_delta_stream_v2_sequence_prefix/exp277_full_epoch_tokenized_cache/2026.09.22.1/`.
