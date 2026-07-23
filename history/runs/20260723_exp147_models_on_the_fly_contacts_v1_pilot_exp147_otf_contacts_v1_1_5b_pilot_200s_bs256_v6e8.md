---
marinfold_run:
  user: jder
  launched_at: '2026-07-23T20:00:16Z'
  experiment: exp147_models_on_the_fly_contacts_v1_pilot
  kind: models
  short_description: 200-step on-the-fly contacts-v1 TPU pilot on 16 staged shards
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8
    entity: open-athena
    project: MarinFold
    run_id: exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8
    run_name: exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8
  git_sha: b317807
  iris_job_ids:
  - /jder/iris-run-train-20260723-193827
  - /jder/iris-run-train-20260723-194543
  - /jder/iris-run-train-20260723-195327
---

# 2026-07-23 · exp147_models_on_the_fly_contacts_v1_pilot · exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8

**Launched:** 2026-07-23T20:00:16Z by jder  
**Kind:** models  
**Experiment:** exp147_models_on_the_fly_contacts_v1_pilot  
**W&B:** [exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8](https://wandb.ai/open-athena/MarinFold/runs/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8)  
**Git:** `b317807`  

## Description

200-step on-the-fly contacts-v1 TPU pilot on 16 staged shards

## Detailed plan

- Train the exp117 Qwen3 1.47B recipe for 200 steps on a v6e-8 in
  `us-east5-b`, with sequence length 8192 and global batch size 256.
- Read the 16 staged ESM Atlas contact shards, reconstruct deterministic
  contacts-v1 documents on demand, and pack each complete shard to a fixed
  quota of 2,650 examples.
- Evaluate the ordinary tokenized contacts-v1 validation set at steps 100 and
  200 so its loss is on the same metric surface as exp117.
- Confirm that the stateless loader sustains training without foreground input
  stalls before choosing the epoch size and scale of a longer run.

## Changes from previous runs

- Documents are reconstructed and packed from premade contact rows rather than
  read from a materialized token corpus.
- The model architecture and optimizer match the best finished exp117 1.47B
  configuration, but this pilot has only a 200-step budget and is not intended
  as a training-budget comparison.
- Checkpoint and data I/O are co-located with the TPU in `us-east5`.

## Notes

- The first coordinator attempt failed before training because v6e capacity was
  unavailable in `us-east5-a`; the retry moved to `us-east5-b` in the same
  region.
- The second attempt acquired the TPU but failed before W&B initialization
  because the tokenizer client did not parse the pinned `repo@revision` form.
  The third attempt stages that exact tokenizer revision locally on the worker.
- The successful run is pinned to git SHA `b317807`. The subsequent
  no-rendering loader optimization in `7260479` is not part of this run, so its
  throughput is the pre-optimization baseline.
- Iris completed successfully with zero failures or preemptions. W&B uploaded
  its final summary but timed out during `finish()`, leaving the run state shown
  as running; the uploaded final summary was restored through the W&B API.

## Results

- Final train loss: `4.541664`.
- `eval/tokenized/contacts-v1-val/loss`: `5.510117` at step 100 and `4.693140`
  at final step 199.
- Steps 3–198: median 27.792 seconds/step, 9.211 examples/second, 75,460
  tokens/second, 14.057% MFU, and 1.1 ms foreground loading time.
- The noisy 8,192-example background-construction warnings did not translate
  into sustained foreground starvation.

## Artifacts

- Levanter checkpoint:
  `gs://marin-us-east5/protein-structure/MarinFold/exp147_on_the_fly_contacts_v1_pilot/users/jder/checkpoints/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8/exp147-dev/checkpoints/step-199`
- HF export with tokenizer:
  `gs://marin-us-east5/protein-structure/MarinFold/exp147_on_the_fly_contacts_v1_pilot/users/jder/checkpoints/exp147-otf-contacts-v1-1_5b-pilot-200s-bs256-v6e8/exp147-dev/hf/step-199`
