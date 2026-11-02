---
marinfold_run:
  user: bizon
  launched_at: '2026-09-10T13:42:41Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: Ten-step GPU validation of finite full-corpus epoch data path
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke
    run_name: contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke
  git_sha: 9783b3848045c54a725daf31e59a14522b6e78d5
  iris_job_ids:
  - /bizon/exp277-train-smoke-a02
  - /bizon/exp277-train-smoke-a02/exp277-train-61f27a36
---

# 2026-09-10 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke

**Launched:** 2026-09-10T13:42:41Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B-smoke)  
**Git:** `9783b384`  

## Description

Ten-step GPU validation of finite full-corpus epoch data path

## Detailed plan

Validate the finite full-corpus data path on 16 nodes / 128 H100s at batch priority before the one-epoch production replacement. Ten updates and two validation batches, using all four complete caches and the same 34,092,146-example coverage check.

## Changes from previous runs

Uses a finite concatenation of all native and redesigned packed examples, shuffled once. Removes the initial pilot’s 50:50 native/redesigned mixture. The production target is 266,345 steps; this smoke stops after ten updates and uses its own output identity.

## Notes

Submitted at 13:41:18 UTC on 2026-09-10. All 16 GPU workers started at batch priority and W&B initialized. Source commit 9783b384. Startup validation succeeded: ten updates, final train loss 7.77266, two-batch validation loss 7.03359, and step duration 0.85051 seconds. W&B finished and both Iris driver and GPU child succeeded. Native step-9 checkpoint has 101 objects / 17,657,161,218 bytes; HF step-9 export has 6 objects / 5,885,614,887 bytes, including tokenizer.json and tokenizer_config.json. The full packed-example count check passed before the first update.
