---
marinfold_run:
  user: bizon
  launched_at: '2026-09-14T13:45:00Z'
  experiment: exp277_models_single_mpnn_pilot
  kind: models
  short_description: 'Failed restore smoke: loader probed data index 0 before the
    restored offset'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke
    entity: open-athena
    project: MarinFold
    run_id: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke
    run_name: contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke
  git_sha: 07a8bccc617a6dda64c8161ac02d5b5b6158c019
  iris_job_ids:
  - /bizon/exp277-continue-smoke-a01
  - /bizon/exp277-continue-smoke-a01/exp277-train-2ff47238
---

# 2026-09-14 · exp277_models_single_mpnn_pilot · contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke

**Launched:** 2026-09-14T13:45:00Z by bizon  
**Kind:** models  
**Experiment:** exp277_models_single_mpnn_pilot  
**W&B:** [contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke](https://wandb.ai/open-athena/MarinFold/runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke)  
**Git:** `07a8bccc`  

## Description

Failed restore smoke: loader probed data index 0 before the restored offset

## Detailed plan

One node / eight H100s at batch priority, validating the full-state restore
from `step-213072` of `contacts-v1-exp277-m2-p06-full-epoch-1.5B` before the
266,345-step reshuffled second epoch. Ten updates, two validation batches,
and one checkpoint were expected.

## Changes from previous runs

First run to restore model, optimizer, RNG, and step state instead of
initializing from scratch, and the first to use data seed 1 with a second
finite permutation of the same 34,092,146 packed examples.

## Notes

**Failed. No optimizer update ran.** Submitted 2026-09-14T13:45Z, queued for
H100 capacity for about a day, and started W&B at 2026-09-15T20:35:11Z. The
checkpoint restore itself succeeded (`Loading checkpoint from
s3://.../checkpoints/step-213072`), then `levanter.main.train_lm` raised
while building the loader:

```
levanter/data/loader.py:140 initial_example = blocking_wait(self.data_store.getitem_async(0))
IndexError: continuation data starts at offset 27273344, got 0
```

`DataLoader.__init__` reads global data index 0 once to learn the example
structure, regardless of which step the trainer restored, and
`ContinuationEpochDataset.get_batch` rejected every index below its absolute
offset. Both the Iris driver `/bizon/exp277-continue-smoke-a01` and its GPU
child `exp277-train-2ff47238` reported `succeeded exit=0` despite the
traceback, and the step wrote `SUCCESS` to `.executor_status` at
`runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B-smoke/`; that
stale marker would have caused any rerun at the same output path to be
served from cache. Trust the worker traceback, not the job state.

Superseded by the repaired smoke, which serves the structure probe from the
first example of the new epoch and carries its attempt number in its output
identity.
