---
marinfold_run:
  user: timodonnell
  launched_at: '2026-11-19T11:30:52Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: 'TPU continuation gate: first microbatch OOM before any optimizer
    update'
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10
    entity: open-athena
    project: MarinFold
    run_id: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10
    run_name: debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10
  git_sha: dd20bb80
  iris_job_ids:
  - /timodonnell/exp347-tpu-contacts_v1-continue-a10
---

# 2026-11-19 · exp347_models_qwen_base_contacts · debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10

**Launched:** 2026-11-19T11:30:52Z by timodonnell  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10](https://wandb.ai/open-athena/MarinFold/runs/debug-exp347-qwen35-4b-contacts_v1-corpus235m-tpu-east5-a10)  
**Git:** `dd20bb80`  

## Description

TPU continuation gate: first microbatch OOM before any optimizer update

## Detailed plan

Two optimizer updates and full checkpoint recovery on one v5p-8 in us-east5-a,
then continuous training on the same slice only after a successful gate.
Four-document microbatches, global batch 32, full 16K context, frozen raw
parent step 59319; all new artifacts remain in us-east5.

## Changes from previous runs

Explicit FSDP gradient and accumulation output layouts, which passed the
four-device CPU parity regression. Replaced still-pending a09 with a fixed
two-stage job; the gate has a two-hour timeout and any failure prevents
production from starting. Source commit dd20bb80.

## Notes

Terminal failed November 19 at 11:32:59 UTC after one scheduler preemption.
W&B registered at 11:30:52 UTC. Before any optimizer update, loading the first
microbatch's jit_loss_fn executable required 81.17G with 80.05G available
(RuntimeProgramAllocationFailure). No microbatch completed, and the runtime
gradient-layout check was therefore never reached. CPU correctness does not
prove full-model hardware memory fit. The production continuation did not
start and has no W&B registration. No unchanged retry submitted.
