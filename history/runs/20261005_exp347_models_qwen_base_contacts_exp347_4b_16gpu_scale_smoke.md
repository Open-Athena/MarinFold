---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T14:58:19Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Qwen3.5 base full-weight contact fine-tuning
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-16gpu-scale-smoke
    entity: open-athena
    project: MarinFold
    run_id: exp347-4b-16gpu-scale-smoke
    run_name: exp347-4b-16gpu-scale-smoke
  git_sha: bcc2f0c5fe89cc2ed4214436bfe5aa1710b146e3
  iris_job_ids:
  - /timodonnell/exp347-4b-16gpu-scale-smoke-a06
  - /timodonnell/exp347-4b-16gpu-scale-smoke-a07
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-4b-16gpu-scale-smoke

**Launched:** 2026-10-05T14:58:19Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-4b-16gpu-scale-smoke](https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-16gpu-scale-smoke)  
**Git:** `bcc2f0c5`  

## Description

Qwen3.5 base full-weight contact fine-tuning

## Detailed plan

Engineering validation of 16-rank training, durable periodic inference exports, request publication and optimizer recovery; not a scientific accuracy trial.

## Changes from previous runs

Static rendezvous uses the Iris routable IPv4 address, with node-local model staging and local CUDA device selection. The NCCL interface is selected from that IPv4 address.

## Notes

a06 completed the 1.5M-token smoke at step 6 / 1,808,229 tokens. It retained periodic exports at steps 2, 4, 6 after optimizer checkpoint pruning. a07 restored the exact NLL 0.4846291195346441 and committed step 7 / 2,119,709 tokens with 16 optimizer shards and an independent inference export/request. Its optional 3M-token transport test was stopped after correctness passed. Native IB fell back to Socket because the pinned image lacks libibverbs; its throughput was below 8-GPU execution, so the profile is excluded from production. Iris tasks were verified terminal. W&B may label the intentional stop as interrupted; this is not a failed full-scale training run.
