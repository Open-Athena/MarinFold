---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T15:18:45Z'
  experiment: exp347_models_qwen_base_contacts
  kind: models
  short_description: Qwen3.5 base full-weight contact fine-tuning
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-contacts_v1-full-100bt
    entity: open-athena
    project: MarinFold
    run_id: exp347-qwen35-4b-contacts_v1-full-100bt
    run_name: exp347-qwen35-4b-contacts_v1-full-100bt
  git_sha: 2e9d376ac57cd4542eb433778cc8945b44453185
  iris_job_ids:
  - /timodonnell/exp347-qwen35-4b-contacts_v1-full-100bt-a01
---

# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-qwen35-4b-contacts_v1-full-100bt

**Launched:** 2026-10-05T15:18:45Z by bizon  
**Kind:** models  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-qwen35-4b-contacts_v1-full-100bt](https://wandb.ai/open-athena/MarinFold/runs/exp347-qwen35-4b-contacts_v1-full-100bt)  
**Git:** `2e9d376a`  

## Description

Qwen3.5 base full-weight contact fine-tuning

## Detailed plan

Full-corpus 4B phase: 100B additional native tokens on 3,717,047 training proteins; eight H100s, 32 complete documents per update. Initialize from the corresponding 1B pilot FP32 checkpoint at step 6496. Evaluate the frozen 97-protein eval-val set with 100 rollouts every 1B additional tokens and at completion. The100Bbudget is an announced operator default under the full-scale request; execution timeout 90 days.

## Changes from previous runs

Use all 2,067 AFDB shards, with fresh optimizer/data cursor and a new run identity. The objective, tokenizer, document transformation, maximum length and optimizer hyperparameters match the pilot.

## Notes

Checkpoint0 on the accuracy curve is the corresponding 1B-token pilot. Durable periodic BF16 exports live outside rolling optimizer checkpoints. Separate persistent Iris evaluation drivers retain the final result after the trainer exits. Initial per-protein validation timings are committed under data/full_phase_initial_timings. No eval-test read.
