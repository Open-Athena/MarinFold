---
marinfold_run:
  user: bizon
  launched_at: '2026-09-10T14:57:39Z'
  experiment: exp279_models_exact_soft_contact_targets
  kind: models
  short_description: Soft-target LR 0.002 full-state continuation from step 14520;
    5000 updates on 32 Reno H100s
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr200-step14520-a01
    entity: open-athena
    project: MarinFold
    run_id: exp279-soft-lr200-step14520-a01
    run_name: exp279-soft-lr200-step14520-a01
  git_sha: 5e11b8a6847f79b00a572f5c9777a9394fd69ee9
  iris_job_ids:
  - /bizon/exp279-lr200-rno-full-a01
---

# 2026-09-10 · exp279_models_exact_soft_contact_targets · exp279-soft-lr200-step14520-a01

**Launched:** 2026-09-10T14:57:39Z by bizon

**Kind:** models

**Experiment:** exp279_models_exact_soft_contact_targets

**W&B:** [exp279-soft-lr200-step14520-a01](https://wandb.ai/open-athena/MarinFold/runs/exp279-soft-lr200-step14520-a01)

**Git:** `5e11b8a6`


## Description

Soft-target LR 0.002 full-state continuation from step 14520; 5000 updates on 32 Reno H100s

## Detailed plan

Fork the exact soft production step-14520 full state for 5,000 updates at LR 0.002. Compare ordinary validation CE, contact-endpoint CE, and gradient/clipping diagnostics at matched updates.

## Changes from previous runs

Same model, Adam moments, RNG, data position, frozen decontaminated caches, batch 128 and microbatch 32. Only LR, bounded duration and diagnostics change.

## Notes

Runtime source is the frozen manifest revision above; later launcher-checkout commits only update records. All three continuations share the same source/input identity, recorded in the experiment data/lr_sweep_launch.json. Completed all 5,000 additional updates at step 19520 on 2026-09-10; exact Iris dispatch succeeded without failures or preemptions. Final ordinary document CE 3.360499; contact-endpoint CE 4.358667. The 0.001 control leads both metrics in this continuation test.

Final native checkpoint: `s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-lr200-step14520-a01/step-19520`. Full model/Adam/RNG array-manifest shapes and dtypes verified. HF model/config/tokenizer file presence verified at `s3://marin-us-east-02a/MarinFold/exp279/checkpoints/exp279-soft-lr200-step14520-a01/hf/step-19520`; final weights were not fully reloaded or evaluated for contact accuracy. Final results and curves are recorded in the experiment data/lr_completion.json and data/lr_validation.csv.
