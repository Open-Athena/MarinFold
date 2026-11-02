---
marinfold_run:
  user: eczech
  launched_at: '2026-08-18T15:52:48Z'
  experiment: exp232_sweep_cv1_decontam
  kind: models
  short_description: Inclusive-zero 20-step continuation smoke for exp232 m1-p02
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    entity: open-athena
    project: MarinFold
    run_id: prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    run_name: prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
  git_sha: 7ff67a9
  iris_job_ids:
  - /eczech/exp232-cont-smoke-s02-m1p02-rno2a-h100-n1-a01
---

# 2026-08-18 · exp232_sweep_cv1_decontam · prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1

**Launched:** 2026-08-18T15:52:48Z by eczech  
**Kind:** models  
**Experiment:** exp232_sweep_cv1_decontam  
**W&B:** [prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s02-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1)  
**Git:** `7ff67a9`  

## Description

Inclusive-zero 20-step continuation smoke for exp232 m1-p02

## Detailed plan

Revalidate m1-p02 after changing the linear cooldown to reach zero on the final
executed update.

## Changes from previous runs

- Uses the inclusive-endpoint schedule introduced after s01.

## Notes

Stopped before training after W&B configuration logging showed that the custom
schedule lacked a stable Draccus choice name. Superseded by s03 after registering
it as `linear_inclusive`.
