---
marinfold_run:
  user: eczech
  launched_at: '2026-08-18T15:37:58Z'
  experiment: exp232_sweep_cv1_decontam
  kind: models
  short_description: 20-step peak-LR continuation smoke for exp232 m1-p02 with full
    augmentation
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    entity: open-athena
    project: MarinFold
    run_id: prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    run_name: prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
  git_sha: 1813cb9
  iris_job_ids:
  - /eczech/exp232-cont-smoke-s01-m1p02-rno2a-h100-n1-a01
---

# 2026-08-18 · exp232_sweep_cv1_decontam · prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1

**Launched:** 2026-08-18T15:37:58Z by eczech  
**Kind:** models  
**Experiment:** exp232_sweep_cv1_decontam  
**W&B:** [prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s01-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1)  
**Git:** `1813cb9`  

## Description

20-step peak-LR continuation smoke for exp232 m1-p02 with full augmentation

## Detailed plan

Restore full trainer state at step 116160, run 20 updates with augmentation on
every example, and exercise a short 80%-hold/20%-linear-decay LR schedule.

## Changes from previous runs

- Uses the exp232 decontaminated caches and m1-p02 peak-LR checkpoint.
- Writes only to a one-day temporary checkpoint path.

## Notes

Stopped during checkpoint restore after the companion m2-p06 smoke exposed the
exclusive linear-decay endpoint. It was superseded before any training update.
