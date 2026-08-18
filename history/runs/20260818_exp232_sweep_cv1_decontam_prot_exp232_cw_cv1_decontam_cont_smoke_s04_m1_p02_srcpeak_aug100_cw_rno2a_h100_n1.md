---
marinfold_run:
  user: eczech
  launched_at: '2026-08-18T16:08:19Z'
  experiment: exp232_sweep_cv1_decontam
  kind: models
  short_description: Validated inclusive-zero continuation smoke for exp232 m1-p02
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    entity: open-athena
    project: MarinFold
    run_id: prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
    run_name: prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1
  git_sha: 73e6215
  iris_job_ids:
  - /eczech/exp232-cont-smoke-s04-m1p02-rno2a-h100-n1-a01
---

# 2026-08-18 · exp232_sweep_cv1_decontam · prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1

**Launched:** 2026-08-18T16:08:19Z by eczech  
**Kind:** models  
**Experiment:** exp232_sweep_cv1_decontam  
**W&B:** [prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s04-m1-p02-srcpeak-aug100-cw-rno2a-h100-n1)  
**Git:** `73e6215`  

## Description

Validated inclusive-zero continuation smoke for exp232 m1-p02

## Detailed plan

Run 20 continuation updates from the exp232 m1-p02 pre-cooldown
`step-116160` checkpoint. Hold the source peak LR through 80% of the smoke,
cool over four updates, and exercise strict augmentation on every training
example.

## Changes from previous runs

- Moves the registered inclusive LR schedule to a stable importable module.
- Uses a temporary one-day checkpoint root and one H100 node for validation.

## Notes

Succeeded on Iris and finished in W&B without training errors. LR was
`3.1623e-4` through step 116177, then `2.1082e-4`, `1.0541e-4`, and exactly
`0` at step 116180. The strict full-rate augmentation wrapper raised no
invariant errors. The
temporary final checkpoint was verified in CoreWeave object storage.
