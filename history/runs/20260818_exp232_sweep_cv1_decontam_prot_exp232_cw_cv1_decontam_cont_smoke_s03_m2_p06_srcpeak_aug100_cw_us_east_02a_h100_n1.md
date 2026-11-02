---
marinfold_run:
  user: eczech
  launched_at: '2026-08-18T16:01:21Z'
  experiment: exp232_sweep_cv1_decontam
  kind: models
  short_description: Registered inclusive-zero continuation smoke for exp232 m2-p06
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1
    entity: open-athena
    project: MarinFold
    run_id: prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1
    run_name: prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1
  git_sha: afa2e21
  iris_job_ids:
  - /eczech/exp232-cont-smoke-s03-m2p06-use02a-h100-n1-a01
---

# 2026-08-18 · exp232_sweep_cv1_decontam · prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1

**Launched:** 2026-08-18T16:01:21Z by eczech  
**Kind:** models  
**Experiment:** exp232_sweep_cv1_decontam  
**W&B:** [prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1](https://wandb.ai/open-athena/MarinFold/runs/prot-exp232-cw-cv1-decontam-cont-smoke-s03-m2-p06-srcpeak-aug100-cw-us-east-02a-h100-n1)  
**Git:** `afa2e21`  

## Description

Registered inclusive-zero continuation smoke for exp232 m2-p06

## Detailed plan

Revalidate m2-p06 with an inclusive final LR and a registered W&B schedule config.

## Changes from previous runs

- Registers the custom LR schedule as `linear_inclusive`.

## Notes

Stopped before training because the class was still created under `__main__` and
lost its registry identity across Iris serialization. Superseded by s04 after
moving the schedule to an importable module.
