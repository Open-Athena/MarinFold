---
marinfold_run:
  user: zack
  launched_at: '2026-10-05T13:32:14Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Full-state exp299 continuation from step 126294 with 20% linear
    cooldown
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown20-s126294-v6e16-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-cooldown20-s126294-v6e16-a01
    run_name: delta-v2-exp277-cooldown20-s126294-v6e16-a01
  git_sha: 08f6e6cc9defa869af4679f919ec59d10ae01fc4
  iris_job_ids:
  - /zack/delta-v2-exp277-cooldown20-s126294-v6e16-a01
  - /zack/delta-v2-exp277-cooldown20-s126294-v6e8-a02
  - /zack/exp299-cooldown20-export-step165263-a03
  - /zack/exp299-cooldown20-export-step165263-a04
  - /zack/exp299-cooldown20-export-step165393-a05
  - /zack/exp299-cooldown20-export-step165393-v6e8-a07
  - /zack/exp299-cooldown20-export-bf16-step165393-a08
  - /zack/exp299-cooldown20-export-bf16-step165393-v6e8-a09
  - /zack/exp299-cooldown20-gcs-s3-export-step165393-a01
  - /zack/exp299-cooldown20-gcs-s3-export-step165393-a02
  - /zack/exp299-cooldown20-gcs-s3-export-step165393-a03
  - /zack/exp299-cooldown20-rprec-s165393-smoke-a01
  - /zack/exp299-cooldown20-rprec-s165393-0of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-1of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-2of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-3of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-4of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-5of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-6of8-a01
  - /zack/exp299-cooldown20-rprec-s165393-7of8-a01
  - /zack/exp299-cooldown-interim-aggregate-a02
  - /zack/exp299-cooldown-interim-bootstrap-a01
  - /zack/exp299-cooldown20-export-bf16-step205710-v6e8-a10
  - /zack/exp299-cooldown20-gcs-s3-export-step205710-a04
  - /zack/exp299-cooldown20-gcs-s3-export-step205710-a05
  - /zack/exp299-cooldown20-rprec-s205710-smoke-a02
  - /zack/exp299-cooldown20-rprec-s205710-0of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-1of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-2of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-3of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-4of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-5of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-6of8-a02
  - /zack/exp299-cooldown20-rprec-s205710-7of8-a02
  - /zack/exp299-cooldown-latest-aggregate-a01
  - /zack/exp299-cooldown-latest-bootstrap-a01
---
# 2026-10-05 · exp299_models_delta_stream_contacts · delta-v2-exp277-cooldown20-s126294-v6e16-a01

**Launched:** 2026-10-05T13:32:14Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-cooldown20-s126294-v6e16-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown20-s126294-v6e16-a01)  
**Git:** `08f6e6cc`  

## Description

Full-state exp299 continuation from step 126294 with 20% linear cooldown

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

The arm remained in progress on 2026-11-02. Its durable step-205710 snapshot was
exported as bf16 and evaluated on the canonical 670-protein, 100-rollout suite:
R/all 0.5979 and R/long 0.5477 overall; `eval-val` 0.5272 / 0.5037. All 67,000
generations parsed successfully. This is an interim checkpoint, not the final
matched-endpoint result.
