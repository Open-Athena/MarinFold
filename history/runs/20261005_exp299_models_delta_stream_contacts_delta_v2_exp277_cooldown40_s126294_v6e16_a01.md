---
marinfold_run:
  user: zack
  launched_at: '2026-10-05T13:32:15Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Full-state exp299 continuation from step 126294 with 40% linear
    cooldown
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown40-s126294-v6e16-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-cooldown40-s126294-v6e16-a01
    run_name: delta-v2-exp277-cooldown40-s126294-v6e16-a01
  git_sha: 08f6e6cc9defa869af4679f919ec59d10ae01fc4
  iris_job_ids:
  - /zack/delta-v2-exp277-cooldown40-s126294-v6e16-a01
  - /zack/delta-v2-exp277-cooldown40-s126294-v6e8-a02
  - /zack/exp299-cooldown40-export-step147250-a03
  - /zack/exp299-cooldown40-export-step147250-a04
  - /zack/exp299-cooldown40-export-step147343-a05
  - /zack/exp299-cooldown40-export-step147343-v6e8-a09
  - /zack/exp299-cooldown40-gcs-s3-export-step147343-a01
  - /zack/exp299-cooldown40-gcs-s3-export-step147343-a02
  - /zack/exp299-cooldown40-rprec-s147343-smoke-a01
  - /zack/exp299-cooldown40-rprec-s147343-0of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-1of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-2of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-3of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-4of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-5of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-6of8-a01
  - /zack/exp299-cooldown40-rprec-s147343-7of8-a01
  - /zack/exp299-cooldown-interim-aggregate-a02
  - /zack/exp299-cooldown-interim-bootstrap-a01
  - /zack/exp299-cooldown40-export-bf16-step177838-v6e8-a10
  - /zack/exp299-cooldown40-gcs-s3-export-step177838-a04
  - /zack/exp299-cooldown40-gcs-s3-export-step177838-a05
  - /zack/exp299-cooldown40-rprec-s177838-smoke-a02
  - /zack/exp299-cooldown40-rprec-s177838-0of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-1of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-2of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-3of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-4of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-5of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-6of8-a02
  - /zack/exp299-cooldown40-rprec-s177838-7of8-a02
  - /zack/exp299-cooldown-latest-aggregate-a01
  - /zack/exp299-cooldown-latest-bootstrap-a01
---
# 2026-10-05 · exp299_models_delta_stream_contacts · delta-v2-exp277-cooldown40-s126294-v6e16-a01

**Launched:** 2026-10-05T13:32:15Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-cooldown40-s126294-v6e16-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown40-s126294-v6e16-a01)  
**Git:** `08f6e6cc`  

## Description

Full-state exp299 continuation from step 126294 with 40% linear cooldown

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

The arm remained in progress on 2026-11-02. Its durable step-177838 snapshot was
exported as bf16 and evaluated on the canonical 670-protein, 100-rollout suite:
R/all 0.5785 and R/long 0.5268 overall; `eval-val` 0.4909 / 0.4674. All 67,000
generations parsed successfully. This is an interim checkpoint, substantially
earlier than the other evaluated arms, and is not a matched-endpoint schedule
comparison.
