---
marinfold_run:
  user: zack
  launched_at: '2026-10-05T13:32:14Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Full-state exp299 continuation from step 126294 with 30% linear
    cooldown
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown30-s126294-v6e16-a01
    entity: open-athena
    project: MarinFold
    run_id: delta-v2-exp277-cooldown30-s126294-v6e16-a01
    run_name: delta-v2-exp277-cooldown30-s126294-v6e16-a01
  git_sha: 08f6e6cc9defa869af4679f919ec59d10ae01fc4
  iris_job_ids:
  - /zack/delta-v2-exp277-cooldown30-s126294-v6e16-a01
  - /zack/exp299-cooldown30-export-step165615-a03
  - /zack/exp299-cooldown30-export-step165615-a04
  - /zack/exp299-cooldown30-export-step165730-a05
  - /zack/exp299-cooldown30-export-step165730-v6e8-a09
  - /zack/exp299-cooldown30-gcs-s3-export-step165730-a01
  - /zack/exp299-cooldown30-gcs-s3-export-step165730-a02
  - /zack/exp299-cooldown30-rprec-s165730-smoke-a01
  - /zack/exp299-cooldown30-rprec-s165730-0of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-1of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-2of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-3of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-4of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-5of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-6of8-a01
  - /zack/exp299-cooldown30-rprec-s165730-7of8-a01
  - /zack/exp299-cooldown-interim-aggregate-a02
  - /zack/exp299-cooldown-interim-bootstrap-a01
  - /zack/exp299-cooldown30-export-bf16-step210491-v6e8-a10
  - /zack/exp299-cooldown30-gcs-s3-export-step210491-a04
  - /zack/exp299-cooldown30-gcs-s3-export-step210491-a05
  - /zack/exp299-cooldown30-rprec-s210491-smoke-a02
  - /zack/exp299-cooldown30-rprec-s210491-0of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-1of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-2of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-3of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-4of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-5of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-6of8-a02
  - /zack/exp299-cooldown30-rprec-s210491-7of8-a02
  - /zack/exp299-cooldown-latest-aggregate-a01
  - /zack/exp299-cooldown-latest-bootstrap-a01
---
# 2026-10-05 · exp299_models_delta_stream_contacts · delta-v2-exp277-cooldown30-s126294-v6e16-a01

**Launched:** 2026-10-05T13:32:14Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [delta-v2-exp277-cooldown30-s126294-v6e16-a01](https://wandb.ai/open-athena/MarinFold/runs/delta-v2-exp277-cooldown30-s126294-v6e16-a01)  
**Git:** `08f6e6cc`  

## Description

Full-state exp299 continuation from step 126294 with 30% linear cooldown

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

Training completed successfully at step 210491 with `run_progress=1`, final
train loss 0.72254, and validation loss 0.97386. Canonical evaluation over 670
proteins and 100 rollouts per protein produced R/all 0.6136 and R/long 0.5668;
`eval-val` was 0.5469 / 0.5249. All 67,000 generations parsed successfully.
Paired against the completed CoreWeave 20%-cooldown control at the same endpoint,
the differences were -0.00054 R/all (95% CI [-0.00509, 0.00367]) and +0.00010
R/long ([-0.00523, 0.00528]), establishing a tie rather than a new leader.
