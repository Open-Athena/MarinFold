---
marinfold_run:
  user: zack
  launched_at: '2026-09-17T19:04:23Z'
  experiment: exp299_models_delta_stream_contacts
  kind: models
  short_description: Fresh 78.5k-step V2 delta-stream run matched to exp177 protein
    exposure
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17
    entity: open-athena
    project: MarinFold
    run_id: protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17
    run_name: protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17
  git_sha: fc8745cce37b5fed9f128cd3d6f06114effbcf3b
  iris_job_ids:
  - /zack/exp299-v2-train-full78500-driver-a17
  - /zack/exp299-v2-train-full78500-driver-a17/protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17
  - /zack/exp299-v2-full-export-step22000-a01
  - /zack/exp299-v2-full-mirror-hf-step22000-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s0of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s1of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s2of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s3of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s4of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s5of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s6of8-a01
  - /zack/exp299-v2-full-rprec-step22000-u670-s7of8-a01
---

# 2026-09-17 · exp299_models_delta_stream_contacts · protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17

**Launched:** 2026-09-17T19:04:23Z by zack  
**Kind:** models  
**Experiment:** exp299_models_delta_stream_contacts  
**W&B:** [protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17](https://wandb.ai/open-athena/MarinFold/runs/protein-delta-stream-v2-1_5b-1e-3-2x4gb200-full78500-a17)  
**Git:** `fc8745cc`  

## Description

Fresh 78.5k-step V2 delta-stream run matched to exp177 protein exposure

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

At step 22,000 (28.0% of the configured schedule), validation loss was 1.036.
The canonical 100-rollout evaluation covered all 670 targets with no malformed
V2 samples. On the legacy 554 proteins, R-precision was 0.4691 all-range and
0.4028 long-range, versus 0.0250 and 0.0223 for the exposure- and
schedule-matched exp177 contacts-v1 step-20,000 control. On the 19-protein
exp277 de-novo split, V2 scored 0.4383 all-range and 0.3935 long-range.
