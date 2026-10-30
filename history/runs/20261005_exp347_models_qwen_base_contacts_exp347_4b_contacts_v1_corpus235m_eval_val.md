---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T16:10:25Z'
  experiment: exp347_models_qwen_base_contacts
  kind: evals
  short_description: Periodic Qwen checkpoint eval-val R-precision
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-contacts-v1-corpus235m-eval-val
    entity: open-athena
    project: MarinFold
    run_id: exp347-4b-contacts-v1-corpus235m-eval-val
    run_name: exp347-4b-contacts-v1-corpus235m-eval-val
  git_sha: e0182f6cdcdb279323d2b3de833cdc3ff1ba001c
  iris_job_ids:
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a01
  - /timodonnell/exp347-raw-step4620-eval-a01-s00
  - /timodonnell/exp347-raw-step4620-eval-a01-s01
  - /timodonnell/exp347-raw-step4620-eval-a01-s02
  - /timodonnell/exp347-raw-step4620-eval-a01-s03
  - /timodonnell/exp347-raw-step4620-eval-a01-s04
  - /timodonnell/exp347-raw-step4620-eval-a01-s05
  - /timodonnell/exp347-raw-step4620-eval-a01-s06
  - /timodonnell/exp347-raw-step4620-eval-a01-s07
  - /timodonnell/exp347-raw-step4620-eval-a01-s08
  - /timodonnell/exp347-raw-step4620-eval-a01-s09
  - /timodonnell/exp347-raw-step4620-eval-a01-s10
  - /timodonnell/exp347-raw-step4620-eval-a01-s11
  - /timodonnell/exp347-raw-step4620-eval-smoke-a01-s00
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a02
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a03
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a05
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a06
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-4b-contacts-v1-corpus235m-eval-val

**Launched:** 2026-10-05T16:10:25Z by bizon  
**Kind:** evals  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-4b-contacts-v1-corpus235m-eval-val](https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-contacts-v1-corpus235m-eval-val)  
**Git:** `e0182f6c`  

## Description

Periodic Qwen checkpoint eval-val R-precision

## Detailed plan

_(Why we ran this, what we expect to see, unusual parameters.)_

## Changes from previous runs

_(Bullet list of differences from the last run of this kind.)_

## Notes

October 26: a03 failed at its first 1B request because a child submitted through
the RNO controller requested another federation hop to RNO. After removing that
directive, a05 workers exposed an old east-only checkpoint guard and failed.
Both are fixed; a06 loaded the local step-8406 export and began scoring proteins
by 16:05 UTC. Five requests (1B through 5B) await complete aggregation. The metric
contract is unchanged and all new outputs remain in marin-rno2a.

October 26: attempted a04 submission was rejected by the Iris client date gate
before a job was created (client October 8, required October 9).

October 21 interim checkpoint: step 4620 / 498,970,942 additional tokens.
All 97 eval-val proteins completed 100 rollouts each, with zero unfinished samples.
All-range R-precision 0.144811996974225; long-range 0.10787052842698255.
The one-protein smoke and 12 direct root workers contributed to this measurement;
the persistent driver imported the committed aggregate and logged it to this run.
Metrics, timings, dispatches, and per-worker provenance are under
`experiments/exp347_models_qwen_base_contacts/data/eval_val_interim_20261021/contacts_v1/`
and the public HF bucket at `data/exp347-qwen-base-contacts/data/eval_val_interim_20261021/contacts_v1/`.
The regular driver remains active at the 1B-additional-token cadence.

October 22: prior east a01 driver confirmed killed before replacement. The a02 driver is live in RNO2A, watches the regional request prefix, and explicitly places GPU children there. New checkpoint exports and evaluation results stay in marin-rno2a. Evaluation cadence remains 1B additional tokens; October 21 remains the latest completed R-precision point.

October 30: all six periodic milestones (1B through6B) are complete. Latest step39284/6000012722 tokens: all R=0.2406436906219813, long R=0.19799282973966936. Each97 proteins x100 terminated rollouts. Driver a06 remains live, caught up with the request queue. Small artifacts and per-input timings saved under experiments/exp347_models_qwen_base_contacts/data/eval_val_periodic/contacts_v1/.
