---
marinfold_run:
  user: bizon
  launched_at: '2026-10-05T16:11:35Z'
  experiment: exp347_models_qwen_base_contacts
  kind: evals
  short_description: Periodic Qwen checkpoint eval-val R-precision
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-prompted-corpus235m-eval-val
    entity: open-athena
    project: MarinFold
    run_id: exp347-4b-prompted-corpus235m-eval-val
    run_name: exp347-4b-prompted-corpus235m-eval-val
  git_sha: e0182f6cdcdb279323d2b3de833cdc3ff1ba001c
  iris_job_ids:
  - /timodonnell/exp347-4b-prompted-corpus235m-eval-val-a01
  - /timodonnell/exp347-prompt-step12051-eval-a01-s00
  - /timodonnell/exp347-prompt-step12051-eval-a01-s01
  - /timodonnell/exp347-prompt-step12051-eval-a01-s02
  - /timodonnell/exp347-prompt-step12051-eval-a01-s03
  - /timodonnell/exp347-prompt-step12051-eval-a01-s04
  - /timodonnell/exp347-prompt-step12051-eval-a01-s05
  - /timodonnell/exp347-prompt-step12051-eval-a01-s06
  - /timodonnell/exp347-prompt-step12051-eval-a01-s07
  - /timodonnell/exp347-prompt-step12051-eval-a01-s08
  - /timodonnell/exp347-prompt-step12051-eval-a01-s09
  - /timodonnell/exp347-prompt-step12051-eval-a01-s10
  - /timodonnell/exp347-prompt-step12051-eval-a01-s11
  - /timodonnell/exp347-prompt-step12051-eval-smoke-a01-s00
  - /timodonnell/exp347-4b-prompted-corpus235m-eval-val-a02
  - /timodonnell/exp347-4b-prompted-corpus235m-eval-val-a03
  - /timodonnell/exp347-4b-prompted-corpus235m-eval-val-a04
  - /timodonnell/exp347-4b-prompted-corpus235m-eval-val-a05
---
# 2026-10-05 · exp347_models_qwen_base_contacts · exp347-4b-prompted-corpus235m-eval-val

**Launched:** 2026-10-05T16:11:35Z by bizon  
**Kind:** evals  
**Experiment:** exp347_models_qwen_base_contacts  
**W&B:** [exp347-4b-prompted-corpus235m-eval-val](https://wandb.ai/open-athena/MarinFold/runs/exp347-4b-prompted-corpus235m-eval-val)  
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
directive, a04 workers exposed an old east-only checkpoint guard and failed.
Both are fixed; a05 loaded the local step-23470 export and began scoring proteins
by 16:05 UTC. Two requests (1B and 2B) await complete aggregation. The metric
contract is unchanged and all new outputs remain in marin-rno2a.

October 21 interim checkpoint: step 12051 / 450,586,475 additional tokens.
All 97 eval-val proteins completed 100 rollouts each, with zero unfinished samples.
All-range R-precision 0.15239278226983136; long-range 0.10738360714287244.
One sampled prefix required two continuation rounds; all final samples terminated.
The one-protein smoke and 12 direct root workers contributed to this measurement;
the persistent driver imported the committed aggregate and logged it to this run.
Metrics, timings, dispatches, and per-worker provenance are under
`experiments/exp347_models_qwen_base_contacts/data/eval_val_interim_20261021/prompted/`
and the public HF bucket at `data/exp347-qwen-base-contacts/data/eval_val_interim_20261021/prompted/`.
The regular driver remains active at the 1B-additional-token cadence.

October 22: prior east a01 driver confirmed killed before replacement. The a02 driver is live in RNO2A, watches the regional request prefix, and explicitly places GPU children there. New checkpoint exports and evaluation results stay in marin-rno2a. Evaluation cadence remains 1B additional tokens; October 21 remains the latest completed R-precision point.
