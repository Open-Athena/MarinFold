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
  - /timodonnell/exp347-4b-contacts-v1-corpus235m-eval-val-a07
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

November 5: user explicitly authorized restarting the stopped evaluator. Replacement a07 was accepted at 18:54:21 UTC in RNO2A, retaining this W&B identity and the 1B-token schedule. It uses 12 independent single-H100 shards per checkpoint; combined evaluation/training reservations remain within the 48-H100 cap. The 97-protein, 100-rollout contract and checkpoint-local outputs are unchanged. Recovered results and timings are collected with `collect_periodic_evals.py`, checked against per-protein tables and worker checkpoint identities, and plotted by `plot_eval_progress.py`.

November 5 refreshed result: step 64544 / 10,000,117,732 additional tokens; all-range R=0.28559874, long-range R=0.24709758. All 97 proteins completed 100 rollouts. Full per-protein metrics, timings, and 12 worker provenance records are committed for each new checkpoint.

November9: all19 periodic raw milestones are complete. Latest step119246/19000027107tokens scores0.3378773050 all and0.3093641769 long; best step114806/+18B scores0.3470314515/0.3165192998. Nine new milestones each have97proteins×100terminatedrollouts with metrics, timings and12worker provenance files. Driver a07 remains live and caught up.

November10: collected the completed20B milestone at step127506 /20,000,079,692 additional tokens. All97 eval-val proteins have100 terminated rollouts and measured timings. R-precision all=.34567054900005284, long=.32119943032891185. Canonical result prefix: s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/eval-val/exp347-qwen35-4b-contacts_v1-corpus235m-100bt/step-127506. Driver a07 remains running and caught up.

November 13: collected the completed 21B and 22B milestones, steps 133218 and
139024. Latest 22,000,068,018 additional tokens score 0.36120782042183486
all-range and 0.3346670424390309 long-range R-precision. Both have all 97 eval-val
proteins × 100 terminated rollouts, measured timings, and 12 worker provenance
records. Driver a07 is running and caught up. Canonical regional prefix:
`s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/eval-val/exp347-qwen35-4b-contacts_v1-corpus235m-100bt/step-139024`.
