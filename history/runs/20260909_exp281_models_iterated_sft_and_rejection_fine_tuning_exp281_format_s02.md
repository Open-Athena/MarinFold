---
marinfold_run:
  user: bizon
  launched_at: '2026-09-09T21:31:43Z'
  experiment: exp281_models_iterated_sft_and_rejection_fine_tuning
  kind: models
  short_description: Continue format warm-up from step 256 to 2000 with transition
    diagnostics
  wandb:
    url: https://wandb.ai/open-athena/MarinFold/runs/exp281-format-s02
    entity: open-athena
    project: MarinFold
    run_id: exp281-format-s02
    run_name: exp281-format-s02
  git_sha: 4721c76c
  iris_job_ids:
  - /bizon/exp281-format-s02
  - /bizon/exp281-format-s02-r1
---
# 2026-09-09 · exp281_models_iterated_sft_and_rejection_fine_tuning · exp281-format-s02

**Launched:** 2026-09-09T21:31:43Z by bizon\
**Kind:** models\
**Experiment:** exp281_models_iterated_sft_and_rejection_fine_tuning\
**W&B:** [exp281-format-s02](https://wandb.ai/open-athena/MarinFold/runs/exp281-format-s02)\
**Git:** `4721c76c`\

## Description

Continue format warm-up from step 256 to 2000 with transition diagnostics

## Detailed plan

Continue exp281-trial-s03 step 256 to global step 2,000 while preserving model,
Adam state, per-rank data offsets and RNG. Same frozen corpus, batch 32,
hypothesis weight 1 and plain rehearsal. New schedule: 100-step rewarmup from
10% of peak, then cosine decay to 10% at 2,000; peak LR 1e-4.

Eight H100s on cw-us-east-02a at batch priority, same regional S3 inputs/output.
Save every 250 global steps, validation every 125 plus initial step 256.
Final gate: same 25 internal-validation proteins, 8 completions per mode, seeds
281/282, 99% validity per mode and 90% natural multiple nonempty hypotheses.

## Changes from previous runs

Adds separate natural final-marker, final-answer, final-end and plain-end
teacher-forced loss/accuracy/count metrics. Checkpoint publication has a real
300-second process deadline and refuses to overwrite completed prefixes.
W&B uses an automatic log step with train/step as its x-axis to retain replayed
measurements after recovery. Nineteen tests and CPU/GPU continuation-resume
smokes passed before submission.

## Notes

Initial step-256 diagnostics: overall validation loss 2.244495; natural final-marker
loss 6.584968 with 0/6 top-1 correct; multi final-end loss 4.272908 with 0/9 correct.
These sparse teacher-forced diagnostics do not replace free-running evaluation.

Parent:
`s3://marin-us-east-02a/protein-structure/MarinFold/exp281/trial-s03/checkpoints/exp281-trial-s03/step-256`.

Output:
`s3://marin-us-east-02a/protein-structure/MarinFold/exp281/format-s02/`.

First attempt reached global step 1,750, then failed when publication exceeded
the 300-second deadline. Five preceding saves (500 through 1,500) succeeded in
97–101 seconds. Worker duration: 2,626.48 seconds. Latest complete checkpoint:
`s3://marin-us-east-02a/protein-structure/MarinFold/exp281/format-s02/checkpoints/exp281-format-s02/step-1500`.

At step 1,750: training loss 0.002406, validation loss 5.946770, natural-marker
loss 16.095710 and final-end loss 13.313005. Marker/end accuracies stayed zero
at all observed checks. The small frozen corpus is overfitting; this is not a
result for refreshed SFT or rejection training.

Recovery `/bizon/exp281-format-s02-r1` preserves the exact source/config and
resumes from step 1,500 under this same W&B run. It is queued in US-EAST-02A
as of 2026-09-09 23:43 UTC. The separate CPU cleanup job
`/bizon/exp281-format-s02-cleanup` succeeded; its small audit is under the output
prefix as `recovery-cleanup.json`. RNO2A recovery remains unsubmitted pending
explicit approval for approximately 53 GB of cross-region checkpoint I/O.

Step 2,000 and final free-running evaluation are not complete. Partial metrics,
raw history, publication timings and plots are committed in the experiment.
