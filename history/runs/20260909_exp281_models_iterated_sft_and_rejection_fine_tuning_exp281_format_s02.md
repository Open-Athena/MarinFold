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
  - /bizon/exp281-format-s02-rno-r1
  - /bizon/exp281-format-s02-eval-rno
  - /bizon/exp281-format-s02-report
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

On 2026-09-10 the user explicitly approved the RNO2A recovery and its estimated
53 GB of cross-region checkpoint I/O. The east-region recovery had never begun
training (Kueue scheduling gate); it was cancelled and confirmed killed before
submitting `/bizon/exp281-format-s02-rno-r1` at 14:17:25 UTC. The new attempt uses
eight H100s, batch priority, the unchanged training source fingerprint, and the
same step-1,500 checkpoint/run/config. The submission bundle revision is
`81448651`; its training source matches `4721c76c` byte-for-byte.

## Completed result: 2026-09-10

RNO2A recovery succeeded through global step 2,000 in 1,349.15 worker seconds.
Steps 1,750 and 2,000 each published in 218.6 seconds. Final checkpoint and
tokenizer: `s3://marin-us-east-02a/protein-structure/MarinFold/exp281/format-s02/checkpoints/exp281-format-s02/step-2000`.

Final training minibatch loss: 0.001169; held-out loss: 6.046831. Natural-marker
loss: 16.192715; final-end loss: 13.607997. Both sparse transition accuracies
remain zero. Extended repetition overfits the frozen pilot corpus.

The final natural/forced evaluation and scoring succeeded on one RNO2A H100 in
472.22 seconds, with one shared 5.886 GB inference-model download. Bundle
revision: `6ff2ca96`; generation source fingerprint matches the planned version.
Natural validity is 0/200; forced validity is 4/200 (2%), versus 0/200 and 14/200
in the pilot. Natural final-marker emission increases to 98/200, but valid
multiple-hypothesis trajectories remain absent. Invalid-zero final F1 is 0.0000
natural and 0.01220 forced. The fixed format gate fails; no synthesis or RFT
round was launched.

All 400 candidates and 50 timing rows pass the independent report audit.
The report job succeeded in 26.88 seconds and attached
`open-athena/MarinFold/exp281-format-s02-report:v0`. The 2,193,427-byte report is
public under `hf://buckets/open-athena/MarinFold/data/exp281/format-s02/report.zip`;
anonymous download SHA256 verification passed. The experiment's `data/` records
the exact checksum, raw history, canonical replay-aware CSV and per-input metrics.
