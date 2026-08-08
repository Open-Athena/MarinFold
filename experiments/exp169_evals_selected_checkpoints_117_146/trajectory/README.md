# Checkpoint trajectories

This extension evaluates how contact accuracy changes during three training
runs compared by exp169. It uses every permanent checkpoint from the 3B E8 and
1.5B E8 BS64 runs, plus every second permanent checkpoint from the 1.5B E16
run. The E16 sample includes the final checkpoint and spans twice the training
tokens with the same eight evaluation points.

Each checkpoint is one independent, resumable Iris job. The job reads its
Levanter checkpoint from region-local GCS, exports and casts it on ephemeral
disk, and runs the complete 554-protein evaluation with 100 rollout-resampled
generations per protein. GCS is read-only. Sparse votes, timing completion
markers, exact inputs, and provenance are written below:

```text
hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp169/
  trajectory/runs/<wandb-run-name>/step-<N>/
```

Compact per-protein metrics and summaries are published below
`trajectory/derived/`. Aggregate tables and the figure are published under
`trajectory/summary/` and stored in the parent experiment's `data/` and
`plots/` directories.

## Selection

| Run | Permanent checkpoints | Selected steps | Token span |
| --- | ---: | --- | ---: |
| exp146 3B E8 | 8 | 2230, 4460, 6690, 8920, 11150, 13380, 15610, 17839 | 4.68B–37.41B |
| exp117 1.5B E8 BS64 | 8 | 8920, 17840, 26760, 35680, 44600, 53520, 62440, 71359 | 4.68B–37.41B |
| exp117 1.5B E16 | 16 | 4460, 8920, 13380, 17840, 22300, 26760, 31220, 35679 | 9.35B–74.82B |

The BS64 run takes four optimizer steps for every step in the BS256 runs at a
fixed token budget. Its E8 schedule also ends after half as many tokens as the
E16 run, so this comparison describes the complete training configurations
rather than isolating batch size alone.

## Results

All three runs show a sharp transition from nearly random contacts to useful
contact prediction. The BS64 run moves first, from 0.0240 all-range R-precision
at 9.35B tokens to 0.1918 at 14.03B. At 18.71B it reaches 0.3392, ahead of the
3B at 0.2831 and the E16 run at 0.0256. This ordering follows their validation
losses at that point. The transition appears when loss falls to roughly 2.9,
rather than at one token count shared by every configuration.

The BS64 run leads the 3B through 23.38B tokens, then the 3B passes it between
23.38B and 28.06B. At their shared endpoint near 37.41B, the 3B reaches 0.5077
and BS64 reaches 0.4944, a paired difference of 0.0133 (95% CI [0.0093,
0.0173]). BS64 remains 0.0712 ahead of E16 at the same token budget (95% CI
[0.0640, 0.0784]). Continued E16 training eventually reaches 0.5338 at 74.82B
tokens.

The 3B improves at every measured checkpoint, so its trajectory provides no
evidence of contact overfitting. From epoch 7 to 8, validation loss falls by
0.0115 and all-range R-precision rises by 0.0225 (paired 95% CI [0.0181,
0.0269]). BS64 also improves through epoch 8. E16 is the exception late in
training: loss rises by 0.0067 from epoch 14 to 16 while R-precision rises by
0.0058 (95% CI [0.0020, 0.0095]). Loss locates the broad learning transition,
but does not order contact accuracy reliably once runs are close.

The plot is [`../plots/checkpoint_trajectory.png`](../plots/checkpoint_trajectory.png).
Plot-ready checkpoint rows are in
[`../data/trajectory_checkpoint_metrics.csv`](../data/trajectory_checkpoint_metrics.csv),
paired adjacent-checkpoint changes are in
[`../data/trajectory_paired_changes.csv`](../data/trajectory_paired_changes.csv),
paired comparisons at shared token budgets are in
[`../data/trajectory_matched_token_changes.csv`](../data/trajectory_matched_token_changes.csv),
and the complete per-protein metric table is in
[`../data/trajectory_metric_rows.csv.gz`](../data/trajectory_metric_rows.csv.gz).

## Submission

Load the open-athena write token and submit one checkpoint per job from this
directory. Every Iris command produced by the submitter includes `--user
eczech`, uses `marin-dev`, and sets the checkpoint's source region.

```bash
uv run --frozen python submit_trajectory.py \
  --run-key exp146-3b-e8 --tpu v6e-8

uv run --frozen python submit_trajectory.py \
  --run-key exp117-1_5b-e16 --tpu v6e-8

uv run --frozen python submit_trajectory.py \
  --run-key exp117-1_5b-e8-bs64 --tpu v6e-8
```

If a requested shape has no worker after ten minutes, stop only those queued
evaluation jobs and resubmit them on the next smaller co-located shape. Started
jobs keep their requested tensor-parallel configuration and output prefix.
Use only shapes that Iris represents as one task because one task owns each
checkpoint's output prefix.

## Production run

The 2026-08-08 submission used read-only autoscaler status from `marin-dev` to
choose shapes. A `v6e-16` probe expanded into four Iris tasks, so it was stopped
before any worker or output existed. Single-task `v6e-8` requests in both source
regions failed to provision within ten minutes. Europe `v5litepod-8` and
`v5litepod-4` probes then hit the project's v5e quota. The production jobs were
submitted on single-task `v6e-4` workers in `us-east1` and `europe-west4`.

Before fan-out, the first 3B worker was inspected with `iris task exec`. The
probe confirmed the regional GCS restore, exact model geometry, 5.59 GiB BF16
export, four-chip vLLM mesh, 554-target input, local FP32 cleanup, and durable HF
part upload. The remaining checkpoints were submitted only after the first 16
proteins had both vote and timing files in the public bucket.

All 24 jobs completed. The 1.5B E16 step-22300 worker was reclaimed after 480 of
554 proteins. During the BS64 extension, `v6e-8` could not provision within ten
minutes and `v5litepod-8` was blocked by project quota, so all eight jobs ran on
four concurrently available `v6e-4` slices. The step-44600 worker was replaced
after its first durable 16-protein part. Both replacements resumed from HF at
the first unfinished part instead of repeating completed work.

Progress can be read anonymously from the durable timing markers:

```bash
uv run --frozen --extra analysis python analyze_trajectory.py --status-only
```

With the write token loaded, omit `--status-only` to validate and finalize every
newly complete checkpoint. Re-running the command loads existing derivatives,
so a completed checkpoint is not downloaded and recomputed.
