# Exp199 contact prediction evaluations

This workspace evaluates an extensible catalog of exp199 checkpoints with the
fixed 554-protein contact set and 100 rollout-resampled generations per protein.
Each checkpoint runs as one resumable Iris job. Native Levanter checkpoints
remain read-only in their region-local GCS buckets.

## Finished exp199 runs and their control

The completed `prot-exp199-cv1-s01-m1-p03-aug-us-east1` run has nine permanent
checkpoints. The forced final checkpoint is step 72,599. All nine steps are in
`checkpoint_specs.py`, so a later trajectory evaluation only requires selecting
another catalog key.

The only other completed trial is
`prot-exp199-cv1-s01-m1-p06-aug-us-east1`. Its forced final checkpoint is also
step 72,599. `exp199_final_checkpoint()` makes each future completed trial a
one-line catalog addition.

The control is the exact exp117 1.5B, 16-epoch step 35,679 HF export used by PR
#190. Its repository revision is pinned. The exp199 analyzer first rescored PR
#190's lossless sparse votes and recovered mean all-range R-precision
`0.5335961341539802` across 554 proteins. End-to-end generation is stochastic.
The two fresh control runs produced `0.5347972614575084` and
`0.535215598085612`. Both passed PR #190's 0.006 tolerance, but neither exactly
reproduced the archived value.

Submit or resume each checkpoint independently:

```bash
uv run --frozen python submit_contact_eval.py \
  --checkpoint exp117-control-step35679 \
  --run-tag <unique-tag> \
  --cluster marin-dev \
  --user eczech

uv run --frozen python submit_contact_eval.py \
  --checkpoint s01-m1-p03-aug-step72599 \
  --run-tag <unique-tag> \
  --cluster marin-dev \
  --user eczech

uv run --frozen python submit_contact_eval.py \
  --checkpoint s01-m1-p06-aug-step72599 \
  --run-tag <unique-tag> \
  --cluster marin-dev \
  --user eczech
```

Catalog placement sends the control to `europe-west4` and exp199 checkpoints to
`us-east1`. Each job requests one `v6e-4`.

The isolated `rerun02-20260809` jobs were submitted concurrently on 2026-08-09.
All three succeeded without a failure or preemption:

- [`/eczech/marinfold-exp199-exp117-control-step35679-eval-rerun02-20260809`](https://iris-dev.oa.dev/#/job/%2Feczech%2Fmarinfold-exp199-exp117-control-step35679-eval-rerun02-20260809), 45m 54s
- [`/eczech/marinfold-exp199-s01-m1-p03-aug-step72599-eval-rerun02-20260809`](https://iris-dev.oa.dev/#/job/%2Feczech%2Fmarinfold-exp199-s01-m1-p03-aug-step72599-eval-rerun02-20260809), 43m 57s
- [`/eczech/marinfold-exp199-s01-m1-p06-aug-step72599-eval-rerun02-20260809`](https://iris-dev.oa.dev/#/job/%2Feczech%2Fmarinfold-exp199-s01-m1-p06-aug-step72599-eval-rerun02-20260809), 44m 53s

## Final checkpoint results

| Run | Step | contacts-v1 loss | R-all | R-short | R-medium | R-long |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [exp117 1.5B E16 control](https://wandb.ai/eric-czech/marin/runs/prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4) | 35,679 | 2.703709 | 0.535216 | 0.629265 | 0.585215 | 0.484878 |
| [exp199 p03-aug](https://wandb.ai/eric-czech/marin/runs/prot-exp199-cv1-s01-m1-p03-aug-us-east1) | 72,599 | 3.011531 | 0.574333 | 0.660688 | 0.623301 | 0.526172 |
| [exp199 p06-aug](https://wandb.ai/eric-czech/marin/runs/prot-exp199-cv1-s01-m1-p06-aug-us-east1) | 72,599 | 3.054504 | 0.524407 | 0.629089 | 0.579260 | 0.469668 |

Loss is `eval/tokenized/contacts-v1-val/loss` from W&B at the listed checkpoint
step. Every R value in the table comes from the fresh `rerun02-20260809` jobs.
The three manifests agree on the evaluator revision, 554 targets, target hashes,
100 rollouts, sampling settings, and tensor parallelism.

The current scorer can separately reproduce PR #190's archived lossless votes:

```bash
uv run --frozen --extra analysis python analyze_contact_eval.py \
  --verify-pr190-control
```

That deterministic rescore returns R-all `0.5335961341539802`. The fresh control
generation returned `0.535215598085612`, a `+0.001619463931631815` mismatch.
The gate passed because its declared absolute tolerance is 0.006. PR #190 did
not assign per-request TPU seeds, so decoded samples vary across runs. The p03
and p06 jobs used the same stochastic generation, vote, and metric pipeline as
the fresh control.

Within `rerun02-20260809`, the p03-aug final is `+0.0391170928910645` R-all
above the fresh control and p06-aug is `-0.0108086005791727` below it. The local
[`contact_eval_final_checkpoint_summary.csv`](../../data/contact_eval_final_checkpoint_summary.csv)
index preserves unrounded range values, the PR #190 reference and delta, run
identities, job names, and artifact prefixes. A
[public copy](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/contacts-v1-model-eval-exp199/derived/contact_eval_final_checkpoint_summary.csv)
lives beside the per-checkpoint artifacts.

Public derived artifacts:

- [canonical PR #190 exp117 control](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-model-eval-exp166/derived)
- [fresh exp117 control replicate](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-model-eval-exp199/replicates/rerun02-20260809/derived/prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4/step-35679)
- [exp199 p03-aug step 72,599](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-model-eval-exp199/replicates/rerun02-20260809/derived/prot-exp199-cv1-s01-m1-p03-aug-us-east1/step-72599)
- [exp199 p06-aug step 72,599](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-model-eval-exp199/replicates/rerun02-20260809/derived/prot-exp199-cv1-s01-m1-p06-aug-us-east1/step-72599)

The exp199 raw votes, timings, exact inputs, and manifests use exact run
identities:

```text
hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp199/
  replicates/<run-tag>/runs/<wandb-run-name>/step-<N>/
```

The canonical control raw votes remain under PR #190's
`contacts-v1-model-eval-exp166/scores/exp117-control-step-35679` prefix.

After another job completes, validate all 554 completion markers and compute
the standard contact metrics locally:

```bash
uv run --frozen --extra analysis python analyze_contact_eval.py \
  --checkpoint exp117-control-step35679 \
  --run-tag <unique-tag>
```

The finalizer writes compact per-protein metrics, summaries, timings, and
provenance to the experiment `data/` directory and publishes them under the
matching HF derived prefix. Raw sparse votes stay available, so score matrices
and later analyses can be rebuilt without another TPU run. Git retains only
manifests, aggregate summaries, and the consolidated index. Per-protein rows
and timing archives remain in HF.

## Step 26760 pilot record

The temporary Iris job
`/eczech/marinfold-exp199-step26760-eval-shell-01` was used with `iris task
exec` before unattended submission. Commands issued inside that live `v6e-4`
verified the locked environment, four TPU devices, GCS restore, HF export,
shardwise BF16 cast, tokenizer and Qwen3 geometry, TPU vLLM load, and one
100-rollout inference batch. The batch completed for `foldbench100/5sbj_A` with
all 100 generations reaching the stop token.

The complete 554-target evaluation ran as
[`/eczech/marinfold-exp199-s01-m1-p06-base-step26760-eval-retry01`](https://iris-dev.oa.dev/#/job/%2Feczech%2Fmarinfold-exp199-s01-m1-p06-base-step26760-eval-retry01).
The `retry01` suffix distinguishes it from an earlier container-build failure.
Both attempts targeted the same immutable result prefix, and completion markers
made a later retry skip finished parts. The job succeeded without a preemption
or task failure in 47 minutes 26 seconds. Inference generated 55,400 rollouts
for all 554 proteins in 2,509 seconds. Checkpoint preparation took 40 seconds,
and TPU vLLM loading and compilation took 195 seconds.

The finalized mean metrics are:

| Range | R-precision | AUC |
| --- | ---: | ---: |
| All | 0.461997 | 0.910293 |
| Short | 0.588240 | 0.956308 |
| Medium | 0.508059 | 0.943702 |
| Long | 0.404464 | 0.885805 |

The raw prefix contains 35 sparse-vote parts and 35 timing completion markers.
The public
[`derived/prot-exp199-cv1-s01-m1-p06-base/step-26760`](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-model-eval-exp199/derived/prot-exp199-cv1-s01-m1-p06-base/step-26760)
prefix contains the validated per-protein metric rows, aggregate summary,
timings, and provenance manifest. The finalizer validated 2,516,122 sparse vote
rows and produced 11,080 metric rows.
