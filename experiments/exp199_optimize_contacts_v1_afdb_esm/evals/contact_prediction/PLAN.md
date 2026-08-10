# Exp199 contact-evaluation plan

## Scope

Evaluate final checkpoints from each completed exp199 run against the exp117
control used by PR #190. The initial set contains
`prot-exp199-cv1-s01-m1-p03-aug-us-east1` and
`prot-exp199-cv1-s01-m1-p06-aug-us-east1`. Keep the same catalog and artifact
layout ready for new runs as they finish and for other permanent p03 steps.

## Fixed inputs

- Candidate run: nine permanent native checkpoints at steps 8,920, 17,840,
  26,760, 35,680, 44,600, 53,520, 62,440, 71,360, and the forced final step
  72,599 under the run's read-only `marin-us-east1` checkpoint prefix.
- Initial candidate selection: step 72,599.
- Second completed run: p06-aug forced final step 72,599 under its read-only
  `marin-us-east1` checkpoint prefix.
- Control: `open-athena/marinfold-exp117` step 35,679 from the 1.5B, 16-epoch
  run, pinned at HF revision `f07366720aee0f62d7629ad3bd91dbcacc80ddef`.
- Targets and ground truth: the same checksummed 554-protein exp169 inputs used
  by PR #190.
- Recipe: 100 resampled rollouts, temperature 1.0, top-p 0.95, top-k disabled,
  and the established all, short, medium, and long contact metrics.

## Control verification

PR #190 published 35 lossless vote parts for the selected control. On 2026-08-09,
the current exp199 `metric_rows` implementation rescored those parts one at a
time. It recovered 554 all-range R values with sum `295.61225832130503` and mean
`0.5335961341539802`. The result is the full-precision source for PR #190's
displayed `0.5336`.

`analyze_contact_eval.py --verify-pr190-control` now reads PR #190's archived
lossless votes and applies the current scorer. It reproduces R-all
`0.5335961341539802` exactly, along with R-short `0.6284087546857455`, R-medium
`0.5853640942736399`, and R-long `0.4825581312405502`. These are the canonical
control values used for exp199 comparisons.

Fresh control runs use the same HF checkpoint subtree, target hashes,
generation settings, TPU vLLM versions, and metric implementation. PR #190 did
not assign per-request TPU seeds, so each fresh decode is a stochastic
replicate. The first produced `0.5347972614575084`, a delta of
`+0.001201127303528171`. Its paired standard error against PR #190's 554 values
is `0.0012401813924336504`. The isolated `rerun02-20260809` control produced
`0.535215598085612`, a delta of `+0.001619463931631815`. Both passed the 0.006
gate, while neither exactly reproduced the archived generation.

## Placement and storage

- Submit every job to `marin-dev` with Iris `--user eczech`.
- Run exp199 checkpoints in `us-east1` and the exp117 control in `europe-west4`.
- Request one `v6e-4` per independent checkpoint job.
- Keep prepared weights only in worker-local `/app/scratch`.
- Write raw votes, timings, exact inputs, and manifests below
  `hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp199/replicates/<run-tag>/runs/`.
- Write derived rows and summaries under the matching
  `replicates/<run-tag>/derived/` path and keep compact copies in the experiment
  `data/` directory.
- Track only manifests, aggregate summaries, and the consolidated index in Git.
  Per-protein rows and timing archives remain in HF and are ignored locally.
- Never modify, move, or delete a source checkpoint in GCS.

## Execution order

1. Run the exp117 control and each completed exp199 final checkpoint as separate,
   concurrent jobs.
2. Finalize every complete 554-protein output independently.
3. Report the control's unrounded all-range R-precision and reference-gate result.
4. Compare exp199 runs only after their outputs validate.
5. Add another finished exp199 trial with `exp199_final_checkpoint(<trial>)`.
6. Add other p03 permanent steps by selecting their existing catalog keys.

## Completion record

The isolated `rerun02-20260809` control, p03 final, and p06-aug final jobs ran
concurrently on 2026-08-09. All three completed the 554-protein,
55,400-rollout evaluation without a task failure or preemption. Each has 35 raw
vote parts and 35 timing parts in its own immutable run prefix. The finalizer
validated and published all three. Their manifests record the same MarinFold
revision, target hashes, rollout count, sampling settings, and tensor
parallelism.

| Checkpoint | Job duration | R-all | Gate |
| --- | ---: | ---: | --- |
| exp117 canonical PR #190 scores | archived | 0.5335961341539802 | exactly reproduced |
| exp117 fresh rerun02 step 35,679 | 45m 54s | 0.535215598085612 | passed tolerance; not exact |
| p03-aug rerun02 step 72,599 | 43m 57s | 0.5743326909766765 | complete |
| p06-aug rerun02 step 72,599 | 44m 53s | 0.5244069975064393 | complete |

Future final runs follow the same path. Add one `exp199_final_checkpoint()`
catalog entry, submit its independent region-local job, finalize it, and append
the validated row to `data/contact_eval_final_checkpoint_summary.csv`.
