---
marinfold_experiment:
  issue: 354
  title: 'exp: 1000 short-rollout consensus versus 100 full rollouts on eval-val'
  kind: evals
  branch: main
---

# exp: 1000 short-rollout consensus versus 100 full rollouts on eval-val

**Issue:** [#354](https://github.com/Open-Athena/MarinFold/issues/354) · **Kind:** `evals` · **Branch:** `main`

## Question

Does the current default contacts-v1 model improve on the 97 eval-val proteins when voting over 1,000 short rollouts capped at 5, 10, 20, floor(L/5), or floor(L/2) emitted contacts, instead of 100 full rollouts?

## Hypothesis

More independently resampled prefixes may improve consensus by reducing within-rollout redundancy, but very small contact budgets may lose useful conditional structure.

## Background

Current registry default: contacts-v1-exp277-m2-p06-full-epoch-1.5B, step 266344 (#277). Evaluation uses #245 eval-val, #82 rollout/resample settings, and #89 metrics.

## Approach

Run only eval-val (97); do not evaluate eval-test. Use temperature 1, top-p .95, top-k disabled, fresh deterministic document realizations per rollout. Generate 1,000 trajectories to max(20, floor(L/2)); extract exact contact-count prefixes for each requested cap, respecting early EOS and live-contact semantics. Score 100-prefix subsets as controls and a fresh 100-full-rollout baseline. Record per-input wall time and hardware, generated-token budgets, exact provenance, and paired bootstrap deltas for all/long R-precision. Reuse the existing checkpoint in CoreWeave S3. Publish reproducible artifacts and small CSVs/plots.

## Success criteria

A gain of at least .005 all-range R-precision, with paired protein bootstrap uncertainty reported. Treat this sweep as exploratory; report every cap and all/long metrics, precision at L/L2/L5, and AUC diagnostics.

## Results

The contact-count and live-vote parser passes three boundary-case tests. Immutable #245 inputs have been verified by SHA256 and filtered to exactly 97 eval-val proteins (length 38–761). The sweep requires at most 39.426 million short-rollout output tokens before early EOS, plus the full baseline.

GPU smoke tests are in progress. Submission manifests under `data/` record job IDs, payload digests, the exact checkpoint, and execution region. No accuracy result is available yet.

## Conclusion

Pending the complete paired evaluation; no inference from a smoke protein.
