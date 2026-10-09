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

A local RTX A5000 smoke passed with the exact published checkpoint (all file SHA256 digests checked): 1,000 capped trajectories, 100 full trajectories with zero unfinished samples, 11 symmetric vote matrices, and 220 metric rows. It is excluded from the production comparison.

H100 batch capacity was unavailable, so production moved to one available GB200 on `cw-us-east-08a`, using the ARM64 image `vllm/vllm-openai:v0.11.0`. Job `/bizon/exp354-gb-v1-20261009-production-00` reads the existing 5.89 GB CoreWeave checkpoint once. The two queued H100 smoke attempts were cancelled before model execution. The GPU job writes to `s3://marin-us-east-02a/MarinFold/exp354_evals_short_rollout_consensus/gb-v1-20261009/production/`; a CPU exporter publishes completed artifacts to `hf://buckets/open-athena/MarinFold/data/exp354-short-rollout-consensus/gb-v1-20261009/production/`.

Submission manifests under `data/` record job IDs, payload digests, and placement. No production accuracy result is available yet.

## Conclusion

Pending the complete paired evaluation; no inference from a smoke protein.
