## Inference-time proofreader guidance

Can the released proofreader improve complete contact maps by changing sampling,
looking ahead, maintaining several prefixes, repairing errors, or combining
larger independent rollout pools?

Frozen generator: exp277 step 266344. Frozen proofreader: exp356 step 18000.
Implementation and GPU validation are in progress; no guided-policy result yet.

## First result: aggregation baseline

On existing validation rollouts, sum q remains strongest among five simple
aggregators: all-contact R-precision 0.5315; long-range R-precision 0.5046.
Frequency gives 0.5167 and 0.4877 respectively. Mean q alone gives 0.4895.

With up to eight rollouts, the pool contains about 75% of native contacts.
This separates missing-contact generation from ranking errors. Oracle pool
coverage is an upper bound, not the performance of an inference method.

## Frozen pilot and execution

32 validation proteins from distinct homology groups span 33–937 residues.
Compare top-percentile gates, soft weighting, lookahead, population search and
one-pass repair, plus ordinary vLLM pools of 8, 32 and 128 rollouts.

Measure full-universe AP, all/long-range R-precision and complete search cost.
Select policies on validation, then lock a reserved-test confirmation plan.
Jobs use co-located CoreWeave H100s and resumable per-rollout artifacts.

## Numerical validation

All frozen backbone tensors match the generator exactly. CPU cache tests agree
with full forward passes. The first H100 smoke caught BF16 incremental/full-pass
drift of up to 0.02394 in one contact probability. FP32 cached search passes on
three real proteins, with maximum error 0.00000358. Final maps use the released
BF16 full-pass reader for comparison.
