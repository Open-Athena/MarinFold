## Inference-time proofreader guidance

Can the released proofreader improve complete contact maps by changing sampling,
looking ahead, maintaining several prefixes, repairing errors, or combining
larger independent rollout pools?

Frozen generator: exp277 step 266344. Frozen proofreader: exp356 step 18000.
Implementation and GPU validation are in progress; no guided-policy result yet.

## Complete ordinary-pool pilot

On 32 validation structures, sum-q R-precision increases from 0.5792 with eight
rollouts to 0.5993 with 128. The paired gain is 0.0200, with 95% group-bootstrap
interval 0.0050–0.0329. Long-range R-precision rises from 0.5537 to 0.5806.

Candidate coverage grows from 0.7765 to 0.9509. At 128 rollouts, frequency alone
slightly exceeds sum-q: all-contact R-precision 0.6040, long-range 0.5864. Most
remaining error is ranking error despite the much larger candidate pool.

Confidence tie-breaking and predicted-F1 vote weighting were added as exploratory
validation tests. Their tiny gains over frequency have intervals including zero.

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

## Execution recovery

An eight-H100 RNO worker was preempted repeatedly. Iris failure-retry limits do
not limit preemption retries, so the worker repeated its shared model transfer
and lost local progress. It was canceled on detection; the incident is recorded.

Recovery uses original source snapshots and durable S3 trajectories. After East
capacity remained full, a durable 6.014 GB RNO2A input mirror was completed with
per-chunk reservations and a restart-safe 7 GB cap. Regional H100 smoke tests
verify identical model and target hashes. The complete guided comparison is pending.
Ordinary baselines are queued for 1,375 structures from 255 validation groups
outside the pilot. No new reserved-test scores have been computed or inspected.

## Numerical validation

All frozen backbone tensors match the generator exactly. CPU cache tests agree
with full forward passes. The first H100 smoke caught BF16 incremental/full-pass
drift of up to 0.02394 in one contact probability. FP32 cached search passes on
three real proteins, with maximum error 0.00000358. Final maps use the released
BF16 full-pass reader for comparison.
