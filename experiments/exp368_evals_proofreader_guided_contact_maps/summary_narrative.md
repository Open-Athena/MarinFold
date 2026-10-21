## Inference-time proofreader guidance

Can the released proofreader improve complete contact maps by changing sampling,
looking ahead, maintaining several prefixes, repairing errors, or combining
larger independent rollout pools?

Frozen generator: exp277 step 266344. Frozen proofreader: exp356 step 18000.
Implementation and GPU validation are in progress; no guided-policy result yet.

## Frozen models and experimental references

The generator is the exp277 step-266344 Qwen3 model. The proofreader freezes
its causal backbone and adds a four-layer, 512-wide bidirectional contact reader
with 22.18 million trainable parameters. It predicts each supplied contact's
correctness and the current rollout's recall. No model is retrained here.

The proofreader was trained on 50,000 experimental PDB chain structures.
References use the complete exp222 native-only pyconfind contact set: a 3 Å
side-chain threshold, minimum contact degree 0.001 and sequence separation ≥6.
The validation split has 1,552 structures in 287 related groups.

## What top ten percent means here

At each contact boundary, sample K candidate contacts from the constrained
generator. Score them with the proofreader after appending each candidate.
Keep the highest ceil(0.1 K) candidates and sample uniformly among survivors.
This keeps 2 of 16 or 7 of 64 sampled proposals, including possible repeats;
it does not enumerate the entire residue-pair universe.

Soft guidance weights the same proposals by q raised to a chosen power.
Lookahead adds future contacts before scoring, then commits only the first.
The generator's probability already enters through proposal sampling.

## Complete nested-pool pilot

On 32 validation structures, sum-q R-precision increases from 0.5792 with eight
rollouts to 0.5993 with 128. The paired gain is 0.0200, with 95% group-bootstrap
interval 0.0050–0.0329. Long-range R-precision rises from 0.5537 to 0.5806.

Candidate coverage grows from 0.7765 to 0.9509. At 128 rollouts, frequency alone
slightly exceeds sum-q: all-contact R-precision 0.6040, long-range 0.5864. Most
remaining error is ranking error despite the much larger candidate pool.

Confidence tie-breaking and predicted-F1 vote weighting were added as exploratory
validation tests. Their tiny gains over frequency have intervals including zero.

## Baseline calibration and coverage

On 128 ordinary rollouts per pilot structure, per-rollout predicted precision
has mean absolute error 0.0606 and predicted recall 0.0547. Errors average within
structures and then over the same 32 structures. Contact calibration curves
weight individual contacts; the two summaries answer different questions.

Mean trajectory overlap is 0.3473. Enlarging the pool exposes about 95% of native
contacts, but ranking only recovers about 60% at the native contact-count budget.
The remaining ranking gap motivates guidance and better aggregation; it does not
establish that either can close the gap. Guided-rollout calibration is pending.

## Practical single-batch baseline

Requesting each pool in one batch gives 0.6036 all-contact and 0.5894 long-range
R-precision with frequency ranking at 128 rollouts. Mean warm generation cost is
7.165 H100 seconds per map; no proofreader pass is needed for this aggregator.

Against eight rollouts with sum-q ranking, the paired all-contact gain is 0.0244
(95% group interval 0.0101–0.0373) and the long-range gain is 0.0357
(0.0154–0.0519). The 128-rollout pool contains 95.39% of native contacts.
The single-batch recipe is the practical cost baseline for guided search.

## Larger validation confirms the pool-size benefit

On all 1,375 structures in 255 validation groups outside the pilot, frequency
ranking improves from 0.5014 at eight rollouts to 0.5477 at 128. The paired gain
is 0.0463 (95% group-bootstrap interval 0.0403–0.0526). Long-range R-precision
rises from 0.4709 to 0.5215; its gain is 0.0506 (0.0430–0.0586).

The pool exposes 95.23% of native contacts. Mean warm generation time increases
from 2.431 to 6.599 H100 seconds. Both sides use frequency ranking; fresh
proofreader-weighted validation scoring remains pending. The long-range mean
covers 1,371 structures with nonempty long-range references.

## Benefit across lengths and measured latency

Increasing frequency-ranked pools from eight to 128 improves R-precision on
1,196 of 1,375 validation structures; 63 tie and 116 worsen. The median paired
gain is 0.0392. Descriptive bands of 32–127, 128–255, 256–511 and 512–1000
residues all have positive mean all-contact and long-range gains.

For 128 rollouts, median warm generation latency is 4.345 H100 seconds, the
95th percentile is 17.024 seconds and the maximum is 63.331 seconds. Long
proteins dominate the tail. Every plotted latency comes from its saved timing
record; model loading and artifact writes are excluded from warm inference.

## Exploratory length control

On the five short proteins completed by every original policy, top-10% filtering
raises per-rollout precision but emits fewer contacts and lowers recall. This
incomplete cohort motivates a stopping-rule ablation, not policy selection.

An added exploratory pilot compares ordinary and top-10%-of-16 sampling at
exactly L and 2L contacts, with EOS disabled for both. Four independent 2L
continuations yield nested L and 2L maps. The original 15-policy pilot remains
frozen. No new reserved-test scores have been inspected.

## Existing 1552-structure validation baseline

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
Ordinary generation is nearly complete for 1,375 structures from 255 validation
groups outside the pilot. New job starts are delayed by a regional controller outage. No new reserved-test scores have been computed or inspected.

## Numerical validation

All frozen backbone tensors match the generator exactly. CPU cache tests agree
with full forward passes. The first H100 smoke caught BF16 incremental/full-pass
drift of up to 0.02394 in one contact probability. FP32 cached search passes on
three real proteins, with maximum error 0.00000358. Final maps use the released
BF16 full-pass reader for comparison.

## Regional recovery status at 1730 UTC

East-region output writes failed after 16:55. All affected workers stopped, and
4,446 saved objects were copied to the GPU region. Source checksums and complete
destination readbacks verified every file. Original code, sharding and seeds
are preserved, with new results resuming from the verified regional snapshot.

The regional controller stopped responding after 17:14. Thirteen replacement
and smoke jobs remain queued. Existing validation-generation workers continue
writing outputs. These are infrastructure delays; guided-policy conclusions
remain pending. No shared cluster state has been changed.

## How to interpret the eventual comparison

R-precision evaluates the top R predictions, where R is the number of native
contacts used only for evaluation. Precision at L, L/2 and L/5 provides usable
sequence-length budgets. AP includes every eligible unproposed pair at score
zero. Contact labels enter only after sampling decisions are finished.

Pilot and larger-validation results select the final methods. The reserved test
was previously used for the proofreader release; new inference-policy test scores
have not been inspected. Generator pretraining decontamination is not established.
Search timings describe this implementation, including discarded work, and are
not hardware-independent lower bounds on the cost of these policy families.

## Execution resumed through the healthy East controller

East H100 capacity and object writes recovered while RNO's controller remained
unresponsive. Verified copies of the 14.2 MB guided snapshot and 404.6 MB complete
validation outputs now live in East. Pending RNO replacements were canceled
before new East jobs were submitted. Existing East weights keep large reads
colocated, with no new cross-region model staging.

The new fixed-length GPU smoke passed exact L and 2L contact counts, nested-prefix
checks, model hashes and FP32 cache errors below 9e-7. Four validation-scoring
jobs, four fixed-length shards and twelve original guided shards are submitted
through the healthy controller. No shared cluster restart was performed.
