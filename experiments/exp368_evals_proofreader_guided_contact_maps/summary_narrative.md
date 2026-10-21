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
Jobs use co-located CoreWeave storage and resumable per-rollout artifacts.
H100 and GB200 execution phases retain separate timing provenance.

## Execution and recoverability

A repeatedly preempted shared worker was canceled after repeated model staging
exceeded its intended transfer budget. The incident is recorded. Recovery jobs
use immutable source bundles, per-trajectory commit markers and durable S3.
Regional model mirrors use persistent reservations with a 7 GB cap each.

The complete H100 ordinary baselines are saved. H100 guided and fixed-length
runs were interrupted again at 18:15 UTC, and both H100 controllers stopped
reporting. No shared controller, priority or other user's workload was changed.
The separate GB200 ordinary phase passed full-forward scoring checks. Native
H100 jobs now use the existing Kueue batch queue despite the Iris outage.

## Numerical validation

All frozen backbone tensors match the generator exactly. CPU cache tests agree
with full forward passes. The first H100 smoke caught BF16 incremental/full-pass
drift of up to 0.02394 in one contact probability. FP32 cached search passes on
three real proteins, with maximum error 0.00000358. Final maps use the released
BF16 full-pass reader for comparison.

## GB200 verification and its limits

Tiny-model cache checks and production ordinary generation passed on GB200.
The production proofreader's CUDA 13 replay failed: the largest BF16 contact-score
difference from saved H100 predictions was 0.0800, against a 0.01 guard. A real
prefix also exceeded the unchanged 1e-5 FP32 cache-equivalence threshold.

Forcing math attention did not meet the BF16 guard; memory-efficient attention
was unsupported for this cache path. Matching CUDA 12.8 passed the BF16 replay:
two batches match exactly and the third has maximum difference 0.001523. One
cache test still fails, so cached guidance remains pending. Ordinary generation
and full-forward scoring can proceed. Cost plots reject mixtures of GPU models.

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

## Stronger comparisons at larger compute budgets

A separate GB200 ordinary-generation phase requests 512 and 2,048 independent
rollouts per pilot structure. These sizes were frozen before complete guided
results were available. Observed lookahead costs already greatly exceeded the
128-rollout baseline, motivating a wider ordinary-sampling cost curve.

Frequency ranking needs no proofreader pass. These are measured compute
envelopes, not exact budget matching. The same 32 validation structures are used,
and the reserved test remains untouched for new policy metrics.

## Reranking helps the small pilot pools

Replacing frequency by sum q changes 15.37% of top-R slots with eight rollouts
and improves R-precision by 0.0163 (paired 95% interval 0.0090–0.0248).
With 128 rollouts, it changes 7.62% of slots; the net effect is −0.0031
(−0.0099–0.0039), which does not establish a difference.

The diagnostic counts true contacts entering and leaving the map. Those counts
exactly reproduce the observed metric difference for every protein. All 32 pilot
structures are included; these results do not replace the larger validation test.

## H100 execution restored without a controller restart

Kubernetes showed 320 unallocated H100s and a healthy existing batch queue.
Native jobs were admitted through cw-rno2a-lq / iris-cq at the original batch
priority. The real-model smoke passed, with maximum cache error 8.94e-7.
All 22 stranded East replacements were canceled.

The saved guided campaign now resumes on 48 H100 workers. Original model,
source, plans, per-trajectory seeds and saved predictions remain unchanged;
only execution partitioning changes. Four fixed-length workers also resumed.
Native job IDs and exact specifications are retained separately from Iris IDs.

## Separate ordinary-pool replication on GB200

All 32 pilot proteins are complete for independent pools of 8, 32 and 128.
Eight-rollout sum q gives all-contact R-precision 0.5783 and long-range 0.5591.
At 128 rollouts, frequency gives 0.6025 and 0.5860; sum q gives 0.5995 and 0.5846.

Mean warm times are 1.511 seconds for eight-rollout sum q, 3.471 seconds for
128-rollout frequency, and 5.204 seconds for 128-rollout sum q. These use
vLLM 0.14.1 rather than the H100 baseline's 0.9.2; this is a separate runtime
cohort and not a controlled measurement of hardware speed alone.

## Larger pools reach a frequency-ranking plateau

The complete GB200 pilot gives frequency R-precision 0.60248, 0.60319 and
0.60257 at 128, 512 and 2,048 rollouts. Native-contact coverage rises from
95.20% to 98.60% and 99.33%, while long-range R-precision is essentially flat.

The paired 2,048-minus-128 all-contact difference is +0.00009 (95% interval
−0.00426 to 0.00503); long-range difference is −0.00148 (−0.00687 to 0.00411).
Mean warm cost rises from 3.471 to 82.198 GB200 seconds, a factor of 23.68.
This does not establish an accuracy gain. Proofreader ranking at 2,048 is untested.

## Larger validation replication on GB200

All 1,375 structures from 255 groups are complete for frequency ranking.
Eight versus 128 rollouts gives R-precision 0.50086 versus 0.54779; the paired
gain is 0.04693 (95% interval 0.04052 to 0.05353). Long-range R-precision rises
from 0.47099 to 0.52194, a gain of 0.05096 (0.04338 to 0.05915).

Mean warm cost is 1.196 versus 3.054 GB200 seconds. This separate runtime
replicates the H100 frequency result. Full-forward proofreader scoring is still
running. No new reserved-test scores have been computed or inspected.
