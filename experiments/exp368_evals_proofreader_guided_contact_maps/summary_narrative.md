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
This does not establish an accuracy gain. The subsequent proofreading check is reported below.

## Larger validation replication on GB200

All 1,375 structures from 255 groups are complete for frequency ranking.
Eight versus 128 rollouts gives R-precision 0.50086 versus 0.54779; the paired
gain is 0.04693 (95% interval 0.04052 to 0.05353). Long-range R-precision rises
from 0.47099 to 0.52194, a gain of 0.05096 (0.04338 to 0.05915).

Mean warm cost is 1.196 versus 3.054 GB200 seconds. This separate runtime
replicates the H100 frequency result. Full-forward proofreader scoring is also
complete. No new reserved-test scores have been computed or inspected.

## Complete H100 validation: confidence helps small pools

The released reader has scored all 2,750 pools from 1,375 validation structures.
At eight rollouts, sum q improves R-precision from 0.50137 to 0.51737:
paired gain 0.01601 (95% interval 0.01239 to 0.02033). Long-range gain is
0.01710 (0.01240 to 0.02223), at 1.061 times the measured warm inference cost.

At 128 rollouts, sum q and frequency are effectively tied in this comparison:
all-contact difference −0.00008 (−0.00386 to 0.00424), long-range +0.00089
(−0.00371 to 0.00599). Mean confidence alone loses the useful vote-count signal.

## Rollout quality weighting gives a small validation gain

With 128 rollouts, weighting each trajectory's votes by its predicted F1 gives
R-precision 0.55195 and long-range R-precision 0.52565. Compared with frequency,
the paired gains are 0.00426 (95% interval 0.00173 to 0.00763) and 0.00412
(0.00129 to 0.00781). Mean warm H100 time is 8.921 versus 6.599 seconds.

The weight uses predicted precision and recall only. This aggregation rule was
added after the ordinary-pool pilot; intervals are not adjusted for method selection. It is a
promising validation result, not a reserved-test confirmation. Guided-policy
selection still waits for the complete pilot.

## Calibration and contact swaps explain the aggregation pattern

On the full H100 panel, sum q replaces 18.29% of top-R slots at eight rollouts
and 8.70% at 128. Correct contacts gained minus correct contacts lost exactly
reproduce each protein's R-precision change. At 128 the two nearly cancel.

The 128-rollout precision and recall prediction MAEs are 0.05238 and 0.05768,
averaged within each structure then across structures. Mean predicted recall is
0.38840 versus observed 0.40864. Good pooled contact calibration does not by
itself guarantee accurate rollout recall or improved map ranking.

## GB200 replicates the rollout-F1 weighting gain

All 2,750 GB200 validation pools are scored. At 128 rollouts, F1-weighted votes
give R-precision 0.55239 and long-range R-precision 0.52647. Paired gains over
frequency are 0.00459 (95% interval 0.00207 to 0.00790) and 0.00453
(0.00164 to 0.00813). Sum q again has no established advantage at this pool size.

Mean warm generation-plus-proofreading time is 4.622 GB200 seconds, versus
3.054 for frequency alone. These are the same validation proteins as the H100
phase, so this is a runtime/stochastic replication, not a new biological cohort.

## Exploratory scoring of the larger existing pools: complete

At 20:15 UTC, the frequency plateau and the 128-rollout F1-weighting gains
motivated one further validation check. All 64 existing 512- and 2,048-rollout pools on the 32 pilot proteins have now been scored. No new rollouts or model
updates are involved, and no new reserved-test scores have been inspected.

The primary 2,048-minus-128 F1-weighted R-precision difference is +0.00002
(95% interval −0.00463 to 0.00500), at 20.83 times the warm GPU cost.
Secondary average precision improves by 0.00897 (0.00480 to 0.01397).
All seven aggregators are reported. This adaptive extension was documented
before scoring and is not a prespecified confirmation.

## Fixed-length filtering: better individual precision, weaker maps

All 512 L/2L maps from 256 continuations are complete on the same 32 proteins.
At L contacts, top-10% filtering raises mean individual-rollout precision from
0.43345 to 0.49286, but four-rollout map R-precision is 0.57084 versus 0.57152
for control, at 3.25 times the cost. The paired interval includes zero.

At 2L, four-rollout R-precision falls from 0.56116 to 0.53276: difference
−0.02839 (95% interval −0.04961 to −0.00827), at 4.17 times the cost.
The reader predicts precision 0.52488 on guided 2L rollouts; actual precision
is 0.28286. This is consistent with distribution shift and score exploitation.

## Larger pools also plateau on the H100 baseline

All 64 H100 pools at 512 and 2,048 rollouts are complete. Frequency R-precision
is 0.60276 and 0.60299, versus 0.60365 at 128. The 2,048-minus-128 difference
is −0.00066 (95% interval −0.00489 to 0.00352), with 15.35 times the warm cost.
Long-range difference is −0.00241 (−0.00656 to 0.00191).

Full-universe average precision improves by 0.01334 (0.00847 to 0.01985), and
native-contact coverage reaches 99.21%. The same distinction appears on GB200:
additional sampling improves coverage and ranking below the top-R cutoff,
without establishing a better top-R map.

## Forced continuation also corrupts retrospective judgments

For top-10% guidance, the second L contacts have actual precision 0.07286,
while mean predicted correctness is 0.50726. The corresponding control values
are 0.12232 and 0.24450. These are complete-cohort descriptive diagnostics.

The identical first L contacts are also rescored too confidently after continuing:
mean predicted correctness rises from 0.49471 to 0.54250, with actual precision
fixed at 0.49286. Early-contact Brier score worsens by 0.02068 (95% paired
interval 0.01511 to 0.02654), and AUROC falls by 0.04665 (−0.06241 to −0.03346).
Later evidence is not reliably helpful under this forced guided distribution.
