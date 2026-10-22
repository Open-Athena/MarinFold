## Inference-time proofreader guidance

Can the released proofreader improve contact maps by filtering proposed contacts,
using future evidence, searching several prefixes, repairing errors, or combining
independent rollout pools?

Completed validation supports larger ordinary pools and a small benefit from
weighting votes by predicted rollout quality. Forced long guided rollouts expose
a substantial overconfidence failure. The original 15-policy pilot is complete and none of its guided methods improves
the measured ordinary-sampling frontier. The bounded weight refinement selected F1 squared; its new reserved-test
comparison is now in progress.

## Frozen models and experimental references

Generator: exp277 step 266344. Proofreader: exp356 step 18000, with a frozen
Qwen3 causal backbone and a four-layer, 512-wide bidirectional reader containing
22.18 million trainable parameters. It estimates contact correctness and rollout
recall. This experiment changes inference, not model weights.

Reader training used 50,000 experimental PDB chain structures. Reference contacts
inherit exp222 native-only pyconfind labels: 3 Å side-chain threshold, minimum
contact degree 0.001, sequence separation at least six. Complete contact sets
and the frozen contiguous-sequence preparation rules determine recall.

## Pilot, validation and reserved comparison

The pilot fixes 32 validation proteins from 32 related groups, spanning 33–937
residues. Representatives and length ranks were chosen without model-quality
selection. Four rollouts for each of 15 policies give 1,920 guided trajectories.

The larger validation panel excludes every pilot group: 1,375 structures in
255 groups, with 1,371 contributing long-range metrics. H100 and GB200 ordinary
pool experiments use this same biological cohort and separate runtime samples.
The reserved cohort contains 1,473 structures in 295 separate related groups.
Its new comparison is restricted to one frozen pair after validation selection.

## What top ten percent means

At each boundary, draw K complete candidate contacts from the grammar-constrained
generator. Score each hypothetical appended prefix with the reader. Uniformly
sample among the highest ceil(0.1 K) proposal slots: two of 16, or seven of 64.
Repeated proposals retain their sampled multiplicity. This does not enumerate
the entire residue-pair universe or guarantee 90% contact accuracy.

Soft guidance instead weights proposals by q to a fixed power. The generator's
probability already enters through proposal sampling. Committed pairs cannot
repeat; valid pairs have sequence separation at least six. The generator samples
EOS once per boundary, without intervention by the proofreader.

## Other search policies

Lookahead scores a candidate after two separate futures of eight or 32 contacts,
then commits only that candidate. Total-score selection lets a proposal change
the reader's judgments of all earlier contacts. Eight-particle search compares
prefix F1 estimates with a Jaccard diversity penalty after blocks of eight.

Repair identifies the lowest-confidence contact in the first half of a rollout,
rewinds to it, and regenerates the suffix. Ordinary and uniform-proposal controls
isolate guidance effects. All 15 original families finished before selection. One new configuration,
F1 weight raised to power 2, advances to larger validation using existing outputs.

## Metrics and compute accounting

Primary quality is all-contact R-precision, with long-range R-precision requiring
separation at least 24. R is the native contact count and is used only for
evaluation. Deployable budgets select L, L/2 or L/5 contacts from the ranked map.
Full-universe AP includes unproposed pairs at score zero; missing prediction
slots count as misses. No native label sets a sampling or ranking decision.

Paired 95% intervals resample related groups and retain equal structure weights.
GPU-stage timings include discarded proposals, lookahead and final scoring.
They exclude staging, loading, CPU aggregation, queueing and preempted work.
H100 and GB200 times remain separate. These are measured implementation costs,
not hardware-independent lower bounds on guidance cost or billed GPU-hours.

## Ordinary sampling: a strong practical baseline

On all 1,375 larger-validation structures, increasing ordinary frequency-ranked
pools from eight to 128 improves R-precision from 0.50137 to 0.54769. The paired
gain is 0.0463 (95% group interval 0.0403–0.0526); long-range R-precision rises
from 0.47089 to 0.52153. Candidate coverage reaches 95.23%.

Mean warm generation cost rises from 2.431 to 6.599 H100 seconds per map.
GB200 replicates the quality result: 0.50086 to 0.54779, at 1.196 and 3.054
seconds. The runtime differs as well as hardware, so this is not a controlled
measurement of GPU speed alone. Single-batch ordinary pools are the preferred
practical cost baseline; staged nested pools remain a separate experiment.

## Contact confidence helps small pools

On the full H100 validation panel, sum-q ranking improves eight-rollout
R-precision from 0.50137 to 0.51737: paired gain 0.01601 (95% interval
0.01239–0.02033). Long-range gain is 0.01710 (0.01240–0.02223), at 1.061
times the measured warm cost.

At 128 rollouts, sum q changes R-precision by −0.00008 (−0.00386–0.00424).
It replaces 8.70% of top-R slots, but correct contacts gained and lost nearly
cancel; exact swap counts reproduce every protein's metric change. Mean
confidence alone performs poorly because it discards the vote-count signal.

## Rollout-quality weighting gives a smaller validation gain

At 128 rollouts, each trajectory can weight its eligible contact votes by
predicted F1, computed from its predicted precision and recall without labels.
On H100, this gives R-precision 0.55195 versus 0.54769 for frequency:
paired gain 0.00426 (95% interval 0.00173–0.00763). Long-range gain is 0.00412
(0.00129–0.00781). Mean warm time is 8.921 versus 6.599 seconds.

GB200 replicates the gain on the same proteins: +0.00459 (0.00207–0.00790)
overall and +0.00453 (0.00164–0.00813) long-range, at 4.622 versus 3.054
seconds. The rule was added after the ordinary-pool pilot. These unadjusted
validation intervals do not replace the pending reserved-test confirmation.

## Larger pools saturate the top-R map

On the complete 32-protein H100 pilot, frequency R-precision is 0.60365,
0.60276 and 0.60299 at 128, 512 and 2,048 rollouts. The 2,048-minus-128
difference is −0.00066 (95% interval −0.00489–0.00352), at 15.35 times the
warm cost. Coverage reaches 99.21%, and AP improves by 0.01334.

The GB200 frequency and scored-pool checks agree on the plateau. With
F1-weighted votes, 2,048 versus 128 changes R-precision by +0.00002
(−0.00463–0.00500), at 20.83 times the cost. AP improves by 0.00897
(0.00480–0.01397). More candidates improve recovery below the top-R cutoff;
these experiments do not establish a better top-R map.

## Forced length: better individual precision, weaker maps

An adaptive control disables EOS for ordinary and top-10%-of-16 generation,
evaluating exactly L and 2L contacts. All 256 continuations and their 512 nested
prefix maps are complete on the pilot. The same continuation supplies both
budgets; they are not independent observations.

At L, filtering raises mean individual precision from 0.43345 to 0.49286,
but four-rollout map R-precision is 0.57084 versus 0.57152 for control,
at 3.25 times the cost. At 2L it falls from 0.56116 to 0.53276:
difference −0.02839 (95% interval −0.04961 to −0.00827), at 4.17 times the cost.
Guided 2L precision is 0.28286 while the reader predicts 0.52488.

## Later contacts can corrupt earlier judgments

In guided fixed-length continuations, the new second L contacts have actual
precision 0.07286 while predicted correctness averages 0.50726. Control values
are 0.12232 and 0.24450. Both sets use all 32 proteins and four trajectories.

The identical first L contacts are also rescored too confidently after continuing:
predicted correctness rises from 0.49471 to 0.54250 while actual precision
remains 0.49286. Brier score worsens by 0.02068 (95% interval 0.01511–0.02654)
and AUROC falls by 0.04665 (−0.06241 to −0.03346). Later evidence is unreliable
under this forced guided distribution. Distribution shift and score exploitation
are consistent with these results, but a unique causal mechanism is not identified.

## Bounded refinement of rollout weights

A small adaptive follow-up fixed F1 exponents 1, 2, 4 and 8 before computing
their outcomes on the complete pilot. It reuses identical 128-rollout pools
and reader outputs without new GPU inference. Original frequency and power-1
results reproduce exactly for every input.

Every paired all-contact and long-range interval versus exponent 1 includes
zero. Power 2 changes all-contact R-precision by +0.00024 (−0.00204–0.00259)
on H100 and +0.00137 (−0.00181–0.00584) on GB200. This pilot does not establish
a stronger weighting rule. It does not select a final policy before the original
guided pilot completes.

## Numerical validation and recoverability

All frozen backbone tensors match the generator. H100 BF16 incremental scoring
failed equivalence checks, so guidance uses verified FP32 caches and final maps
use released BF16 full-forward scoring. GB200 full-forward replay passes with
PyTorch 2.10.0+cu128; cached guidance fails the original tolerance and is excluded.
No tolerance was relaxed. Eleven CPU tests cover numerical and analysis behavior.

Owned jobs use the existing CoreWeave batch queues. Source bundles, models,
plans and seeds are pinned; complete trajectories commit durably to S3.
Controller outages and preemptions required recovery. An early repeated-staging
incident is disclosed in the execution appendix. Later regional mirrors enforce
durable transfer reservations. No shared cluster configuration was changed.

## Limits and remaining confirmation

The original guided cohort and bounded validation refinement are complete.
The frozen reserved comparison contains exactly two configurations and requires all 1,473
structures to finish before metric analysis. The reserved set was previously
used for the proofreader release; it is not an untouched external benchmark.
Generator pretraining decontamination is not established.

Adaptive additions and unadjusted validation intervals are identified explicitly.
Weighted votes are ranking scores, not calibrated pair probabilities. A positive
local proposal-ranking effect need not improve a complete map, and good
calibration on ordinary rollouts need not survive guided distribution shift.
The guided methods do not improve the measured quality/cost frontier. The
remaining comparison will check the selected aggregation rule on reserved data.

## Complete guided pilot: cleaner trajectories, poorer maps

All 1,920 trajectories are saved. Top-10%-of-16 filtering raises mean individual
precision from 0.48287 to 0.71445, but halves emitted contacts per residue
(0.89804 to 0.46402) and lowers recall from 0.49001 to 0.40480. Four-rollout
sum-q map R-precision falls by 0.05164 (95% interval −0.07400 to −0.02909).

The best sum-q guided method is soft exponent-1 weighting: R-precision 0.56773
at 77.076 warm H100 seconds per map. Ordinary frequency ranking at 128 rollouts
gives 0.60365 at 7.165 seconds. Thirty-two-contact lookahead gives 0.49875
at 3,117.978 seconds. No guided method, across all seven aggregators, improves
the measured ordinary-pool quality/cost frontier.

## Useful local ranking is insufficient

With top-10%-of-16 filtering, within-prefix candidate AUC is 0.77781 and the
chosen contact is 0.26803 more likely to be correct than an average sampled
proposal. Thus the reader is useful at the local decision, while reduced contact
coverage and shorter rollouts still harm the final map.

Mean immediate confidence is 0.68368 versus actual selected-contact accuracy
0.72061 under ordinary stopping. This underconfidence differs from the severe
overconfidence after forced 2L continuation. Diagnostics use the first four
decisions and every eighth thereafter; they compare each policy's own prefixes
and do not establish a causal decomposition of the final-map loss.

## Validation refinement selects squared F1 votes

Only power 2 advanced to the larger validation panel after the guided pilot.
Its promotion rule was committed before these new metrics. Against ordinary
F1 votes, squared F1 improves R-precision by 0.00239 (95% interval
0.00022–0.00508) on H100 and 0.00279 (0.00076–0.00523) on GB200.
Long-range effects are positive on both, satisfying every promotion condition.

The final comparison was frozen and committed before new test generation:
128 ordinary rollouts, frequency versus F1-squared votes, on all 1,473 reserved
proteins in 295 groups. Generation and full-forward scoring use the verified
GB200 runtime. No test metric analysis occurs before the complete cohort is
saved, and no additional configurations will be tried after those results.
