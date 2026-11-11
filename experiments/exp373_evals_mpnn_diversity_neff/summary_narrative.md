# ProteinMPNN-only diversity and Neff dependence

## Four checkpoints, matched evaluation

The two-epoch models are now included: exp364 5e-4 and 1e-3, both at step 373768, alongside the original one-epoch 5e-4 model (step 186884) and full-data default (step 266344). The two-epoch 5e-4 model is the primary follow-up from the checkpoint handoff.

We added 440,800 rollouts, bringing the total to 881,600. Each model has 1,000 fixed-prompt samples at temperature 1, 300 at 0.7 and 1.3, and 100 randomized prompts at each temperature. All original predictions are unchanged.

Primary cohort: 97 natural eval-val proteins. Secondary cohort: 19 designs, kept separate. Eval-test was not scored. Prompts, seeds, truth, parser, runtime and scoring are matched. Intervals use 10,000 paired protein-bootstrap draws; secondary comparisons are not multiplicity-adjusted.

## The stronger checkpoint narrows the gap

Natural-protein randomized-100 R-precision, in default / one-epoch / two-epoch 5e-4 / two-epoch 1e-3 order: 0.5540 / 0.4628 / 0.5211 / 0.4790.

Two-epoch 5e-4 improves over one epoch by +0.0583 [0.0425, 0.0763]. Its remaining deficit versus default is -0.0329 [-0.0489, -0.0177]. Two-epoch 1e-3 trails 5e-4 by 0.0421 [0.0277, 0.0575].

Fixed-300 Jaccard distance is 0.7083 / 0.8130 / 0.7576 / 0.8022. The stronger 5e-4 model is less diverse than one epoch, but still more diverse than default. Its excess survives count matching (+0.0530), randomized prompts (+0.0455), unrestricted pairs and all three seed blocks. These are distributional differences, not evidence of physical conformational states.

## Discovery through 1,000: a smaller error cost

At k=1,000, true-contact coverage is 98.51% / 99.01% / 99.22% / 99.02%, in the same four-model order. Distinct false contacts per protein are 8,011 / 12,291 / 9,625 / 12,192.

Two-epoch 5e-4 gains 0.71 percentage points [0.34, 1.15] of coverage over default: about 2 additional true contacts alongside 1,614 additional false contacts. Its false-contact count is 20% higher than default, down from the one-epoch model's 53% excess.

From 300 to 1,000, two-epoch 5e-4 adds 4.5 true and 2,679 false contacts per protein. Ensemble R improves only +0.0024 [0.0002, 0.0050], reaching 0.5203 versus default's 0.5414. Other models' small ensemble gains span zero. Extra diversity still mostly adds benchmark errors, with a substantially smaller cost for the stronger checkpoint.

## Oracle best@k and mean individual F1

For each protein, oracle best@k selects the highest individual contact F1 among the first k seeded samples. Selection uses ground truth; it is an upper bound, not an inference-time selector. All- and long-range oracles select independently. Dashed lines average individual F1 over 1,000 samples within each protein, then across proteins.

Natural-protein mean individual F1: 0.4115 / 0.3020 / 0.3718 / 0.3197. Oracle best@1,000: 0.5607 / 0.4628 / 0.5266 / 0.4798. Two-epoch 5e-4 gains +0.0639 [0.0467, 0.0816] in oracle F1 over one epoch, but remains below default by 0.0341 [0.0214, 0.0483].

Every model gains oracle F1 after 300: +0.0161 / +0.0218 / +0.0152 / +0.0197, all with positive intervals. Ordinary mean F1 is stable over the extension. Long-range results agree. Bands are pointwise protein-bootstrap intervals, not simultaneous bands or rollout-level uncertainty.

## Neff: the earlier tier result does not generalize

Neff is redundancy-weighted depth at 80% identity. Fixed tiers contain 4, 10, 28 and 55 natural proteins. No model receives an MSA at inference.

Two-epoch 5e-4 has a slightly higher low-tier point estimate than default (0.2829 versus 0.2799), but the paired difference is inconclusive: +0.0031 [-0.0921, 0.0989]. Its high-minus-low drop difference is -0.0249 [-0.1229, 0.0735]. The one-epoch model's smaller extreme-tier drop therefore does not carry over clearly to the stronger checkpoint.

Two-epoch 5e-4's continuous interaction is -0.0028 [-0.0233, 0.0212] per Neff decade, also inconclusive after length/viral adjustment and viral exclusion. Two-epoch 1e-3 has a positive long-range interaction in the full cohort, the opposite of flatter dependence. No natural protein has raw MSA depth below 10. Four lowest-Neff proteins limit generalization.

## Dependencies and sequence-similarity limits

All four models have broader pairwise-distance distributions than shuffles preserving every sample size and contact frequency. Excess Jaccard SD is 0.0197 / 0.0169 / 0.0186 / 0.0182. Neither new model differs clearly from default in excess Jaccard or overlap SD. Half/full-chain differences are small relative to the excess; they are not proof of exact uniform mixing.

The contact maps retain the same eight proteins selected by Neff rank before inference. Every model uses the first 300 samples and the same frequency scale.

All redesign models use the same 162.6-million-document corpus. Historical native AFDB/ESM-Atlas searches remain available, but no complete redesign-corpus sequence search or matched KNN null is available. Removing native sequences retains backbone ancestry. These results cannot identify reduced retrieval or memorization as the mechanism.

## Designs and independent replication check

On 19 designs, randomized-100 R is 0.7082 / 0.6613 / 0.6687 / 0.6623. Best@1,000 F1 is 0.7428 / 0.7158 / 0.7171 / 0.7091. The two-epoch 5e-4 improvements over one epoch are inconclusive for both metrics. Its oracle deficit versus default also spans zero. Natural-protein improvements should not be generalized automatically to designs.

The independent exp364 evaluation uses seed 0 instead of our paired SHA-derived seeds. Natural-protein R differs by at most 0.0036 across new checkpoints and ranges. The largest design discrepancy is long-range 5e-4: 0.6309 versus 0.6371, difference -0.0062 [-0.0218, 0.0100]. All eight difference intervals span zero.

Input hashes and all 4,640 reference metric-row candidate/truth/top counts match. This supports a stochastic-draw explanation; protein-bootstrap intervals are conditional on these two realized evaluations, not repeated seed experiments.

## Execution, interpretation and reproducibility

Both smoke tests and all 12 production H100 shards succeeded on their first attempt. New work used 8.64 inference GPU-hours and 10.02 production task GPU-hours including setup/publication. All 440,800 new samples are retained. Four final caps occur only at temperature 1.3; none affect primary temperature 1.

Validation checks paired prompts/seeds, checkpoint hashes, runtime versions and parsed contacts. All 1,624 original raw files are unchanged; 6,496 first-300 curve rows reproduce within 1e-12. Fifteen tests cover parser/scorer behavior, exact shuffle margins, F1/oracle ordering and complete four-model pairing.

Two-epoch 5e-4 is more accurate and less excessively diverse than one epoch. Default still leads natural-protein mean F1, oracle best@1,000 and ensemble accuracy. Reduced Neff dependence is not established for the strongest redesign-only checkpoint. Different training schedules/exposure prevent causal isolation of an extra epoch or native-document removal.

The README is the complete report. Figures regenerate from committed tables. New raw predictions (1.423 GB), source bundles, inputs and this report are public under v3-two-epoch; v1/v2 remain immutable. Each appendix figure prints its generating command.

Public bucket: open-athena/MarinFold/data/exp373-mpnn-diversity-neff/v3-two-epoch
