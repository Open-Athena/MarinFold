# Frozen analysis plan for issue 373

Frozen before new inference on 2026-10-30. Checkpoint and input identities,
representative panel, seeds and sampling parameters are in `data/plan.json`.

## Design and primary endpoints

Evaluate all 97 natural eval-val proteins; report the 19 eval-denovo proteins
separately. Never select or score eval-test. Both checkpoints run in the same
vLLM 0.9.2 image, BF16, 8192-token context, top-p 0.95 and top-k disabled.
At each temperature (0.7, 1.0, 1.3), each protein receives 300 samples from the
same fixed prompt (three independently seeded blocks of 100) and 100 samples
from the matched randomized prompt bank. Prompt IDs and sampling seeds are
identical between checkpoints and temperatures. The primary temperature is 1.

Primary diversity endpoint: per-protein mean Jaccard distance over all unordered
pairs of 300 canonical fixed-prompt contact sets, restricted to the evaluator's
resolved candidate universe (separation >=6). Both-empty sets have distance 0;
one-empty sets have distance 1. Also report unrestricted valid contacts, long
range (>=24), each seed block, randomized prompts, inclusion entropy per candidate
pair, and contact counts. No malformed or capped output is removed. The main
parser folds contact/retract edits in order, then removes offset, orientation,
duplicates and ineligible pairs. Diagnostics preserve each removal reason.

The primary contrast is ProteinMPNN-only minus default, averaged over proteins,
with a paired percentile bootstrap (10,000 draws, seed 373). Proteins, not
rollouts, are the independent units. Seed-block results describe sampling
variability; they do not multiply the biological sample size.

Contact-count control: at each matched sample index, uniformly thin both models'
contact sets to the smaller of their two sizes, using deterministic independent
subsampling; report the same Jaccard contrast. Length is already paired exactly;
add paired-difference regression against log length. Interpret higher diversity
only after these controls and the randomized-prompt sensitivity.

## Accuracy and useful alternatives

Use exp89's exact resolved-pair universe, >=0.001 truth threshold, stable ranking,
and all/short/medium/long precision-at-L,L/2,L/5,R and AUC. Aggregate frequency
votes over retained samples, including capped prefixes. Report single-rollout
precision and recall (empty precision is 0; undefined recall is missing),
true/false unique-contact discovery, and ensemble R-precision at 1, 10, 25, 50,
100, 200 and 300 samples where available. Fixed-prompt curves use the saved
sample order; seed blocks supply separate 100-sample checks. Report temperatures
as diversity–accuracy curves; a higher-diversity arm is not useful unless correct
coverage or ensemble quality improves at comparable quality. Do not infer
physical conformations from contact-list differences.

## Neff contrasts

Primary performance condition: 100 randomized prompts, temperature 1. Use
`n_eff_0.8` from exp260's frozen FoldBench-volume metadata; validate a unique stem
join, sequence length, and exact evaluation sequence against exp245. Metadata
has all 97 natural proteins: tier counts are 4, 10, 28, 55 at <10, 10–<100,
100–<1000, >=1000. Three designed proteins lack metadata and retain their scores.
The length check is not a fresh audit of the original alignment query sequence;
that limitation will be documented.

Report both models' all- and long-range R-precision, tier counts and paired
intervals. Prespecified contrasts: (a) difference of high-minus-low tier accuracy
drops (MPNN minus default; negative supports flatter dependence); (b) OLS slope
of paired model difference against log10(Neff), with a protein bootstrap.
Report absolute low-tier scores. A flatter slope caused by worse high-tier
performance is not improved low-Neff accuracy. The four-protein lowest tier
limits precision. Repeat for raw sequence count, fixed prompts, covariates
log(length) and viral status, and nonviral proteins. Explore diversity and
ensemble gain versus Neff; label exploratory analyses explicitly.

## Qualitative and mechanism checks

Eight panel proteins are selected by evenly spaced ranks of Neff before seeing
predictions (ties broken by stem); display truth, both inclusion maps and their
difference. Use binary-matrix curveball swaps to preserve exactly the number of
contacts per rollout and the inclusion count of every contact. Compare contact
co-occurrence and the spread of pairwise overlaps against shuffled controls;
report mixing diagnostics and avoid calling clusters conformations.

Audit existing decontaminated native-corpus identities and sequence-KNN results,
including their actual source corpus. Join only eval-val rows. Trace the
ProteinMPNN training sources and identify any available redesign-sequence search.
If no corpus-wide redesign search exists, explicitly report the coverage gap and
do not attribute effects to retrieval/memorization. Native similarity and Neff
are distinct exploratory predictors.

## Execution, preservation and reporting

Use 12 independent batch-priority H100 jobs on cw-us-east-02a (six interleaved
length-sorted shards per checkpoint), after one real-protein smoke per model.
Use only existing same-region model exports. Verify source size/ETag or SHA256,
then record downloaded SHA256 for every file. No shared cluster mutation.
Save prompts, token IDs, seeds, canonical contacts, malformed/termination counts,
per-protein timings, hardware, checkpoint identity and source bundle digest.
Continue capped prefixes within the original context using the exp364 policy;
report context-exhausted samples without replacing them.

Publish raw per-protein compressed JSON and completion markers under the public
HF bucket, with small CSVs and figures in git. Bootstrap intervals are descriptive,
not adjusted for the secondary comparisons. Final README, figures and summary PDF
will state null results and limitations directly. Submit the complete record via
a labeled feature-branch PR; do not close the experiment issue.
