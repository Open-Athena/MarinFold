# Summary slides — qualitative contact-map errors

## What was inspected

97 natural eval-val proteins, exp277 epoch 2, step 479417.
Original atlas: 100 saved rollout-and-resample predictions per protein.
Follow-up: 200 fresh whole-map samples per protein; all 19,400 completed.
All 1,940 original metric rows reproduced exactly.
HTML atlas shows every predicted/true map, votes and errors, with zoom.
No eval-test analysis.

## The strongest visual pattern

Severe failures replace long-range traces with near-diagonal contact clusters.
Worst accuracy quartile: 69.3% long-range truth vs 41.2% predictions.
Examples: 8arl_A, 7wz5_A, 8bux_A, 8dmu_A.
Better maps recover broad features but often miss detailed contacts.
Positive control: 7y8h_A reaches 84.3% exact precision.

## Why visual resemblance can mislead

Exact precision: 55.75%. Many-to-one proximity within two residues: 80.72%.
Keep exact matches and pair FP with FN one-to-one: only 64.01%.
The truth-informed local correction gains 8.26 percentage points.
Most errors remain; local proximity is not a substitute for exact accuracy.

## Sampling support in the original atlas

93.9% of true contacts appear in at least one rollout.
29.3% get fewer than ten votes; top-L/5 precision is 81.2%.
Correct contacts are often present but weakly ranked.
Marginal votes cannot reveal within-rollout consistency or competing folds.
The follow-up below tests individual sample quality and selection.
Training causes remain hypotheses; these maps do not establish one.

## Whole-map probe: selection does not improve the mean

Pooled top-R: 55.79% F1; mean individual map: 41.53%.
Truth-oracle best of 200: 55.04%; paired difference −0.75 pp [−2.00, +0.57].
Pooled votes beat even the best individual on 62 of 97 proteins.
Consensus selection: 48.95%; mean-token-likelihood selection: 46.73%.
Best of 100 → 200 adds 1.20 pp [0.95, 1.50]; no claim beyond this budget.
Variable sample sizes are scored with whole-map F1, precision and recall.

## Errors persist within samples, with some exceptions

Best samples remain poor: 7wz5 9.3%, 8arl 13.1%, 8bux 11.2%, 8dmu 10.5%.
8arl recovers part of an arc; 8dmu still misses extensive distant contacts.
8b6a pooled 66.1% beats its oracle sample 60.2%.
35 of 97 proteins have a better individual sample than pooled votes.
Largest gains: 7t9r 29.2% → 52.6%; 7xp9 28.9% → 50.0%.
The HTML shows every sample and sorts its gallery by these oracle gains.

## Clustering does not yet provide a strong modes test

Oracle k=4 cluster voting: 55.73%, tied with pooled 55.79%.
Average linkage mostly isolates outliers: median largest training group 96/100.
Only 37/194 k=4 fits have at least two eligible test clusters (≥10 samples).
This null result is limited by the grouping, not proof that useful modes cannot exist.
Raw maps, cluster memberships/counts, scores and timings are public.

## Validation and interpretation

All 19,400 maps terminate; 582 pooled R-precision checks match the canonical scorer.
Independent 100-sample means 55.56% / 55.66% reproduce saved 55.75% within tolerance.
Pure inference: 19.15 minutes on one H100; staging/loading recorded separately.
Errors within generated maps deserve attention; choosing one sample is insufficient.
Contact-map quality does not establish 3D feasibility or a training cause.
Next causal probe: true long-range anchors, then score generation of remaining contacts.
