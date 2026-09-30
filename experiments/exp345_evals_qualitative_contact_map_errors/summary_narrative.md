# Summary slides — qualitative contact-map errors

## What was inspected

97 natural eval-val proteins, exp277 epoch 2, step 479417.
100 saved rollout-and-resample predictions per protein; no new inference.
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

## Sampling support and next steps

93.9% of true contacts appear in at least one rollout.
29.3% get fewer than ten votes; top-L/5 precision is 81.2%.
Correct contacts are often present but weakly ranked.
Marginal votes cannot reveal within-rollout consistency or competing folds.
Next: inspect whole-map samples and test truth-free structural selection.
Training causes remain hypotheses; these maps do not establish one.
