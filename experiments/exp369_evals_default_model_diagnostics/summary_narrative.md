# Summary slides — exp: default-model contact spiking, length dependence, and AFDB error preference

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

For the registered default contacts-v1-exp277-m2-p06-full-epoch-1.5B checkpoint at step 266344: (1) how many true contacts, or what fraction, put generation on the right path, and does coverage of secondary-structure elements help; (2) how does eval-val R-precision vary with length; (3) on verified AFDB training entries contradicted by subsequently released experimental structures, does MarinFold favor the AFDB contacts or the experimental contacts?

## Why

A few distributed correct contacts may disambiguate the fold more efficiently than uniformly sampled contacts. Predictions on training-label errors may reveal whether the model generalizes beyond its AFDB supervision.

## Accuracy versus length

All 97 eval-val proteins; exp277 step 266344, canonical 100-rollout scores.
All-range R-precision: 0.554 (95% CI 0.512–0.594).
Long-range R-precision: 0.538 (95% CI 0.494–0.580).
Length association: Spearman rho −0.068, p=0.507 (all-range).
No detectable monotonic trend in this sample; longest bin has only eight proteins.

## Contact injection: approved, awaiting execution

6,664 conditions, 666,400 rollouts on the 97 eval-val proteins.
Random versus SSE-balanced true contacts; two independent context draws.
Count, fraction, and one/two/four contacts per element views.
Supplied pairs never score. Unconditioned comparator uses the same remaining pairs.
Report 95% protein-bootstrap intervals and first tested dose with a 0.05 gain or R-precision 0.80.
One H100 validation precedes the approved twelve-H100 batch sweep.

## Verified AFDB/PDB disagreements

Screened 700 retained training entries; 357 usable mappings.
Confirmed 20 against the original AFDB v4 source; training contacts reproduce exactly.
Six cases selected: three large assembly-bound discrepancies (TM-score 0.25–0.33), NDUFA4 contact disagreement, and two alternate-state controls.
Later PDB release dates follow the original November 2022 AFDB objects.
Assembly context and biological state prevent labeling all disagreements as unequivocal folding errors.

## Model preference: awaiting execution

Sequence-only predictions: two independent sets of 100 rollouts per case.
Score against both structures on their common resolved sequence.
Compare AFDB-only and PDB-only contact support, excluding shared contacts from the preference contrast.
No inference-based preference conclusion is available yet.
