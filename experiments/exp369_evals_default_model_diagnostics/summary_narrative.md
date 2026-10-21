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

## Contact injection: running

6,664 conditions, 666,400 rollouts on the 97 eval-val proteins.
Random versus SSE-balanced true contacts; two independent context draws.
Count, fraction, and one/two/four contacts per element views.
Supplied pairs never score. Unconditioned comparator uses the same remaining pairs.
Report 95% protein-bootstrap intervals and first tested dose with a 0.05 gain or R-precision 0.80.
All 1,000 initial validation completions passed. Rare out-of-range model predictions exposed an incorrect shard-stop gate in the larger run.
Corrected bookkeeping retains all samples and reports generation errors plus a zero-invalid-rollout sensitivity score. The one-H100 case run completed all 1,200 samples; twelve workers restarted the full sweep.

## Verified AFDB/PDB disagreements

Screened 700 retained training entries; 357 usable mappings.
Confirmed 20 against the original AFDB v4 source; training contacts reproduce exactly.
Six cases selected: three large assembly-bound discrepancies (TM-score 0.25–0.33), NDUFA4 contact disagreement, and two alternate-state controls.
Later PDB release dates follow the original November 2022 AFDB objects.
Assembly context and biological state prevent labeling all disagreements as unequivocal folding errors.

## Model preference: mixed, no reliable correction

Dynein regulatory complex subunit 2 favors AFDB: exclusive mean votes 0.206 vs 0.025; R-precision 0.326 vs 0.194.
Tex36 weakly favors PDB, but accuracy is low: 0.093 versus AFDB, 0.156 versus PDB.
Centrosomal protein: R-precision zero against both structures.
SemiSWEET strongly favors the AFDB state; NDUFA4 is weak and OCP is mixed.
All 1,200 rollouts finished. Six had generation errors; zeroing them preserves these conclusions.
