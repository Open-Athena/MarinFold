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

## Contact injection: a useful nudge, no sharp transition

All 6,664 conditions / 666,400 rollouts complete on the 97 eval-val proteins.
Eight random contacts: remaining R-precision 0.610; gain 0.061 (95% CI 0.046–0.079).
5% target budget: remaining R-precision 0.615; gain 0.064 (95% CI 0.050–0.081).
Both policies first cross the predeclared five-point mean-gain criterion at eight contacts / 5%.
Random peaks at 0.711 (95% CI 0.698–0.724), at the 40% budget.
No tested dose reaches mean R-precision 0.80. Long-range results are similar.
Supplied pairs never score. As dose rises, the remaining truth and R shrink; the declining absolute curve at large budgets does not establish that more context harms predictions.

## Secondary-structure coverage does not improve the early response

One contact per coverable element uses a mean 16.9 contacts / 7.04% of truth.
Remaining R-precision: 0.628 balanced versus 0.626 random at the same counts.
Paired difference 0.0023 (95% CI −0.0051–0.0098): no clear advantage.
Two contacts per element also show no clear advantage.
At 40% truth supplied, random exceeds balanced by 0.0187 (95% CI 0.0119–0.0253).
The policy comparison also changes contact composition: one balanced pass contains fewer long-range contacts than random sampling at the same count.

## Completeness and robustness

Zero token-cap hits across all 666,400 rollouts; 2,218 invalid rollouts (0.333%).
All samples retained, with malformed / nonexistent-residue pairs unable to vote.
Zeroing entire invalid rollouts changes all-range curve means by less than 0.00022.
Fresh unconditioned mean 0.55449 reproduces archived 0.55375; paired difference CI includes zero.
Eight protocol tests and twenty exact canonical-scorer comparisons passed.
Raw rollouts, vote matrices, timings, frozen inputs, and source code are published with checksums.

## Verified AFDB/PDB disagreements

Screened 700 retained training entries; 357 usable mappings.
Confirmed 20 against the original AFDB v4 source; training contacts reproduce exactly on the common resolved residues used in each comparison.
Six cases selected: three large assembly-bound discrepancies (TM-score 0.25–0.33), NDUFA4 contact disagreement, and two alternate-state controls.
Later PDB release dates follow the original November 2022 AFDB objects.
Assembly context and biological state prevent labeling all disagreements as unequivocal folding errors.

## Model preference: mixed, no reliable correction

Dynein regulatory complex subunit 2 favors AFDB: exclusive mean votes 0.206 vs 0.025; R-precision 0.326 vs 0.194.
Tex36 weakly favors PDB, but accuracy is low: 0.093 versus AFDB, 0.156 versus PDB.
Centrosomal protein: R-precision zero against both structures.
SemiSWEET strongly favors the AFDB state; NDUFA4 is weak and OCP is mixed.
All 1,200 rollouts finished. Six had generation errors; zeroing them preserves these conclusions.
