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

## AFDB comparison corrected: biological monomers

The original complex/state examples do not establish AF2 monomer errors; that conclusion is withdrawn.
Replacement filter: every biological assembly has one protein and no other polymer partners. Crystal copies may be separate monomers. A literature/context audit rejects extracted oligomer protomers even when metadata says monomer.
534 candidate chain mappings / 122 retained AFDB proteins pass assembly metadata; 131 mappings / 37 proteins pass geometry and coverage filters. No unresolved fetch errors.
20 proteins undergo exact AFDB v4 source/contact verification. No clear catastrophic fold failures found (minimum common-CA TM-score 0.676). This is a limited search, not an error-prevalence estimate.

## Three moderate monomer disagreements

Pru p 7: 8X67, monomer assembly 1; solution NMR, all 20 conformers assessed.
RecJ: 8Y30, monomer assembly 2; gel-filtration evidence; chain B is a crystal copy, not a complex subunit.
MrpR/YopR: 8BJ6, monomer assembly 3; apo monomer with independent solution evidence; chain C is a crystal copy. SeMet is canonicalized to Met for contacts.
AFDB/PDB TM-scores: 0.713, 0.729, 0.676. Contact F1: 0.635, 0.825, 0.797.
Pru p 7 ensemble range: contact F1 0.575–0.713; TM-score 0.641–0.747.
These cases assess moderate discrepancies. They do not demonstrate correction of a catastrophic AF2 fold error.

## MarinFold on the corrected monomers

All 600 sequence-only rollouts complete; no token-cap hits. 19 invalid rollouts, all retained; sensitivity analysis preserves the conclusions. Inference: 73.96 s; model initialization: 62.80 s.
RecJ: R-precision 0.681 vs AFDB / 0.651 vs PDB; exclusive mean votes 0.343 / 0.219. Favors AFDB.
MrpR/YopR: R-precision 0.643 / 0.557; exclusive mean votes 0.352 / 0.097. Favors AFDB.
Pru p 7: R-precision 0.233 vs AFDB and 0.216–0.274 across 20 experimental conformers. Low accuracy and conformer-dependent preference; no clear winner.
This corrects the comparison to biological monomers but does not establish error correction on catastrophic AF2 monomer fold failures.
