# Eval-val contact precision and historical CASP context

## What the current default achieves

exp277, step 266344: 97 natural eval-val monomers.

CASP-style contacts: CB <8 Angstrom, CA for glycine; sequence separation >=24.

P@L/5: 74.6% (95% protein-bootstrap CI 69.4-79.4%).
P@L/2: 61.1% (56.4-65.4%). P@L: 45.9% (42.1-49.4%).

All 9,700 archived rollouts finished. No new inference, historical-CASP scoring, or eval-test assessment. Native-metric control: 1,940 rows reproduce to 1.11e-16.

## What can we say about historical CASP?

CASP4 (2000): CIRB 10% across about 2L predictions; 20% for confidence >0.8. Neither is P@L/5 (Lesk et al., Tables V-VI).

CASP7 (2006): long-range P@L/5 13% across specialist groups and 19 FM/FM-TBM targets; SAM-T06 15.5% on 15 FM-only domains (Izarzugaza et al.).

CASP11 / CASP12 (2014 / 2016): best-group P@L/5 26.7% / 47.1%. CASP13 (2018): leading accuracies around 70% (Shrestha et al.).

CASP13 / CASP14: top-five means 65% / 64% on 31 / 22 FM domains (Ruiz-Serra et al.). Sources and exact cohort descriptions: data/historical_context.csv and README.

## The defensible interpretation

Our eval-val contact-ranking accuracy is numerically far above typical early-CASP reports and near the range of leading late-2010s contact predictors.

This does not demonstrate a CASP win, year-equivalent folding ability, or comparable 3D accuracy. CASP assessed blind, difficult domains; eval-val is a reused development set of broad natural monomers. Training corpora and databases also differ across eras.

No historical L/2 or L values are inferred from L/5 summaries.

## Target difficulty changes the number

Recorded MSA depth 10-99: n=5, long P@L/5 38.8%.
Depth 100-999: n=24, 54.8%. Depth >=1000: n=68, 84.3%.

None of eval-val's proteins have depth <10. The frozen natural FoldBench subset at that depth is in eval-test and remains unscored here.

MarinFold uses a single sequence; depth describes homolog availability and target difficulty, not an inference input. These are small descriptive strata, not causal comparisons.

## Metric and robustness checks

Under native pyconfind truth, current-default long P@L/5 / P@L/2 / P@L = 75.8 / 58.8 / 39.0%. Previous exp232 default: 75.3 / 58.6 / 39.2%.

The decontaminated-native-corpus KNN reference scores 57.6 / 43.8 / 29.7%; it does not include exp277's redesigned-sequence arm.

Tie averaging gives 74.61% versus 74.63% for CB long P@L/5. Common-coordinate-universe restriction leaves the result unchanged. Excluding eight coordinate-index audit cases gives 75.6% on 89 proteins.

Bootstrap intervals describe protein variability only. Source votes, coordinates, hashes, CSVs, and timing records are frozen for reproduction.
