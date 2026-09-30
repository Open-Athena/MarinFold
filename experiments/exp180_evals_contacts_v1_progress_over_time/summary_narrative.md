# Summary slides — contacts-v1 progress, September 30, 2026

## Current contact accuracy

On natural eval-val (97 proteins), the current default #277 epoch 1 scores
0.5551 and epoch 2 scores 0.5575. Their paired difference is +0.00240, with a
95% protein-bootstrap interval of -0.00644 to +0.01102: effectively a tie.
The first-epoch model remains the default.

These are the paired rerun from #277 / PR #297, using 100 rollouts plus
resampling. No new inference or eval-test read was needed for this update.

## The historical timeline

On the legacy 554-protein benchmark, the historical maximum remains #199's
0.63068, trained before decontamination. The new lineage moves from #232's
0.59164 to 0.60506 with more training, then to #277's 0.62051 and 0.62203
with native plus ProteinMPNN-redesigned training data.

The September slide extends the original timeline through September 30 and
retains the August goal of 0.600. Black tracks the running maximum across
rollout scores; teal tracks the lineage using decontaminated native data.
Original pairwise measurements are shown separately. Oracles never enter the
frontier. Dates refer to checkpoint training, not evaluation or PR merge.

## What can be compared

The legacy set is about 75% designs and includes baseline training exposure.
Use it to track our own model generations. The slide places external
structure-predictor baselines on the same 97 natural eval-val proteins instead:
Protenix-v2 + MSA 0.8460; ESMFold2 0.8015; ESMFold 0.7504; Protenix-v2
single-sequence 0.2632. The native decontaminated sequence-KNN null is 0.4071.

ProteinMPNN redesigns inherit decontaminated native backbones, but their
sequence homology has not been independently audited. Differences below
0.005 are treated as ties. The default model has not caught the strongest
structure predictors on natural proteins.

## Evaluation coverage and limits

Both #277 checkpoints scored all 670 requested units. Capped rollouts were
excluded from voting: 1 of 67,000 for epoch 1 and 44 of 67,000 for epoch 2.
The 97 natural eval-val proteins were unaffected by caps.

The frozen low-MSA-depth natural cut is partial: 11/16 proteins, with epoch
1/2 scores 0.29962/0.29450. The 5 FoldBench-only natural proteins are all in
eval-test and remain unread. Low-depth designs: 26/26, 0.60258/0.60686.
The published viral cut has six proteins, 0.48542/0.47959; the nonviral cut
includes designs. Do not interpret it as a natural-only comparison.

## Reproduction and historical appendix

plot_update_slide.py (no arguments) reads saved experiment results and writes
16:9 PNG, editable-text SVG, and PDF exports plus two CSVs with source paths.
The checkpoint CSV records the W&B timestamp evidence for new #232 dates.
Run build_summary.py (no arguments) to rebuild this presentation.

The appendix includes both September exports and the original August figures.
The older validation-loss and per-protein charts were not refreshed and are
explicitly labelled as historical snapshots. Their interpretations reflect
what was known before the subsequent contamination audit.
