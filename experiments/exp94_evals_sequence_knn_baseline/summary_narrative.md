# Summary slides — sequence-KNN baseline for contact prediction (#94)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

A no-folding null model for contact prediction: for each of the 554 eval proteins,
mmseqs-align it against all 4.13M contacts-v1 training documents, copy the top-k
neighbors' ground-truth contacts through the alignment, vote, and score R-precision
with the exact same metric harness as every other predictor.

## Why

To test how much of MarinFold's contact-prediction skill is just memorizing training
folds. If copying nearest neighbors matches the model, it's largely retrieval.

## Results so far

seq-KNN k=10 ties MarinFold #61 on AVERAGE long-range R-precision (0.324 vs 0.353) —
but that's a coincidence of averaging. Stratified by nearest-neighbor identity,
seq-KNN rises monotonically (0.00 -> 0.69) while MarinFold stays flat (~0.35) and is
the only predictor that works on the 139 no-homolog proteins. So MarinFold is NOT
just memorizing: its accuracy is independent of training-sequence proximity.
