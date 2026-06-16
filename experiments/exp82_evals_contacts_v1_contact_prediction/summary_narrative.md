# Summary slides — exp: contact-prediction inference algorithms for the contacts-v1 1.5B model

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

How well can the contacts-v1 1.5B model (trained in #67) predict residue–residue contacts from sequence, and which inference algorithm extracts the most signal from it?

## Why

The model has learned a **weak-but-real** contact signal (early probe in #67: teacher-forced ranking AUC ≈ 0.59 vs 0.5 chance; free-generation ≈ random due to set-generation pathologies). We expect:
- Long-range contact precision well **above random** but **far below** a strong contact predictor (it was a quick/simple run; contacts-from-sequence is the folding problem).
- Structured inference — rollout-frequency voting and exp27-style **iterative growing-K** refinement — may extract more signal than naive pairwise scoring, the way exp27's iteration helped the prior model.

## Results so far

First run (`hf/step-11999`, 24 held-out test proteins): all three methods rank
long-range contacts ~2× better than random, but absolute precision is very low
(~1–2%). **iterative** is marginally best (long P@L 0.017 vs pairwise 0.014 vs
random 0.007); **rollout does not beat pairwise.** Unlike exp27, better
inference doesn't rescue this model — the bottleneck is the weak base model, not
the readout.

**Head-to-head (the #67 hypothesis): the prior contacts-and-distances-v1 1.5B
wins ~2–4×** (long P@L 0.028 vs 0.017; medlong P@L 0.048 vs 0.020) on the same
proteins — so the quick contacts-v1 run did NOT beat the previous 1.5B at
contact prediction (the prior trained ~50k steps on more data vs 12k). Both
harnesses re-run against any future checkpoint.
