# Summary slides — exp: evaluate best contacts-v1 model on current eval set

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide. -->

## What we're doing

How well does the best contacts-v1 1.5B model @eric-czech trained in #61/#75
(eval loss 2.7566) predict residue–residue contacts on our curated eval set,
alongside every other predictor — Protenix-v2 (single-seq / MSA), ESMFold,
ESMFold2? This is exp82's named open follow-up: does careful tuning turn the
near-chance #67 model into a real contact predictor?

## The model and the eval

Model: W&B `prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1` (Qwen3 1.47B; epochs 8,
lr 1e-3, wd 0.2; eval/contacts-v1-val/loss = 2.756602 at step 35679), exported
to HF and scored with the exp82 **pairwise** method (the best inference approach
from #82). Eval: 554 proteins (FoldBench-100 + exp65 low-MSA / novel-fold),
pyconfind side-chain ground-truth contacts, the same resolved-residue universe
for every predictor. Metrics: AUC + precision @ {L, L/2, L/5, R}, by range.

## Finding 1 — ranking AUC is competitive

MarinFold's long-range contact-ranking **AUC ≈ 0.88 (0.90 aggregate)** matches
**ESMFold** and beats **single-sequence Protenix-v2** (0.81). It is **robust to
MSA depth and fold novelty** — it even **edges ESMFold2 on novel folds**
(0.81 vs 0.79). So careful tuning took the model from ~chance (the quick #67
model in exp82) to ESMFold-class ranking AUC, **from sequence alone**.

## Finding 2 — top-K precision lags structure methods

At the sharp end — R-precision and contacts@L — MarinFold (≈ 0.27 / 0.19
long-range) sits **well below every structure predictor** (ESMFold2 0.77 / 0.44;
Protenix-MSA 0.80 / 0.47). The LM orders contacts broadly well but does not
concentrate confidence on the true top contacts.

## Takeaway

A 1.5B **sequence-only** LM now has a real, MSA-depth-robust contact signal
(high AUC) but is not yet a high-precision contact predictor (low top-K
precision). Note AUC somewhat favours a full continuous ranker (MarinFold scores
every pair) over structure-derived contact sets (degree 0 for most pairs); the
top-K precision metrics are the fairer "did you find the contacts" comparison,
and there structure + MSA still lead. Plots: contacts @ {L, L/2, L/5}, R and AUC
by predictor and range, stratified by MSA-depth tier and fold novelty.
