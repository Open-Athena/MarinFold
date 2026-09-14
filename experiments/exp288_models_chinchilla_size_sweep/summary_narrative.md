# Summary slides — Experiment: Chinchilla size sweep on cleaned contacts-v1 corpora

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

For the cleaned native + MPNN-redesigned contacts-v1 corpora from #266/#274, what model size is compute/data efficient? In particular, do 0.7B, 1.5B, and 3B Qwen-style contacts-v1 models show the expected Chinchilla-style tradeoff between parameter count, token budget, validation cross-entropy, and downstream contact R-precision?

## Why

The new ~248B-token corpus is large enough that the current 1.5B recipe may be under-sized at fixed token budget. A 3B model should improve validation CE and possibly FoldBench R-precision over 1.5B if the extra MPNN/native data carries independent signal; a 0.7B model provides a cheap lower anchor for fitting a scaling trend rather than only comparing two large runs.

## Planned Chinchilla measurements

Three fixed-recipe arms: ~0.7B, ~1.5B, and ~3B parameters. For each arm we will track validation CE vs tokens, validation CE vs GPU-hours, final token/parameter ratio, throughput, and selected exp245 eval-val contact metrics.

The goal is to collect scaling evidence for the next MarinFold recipe. A smaller arm anchors the slope; the current 1.5B scale preserves continuity; the 3B arm tests whether the cleaned/redesigned corpus makes 1.5B under-sized.

## Results so far

Experiment scaffolded; no training launched yet.
