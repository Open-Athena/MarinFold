# Summary slides — exp: does training on backtracking traces let contacts-v1 self-correct at inference?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does continue-training on the backtracking corpus (1) preserve standard contact-prediction accuracy, and (2) let retraction-enabled rollouts beat the settled resample+vote recipe at matched inference compute?

## Why

A model trained on self-correction traces can walk back a bad early commitment mid-rollout, recovering precision that today's append-only model loses irrevocably. **The bar is high:** #82's resample + pairwise-tiebreak recipe already recovers much of what any single bad rollout loses, by averaging over many independent rollouts. Backtracking has to beat *that*, not merely beat greedy decoding. It is entirely plausible that per-rollout self-correction and cross-rollout voting are redundant, in which case the honest result is "no gain over resample+vote" — worth knowing either way.

## Results so far

_(Fill in as results come in.)_
