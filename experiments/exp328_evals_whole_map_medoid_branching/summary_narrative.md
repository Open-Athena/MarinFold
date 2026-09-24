# Summary slides — exp: geometry-aware whole-map medoid branching

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can whole-contact-map clustering identify coherent rollout basins well enough that partial maps from cluster medoids improve oracle best-of-100 contact accuracy?

## Why

#326's pairwise contact clusters failed to improve held-out oracle accuracy.
Whole-map clustering is a stronger test: it defines modes using complete
predicted structures and branches from spatially distributed contacts copied
from actual medoid maps.

The primary objective is oracle best-of-100, not consensus. The 16-protein
development gate requires +0.005 versus both iid100 and a same-size random-map
seed control before any of the 81 confirmation proteins are scored.

## Results so far

The development gate failed, so the 81 confirmation proteins remain unread.

- 8-contact medoid oracle: -0.0038 vs iid100 and -0.0039 vs random-map control.
- 16-contact medoid oracle: -0.0049 vs iid100 and -0.0011 vs random-map control.
- Long-range oracle: -0.0096 and -0.0274 vs iid100.
- Consensus: -0.0195 and -0.0250 vs iid100.

Medoid seeds were much less accurate than random-map seeds (18% vs 31% at k=8;
17% vs 29% at k=16). The clusters capture coherent model errors, not useful
under-sampled conformations. Fold-switch movement was no better than random
seeding. Do not adopt this brancher.
