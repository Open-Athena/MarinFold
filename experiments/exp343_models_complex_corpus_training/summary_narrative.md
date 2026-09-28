# Summary slides — exp: train 1.5B on the exp277 corpus plus the predicted-complex corpus

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does adding the [#294](https://github.com/Open-Athena/MarinFold/issues/294) predicted-complex corpus to the exp277 native + ProteinMPNN training mixture change contacts-v1 monomer contact accuracy, and does the resulting model actually learn multi-chain complexes?

## Why

The complex corpus is 11.15 B tokens against exp277's 248.58 B — **4.3% of the mixture**. So:

- **Monomer accuracy is a tie.** eval-val R-precision lands within the predeclared 0.005 threshold of exp277's 0.55375. A 4.3% mixture change, spent on a document type the eval does not contain, should not move a monomer benchmark either way. A loss of more than 0.005 would be the interesting result, not the expected one.
- **Complex modelling is not a tie.** Held-out complex LM loss drops far below what exp277's checkpoint achieves on the same documents. exp277 has never seen a multi-chain document, so it must pay for the shared 2000-index ring, the interleaved chain runs, and inter-chain contacts at any index distance. This is the measurement the experiment exists for.
- Designed-protein (eval-denovo) accuracy is the one monomer number with a plausible mechanism for moving: interface contacts are, geometrically, contacts between two pieces of chain that sequence separation does not explain, which is also what designed folds stress. No direction is predeclared.

## Results so far

_(Fill in as results come in.)_
