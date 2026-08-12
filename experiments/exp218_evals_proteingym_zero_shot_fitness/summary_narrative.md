# Summary slides — exp: is contacts-v1 a competitive bidirectional protein language model? (ProteinGym zero-shot DMS)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

contacts-v1 documents open with a **randomly ordered** list of `<pN> <AA>` statements.
So `contacts-v1-exp199-1.5B` is not only a contact predictor — it is an **any-order
(permutation) autoregressive model over amino acids**. Prompt it with every residue of a
protein except one and it returns a distribution over the missing one, conditioned on all
the others *and* on their exact sequence positions. That is the same conditional ESM-1v /
ESM-2 compute with a mask token, and it is the object ProteinGym's zero-shot DMS benchmark
scores.

**Is that conditional any good?** Concretely: what does `contacts-v1-exp199-1.5B` score on
the ProteinGym v1.3 substitution benchmark (217 assays, ~2.47M variants), under the
standard masked-marginals protocol, and where does it sit against the published
leaderboard?

Two things make it more than a curiosity, and they are the actual reasons to run it:

1. **Ensembling over orderings is a knob no baseline on that leaderboard has.** A masked
   LM has one conditional per masked position. We have one per *permutation*, and can
   average.
2. **We can compute the exact joint for multi-mutants.** Every single-sequence PLM on the
   leaderboard scores a k-mutant by *summing k independent single-site log-ratios*, and
   every one of them falls off a cliff as depth grows (ESM2-650M: 0.422 → 0.248 → 0.205 →
   0.163 for depths 1→4). An any-order AR model gets the true joint
   `log p(mutant AAs at S | rest of sequence)` from the chain rule at the same cost. 69 of
   the 217 assays include multi-mutants; **1.77M of the 2.47M variants are multi-mutant.**

## Why

_(Copy from the issue.)_

## Results so far

_(Fill in as results come in.)_
