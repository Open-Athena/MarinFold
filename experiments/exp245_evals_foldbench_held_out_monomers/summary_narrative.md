# Summary slides — exp: FoldBench held-out monomer eval sets (eval-val / eval-test / eval-denovo) for the decontaminated #232 checkpoints

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

**How do the decontaminated #232 checkpoints score on FoldBench's natural monomers — held-out ones we have never scored — and how does that compare to the structure-prediction baselines on the same proteins?**

Every contact number we publish today comes from a 554-protein set whose FoldBench half is the *first 100 rows* of `monomer_protein.csv`, chosen before we understood the training-set overlap ([#213](https://github.com/Open-Athena/MarinFold/issues/213), [#225](https://github.com/Open-Athena/MarinFold/issues/225)). [#232](https://github.com/Open-Athena/MarinFold/issues/232) trained models on corpora decontaminated against **all** of FoldBench, so for the first time the other 234 monomers are a legitimate held-out test set rather than an untested slice of training data.

This experiment cuts FoldBench's 334 monomers into three sets and scores them:

| set | definition | n |
|---|---|---:|
| **eval-val** | the natural monomers inside the historical FoldBench-100 — what every previous eval reported on | 97 |
| **eval-test** | every other natural FoldBench monomer — never scored by anything here | 218 |
| **eval-denovo** | every de novo designed FoldBench monomer | 19 |

Each protein carries a viral / non-viral flag so results can be stratified ([#241](https://github.com/Open-Athena/MarinFold/issues/241) found MarinFold ties ESMFold on viral proteins and loses badly on non-viral ones, so the split is not cosmetic).

## Why

- **H1.** eval-test and eval-val agree within noise for the #232 checkpoints. Both sets are decontaminated at the same rule, so a gap between them would mean the historical 100 is unrepresentative of FoldBench for reasons other than leakage.
- **H2.** The contaminated reference model (#199 CoreWeave cooldown, the current default) drops more from eval-val to eval-test than the #232 checkpoints do — its training data was never filtered against the 234, so eval-val is partly memorised for it and eval-test is not.
- **H3.** Baselines (Protenix-v2 single-seq/+MSA, ESMFold, ESMFold2) move little between eval-val and eval-test: their training data is unchanged by any of this, and eval-test is not novel to them.

## Results so far

_(Fill in as results come in.)_
