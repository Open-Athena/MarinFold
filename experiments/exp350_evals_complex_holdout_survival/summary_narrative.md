# Summary slides — exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## Question

Can we construct a useful evaluation set of experimental protein complexes in which **every constituent protein chain is below 30% sequence identity to the training data**, and measure exp343's inter-chain contact R-precision and structural accuracy after Helico folding?

The first deliverable is a survival table and an evidence-backed feasibility decision, before running predictors.

## Audit contract

Candidates are natural protein-only dimers, with each chain at least 40 residues, total length at most 1,998, and X-ray/EM resolution at most 3.5 A. FoldBench uses the benchmark's exact ground-truth chain copies; PINDER uses its test split.

Every full and resolved chain representation is searched. A hit requires exact identity >=30% and alignment coverage >=50% of either sequence. A single chain hit rejects the complex. Unsearched arms remain unknown.

## Survival

FoldBench: 239 source assemblies -> 184 scoped dimers -> 14 after AFCDB -> 12 after PINDER train -> 4 after AFDB -> **0 after ESM-Atlas**.

PINDER: 1,955 test dimers -> 1,929 scoped dimers -> 40 after AFCDB -> 34 after PINDER train -> 8 after AFDB -> **1 after ESM-Atlas**.

The last PINDER target is 6XNR, a 290-residue homodimer. The two ProteinMPNN arms remain unsearched, so one is an upper bound.

## Why the earlier FoldBench filter was insufficient

exp225 searched all FoldBench chains, but its MMseqs search used the default score-derived identity estimate. Exact backtraces move some retained alignments across the 30% boundary.

For 8AXI-A versus one retained ESM-Atlas target, exp225 reported 29.7%. The exact backtrace has 200 identities over 620 aligned positions: 32.26%. This audit stores `nident` and applies the threshold with integer arithmetic.

## Decision

These sources cannot support a useful held-out evaluation of the current checkpoint. FoldBench has zero local-arm survivors; PINDER has at most one before ProteinMPNN, and 6XNR is exposed to Helico's fine-tuning pool.

Do not launch R-precision or structure predictions on this pool. Search newer experimental complexes against the checkpoint-exact indexes, or freeze the desired evaluation set first and retrain every data arm with that reference excluded.
