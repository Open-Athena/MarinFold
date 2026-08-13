# Summary slides — exp: curate an experimental-PDB contacts-v1 corpus (Protenix-style, <= 2021-09-30) — monomers and multimers

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we build an **experimental-structure** contacts-v1 corpus from the PDB — curated
the way Protenix/AF3 curate their training set — and use it to fine-tune our best
model (`contacts-v1-exp199-1.5B`, R-precision 0.587)?

Every contacts-v1 corpus so far is *predicted* structure: exp53/exp105/exp132 are
AFDB (AlphaFold2 predictions), exp139/exp155 are ESM-Atlas (ESMFold2 distillation).
The model has never seen an experimental structure in training, and it has never
seen a **complex** — contacts-v1 has been single-chain since day one. Both gaps are
addressable with the same curation pass.

## Why

_(Copy from the issue.)_

## Results so far

_(Fill in as results come in.)_
