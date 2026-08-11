# Summary slides — exp: sequence-overlap audit of the contact eval set vs exp199's training data, and a homology-free re-eval

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

**How much of our headline contact-prediction number survives if we throw out every eval protein that has a sequence homolog in the training set?**

The worry is concrete. Our training labels are contacts derived from **AlphaFold2 (AFDB) and ESMFold2 (ESM Atlas) structures** — predictors that had MSAs (AF2) or a large PLM (ESMFold2) available. MarinFold sees only a single sequence at inference, but if an eval protein has a close relative in the training corpus, the coevolution/PLM signal that produced that relative's structure is effectively *baked into the weights* and reachable by homology. In that regime "single-sequence contact prediction" is partly an MSA-derived retrieval, and our eval number is inflated relative to what it claims to measure.

## Why

Two competing predictions, and the experiment distinguishes them:

- **H1 (leakage):** MarinFold's R-precision drops sharply on eval proteins with no training homolog, and its gap to Protenix-v2 single-sequence (currently a tie at 554 proteins) opens up in Protenix's favor.
- **H0 (generalization):** accuracy is roughly flat across training-identity strata, as [#94](https://github.com/Open-Athena/MarinFold/issues/94) found for the much weaker #61 model against the AFDB corpus alone.

Prior from #94 leans H0, but that analysis predates both the ESM-Atlas half of the training data and the +0.26 R-precision of the current model, and its "no homolog" bin was 123/139 de novo *designed* proteins. Both weaknesses are addressed here.

## Results so far

_(Fill in as results come in.)_
