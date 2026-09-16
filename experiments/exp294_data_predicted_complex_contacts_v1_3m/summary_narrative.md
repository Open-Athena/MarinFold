# Summary slides — exp: curate three million predicted complex documents

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we curate at least **3,000,000 usable predicted protein-complex contacts-v1 training documents** from the NVIDIA/AlphaFold Database complex release, with relaxed but calibrated confidence tiers, without allowing a handful of repeated interfaces to dominate the training mixture?

## Why

The NVIDIA/AFDB complex release already contains enough usable predictions to reach three million training documents without running AlphaFold-Multimer ourselves. The earlier [#145](https://github.com/Open-Athena/MarinFold/issues/145) attempt became small because it was designed as a validation set: it restricted candidates to homologues of validation monomers and then selected one representative per coarse pair of independently clustered chains. For training, retaining multiple documents per **interface-aware** cluster while controlling their sampling weight should preserve the desired scale and prevent extreme redundancy.

We expect all usable high-confidence predictions to provide the core corpus. A calibrated medium-confidence tier should fill the remainder to three million. Homodimers and heterodimers, confidence tiers, and source datasets must remain separately identifiable so later training experiments can tune or ablate the mixture.

## Results so far

Stage A is implemented and locally tested: lossless metadata normalization,
confidence-yield census, exact sequence annotations with evaluation
decontamination, deterministic Tier A/B quota selection, and a complete
per-model terminal-status ledger.

The production selector cannot substitute accession pairs for sequence pairs
and cannot run without a completed decontamination annotation table. An unmet
three-million quota writes its quality frontier and exits non-zero instead of
silently lowering the quality floor.

Next: finish the one-time live metadata mirror, run the census, then use its
strata to construct the approximately 50,000-structure calibration pilot.
