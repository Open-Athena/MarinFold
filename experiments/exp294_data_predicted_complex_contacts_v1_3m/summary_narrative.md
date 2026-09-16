# Summary slides — exp: curate three million predicted complex documents

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we curate at least **3,000,000 usable predicted protein-complex contacts-v1 training documents** from the NVIDIA/AlphaFold Database complex release, with relaxed but calibrated confidence tiers, without allowing a handful of repeated interfaces to dominate the training mixture?

## Why

The NVIDIA/AFDB complex release already contains enough usable predictions to reach three million training documents without running AlphaFold-Multimer ourselves. The earlier [#145](https://github.com/Open-Athena/MarinFold/issues/145) attempt became small because it was designed as a validation set: it restricted candidates to homologues of validation monomers and then selected one representative per coarse pair of independently clustered chains. For training, retaining multiple documents per **interface-aware** cluster while controlling their sampling weight should preserve the desired scale and prevent extreme redundancy.

We expect all usable high-confidence predictions to provide the core corpus. A calibrated medium-confidence tier should fill the remainder to three million. Homodimers and heterodimers, confidence tiers, and source datasets must remain separately identifiable so later training experiments can tune or ablate the mixture.

## Results so far — the live census landed

The full 8.45 GB metadata mirror is normalized and censused: **29,025,020
models** (21,430,663 homodimers, 7,594,357 heterodimers), reproducing #145's
dimer count exactly. **1,923,625** clear both the hard validity filter and the
Tier-A quality gate.

Stage A itself is implemented and tested: lossless normalization, the yield
census, exact sequence annotations with eval decontamination, deterministic
Tier A/B quota selection, and a per-model terminal-status ledger.

## Two findings that change the plan

**The `pilot-v0` floor of 0.5 cannot reach 3M.** It yields 2,946,613 candidates
*before* sequence-pair dedup and decontamination. The target needs a floor at or
below ~0.4, and realistically ~0.3 to leave decontamination margin. Whether that
is defensible is the Stage-B pilot's job to decide.

**The 500k heterodimer floor is unreachable from AFCDB.** Only 922,359
hard-eligible heterodimers have a nonzero ipSAE at all, and 353,774 reach
ipSAE >= 0.1. Clearing 500k would mean accepting models with essentially no
predicted interface. The realistic ceiling is ~210k at floor 0.30. The issue
already conditioned this floor on the calibrated yield supporting it; it does
not.

## Archive reconnaissance: Stage D must move to the cloud

The coordinate archive is **48.8 TB** (35.20 TB homodimers, 13.58 TB
heterodimers) across 17,820 referenced tars. Measured single-stream EBI
throughput from the workstation is ~113 KB/s, so no variant of this runs
locally.

Two properties make a selective extraction viable instead of a bulk mirror:
each model is one member named `AF-<modelEntityId>-model_v1.cif.zst`, so member
names are a pure function of the model id; and EBI serves HTTP range requests,
so tar headers can be walked with 512-byte reads and only selected members
fetched. That moves roughly the selected payload (~1.5-2.5 TB) rather than
48.8 TB. Homodimer `chunk_*` tars also carry a redundant `.pdb.zst` per model
that selective extraction skips entirely.

## Blocking dependency

AFCDB ships no FASTA and `modelEntityId` is opaque, so sequences must come from
AlphaFold DB's 110 GB bulk `sequences.fasta` via a cloud-side streaming filter
over the 21,437,370 referenced accessions. `annotations.py` would have
mis-parsed that file's `>AFDB:AF-<acc>-F1` headers; fixed with a regression test
on the real header. The eval-decontamination drop list is still to be produced.
