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

## The sequence dependency, resolved

AFCDB ships no FASTA and `modelEntityId` is opaque, so subunit sequences come
from AlphaFold DB's bulk `sequences.fasta` — measured at **118.0 GB**, not the
110 GB the listing rounds to. A smoke run in `europe-west4-a` measured
**14.5 MB/s** on one 2-vCPU pod, and the same rate with 4 ranges as with 16,
so the scan is Python-bound rather than network-bound. It is therefore sharded
across 16 pods, each owning a byte window and subdividing it into ranges.

A sharded run cannot judge its own completeness, so `--verify` checks the union:
that the windows tile the file contiguously and that every requested accession
was matched exactly once.

## Three bugs the scale found

**The header parser.** AFDB writes `>AFDB:AF-<acc>-F1 ... UA=<acc>`, and the
leading `AFDB:` defeated the `AF-...-F<n>` pattern, so every header would have
produced the whole token as the "accession" and failed the missing-accession
check for every row.

**EBI refuses concurrent connections.** Seven of the first sixteen pods died on
`Connection refused` from the opening HEAD. The reader now retries with backoff
and jitter and resumes a dropped stream from the first unreturned byte.

**A truncated transfer reads as a clean EOF.** Writing the reconnect test found
something worse than the bug it was written for: when a server closes early,
`read()` returns `b""`, so an interrupted transfer was indistinguishable from
the end of the file. The range would stop short, report success, and ship a
short sequence set — under-decontaminating the corpus, silently, weeks
downstream. Ranges now carry the source length and treat running dry before the
range end as fatal.

## Guarding the join

The selector now rejects any model whose joined UniProt subunit lengths
disagree with the modelled residue count `n0chn`. Every AFCDB chain is at most
1,500 residues, well under AFDB's 2,700-residue fragmenting threshold, so an
`-F1` model covers residues 1..L and the two should agree exactly. When they do
not, the sequence we hashed, deduplicated and decontaminated on is not the
sequence the document will be generated from, and the eval-leakage decision for
that row is unsound.

## Stage A result: 3,000,000 documents

The selector produced **3,000,000 documents**, every one a distinct sequence
pair — 1,753,570 Tier A and 1,246,430 Tier B. Every one of the 29,025,020
source models carries exactly one terminal status: the ledger sums to
29,025,020 and its selected rows sum to 3,000,000.

The two dominant rejections are the relaxed quality floor (10.74M) and backbone
clashes (10.44M), then eval homology (2.02M).

## The heterodimer arm is exhausted, not rationed

The run selected **183,809** heterodimers against the reduced 200,000 floor:
every Tier-A heterodimer plus every Tier-B heterodimer above the floor. The
pool ran out. This is the census finding one stage deeper, now after
decontamination and dedup — AFCDB does not contain 200k usable heterodimers,
let alone 500k. Growing this arm needs Stage-E sources.

## A bug the accounting caught

The first run's ledger summed to 3,000,017 against a 3,000,000 manifest.
**AFCDB's `modelEntityId` is not unique across its two source tables**: 130 ids
appear as both a homodimer and a heterodimer, genuinely different complexes
that share an identifier. Joining on the id alone duplicated ledger rows and
let one source's row mask the other's in the Tier-B anti-join. Every join now
keys on `(complex_type, model_id)`, asserted unique rather than assumed.

## Why throughput needs its own sample

The stratified 50k draw spreads across 12,818 tars at **3.9 models per tar**;
the real run sees **180.3**. Extraction cost is per tar, not per model —
walking a tar's headers costs the same for 4 members or 400 — so timing the
stratified draw would overestimate per-document cost by ~46x. A separate
throughput probe takes every selected model from 20 whole tars: 4,273 models at
**213.7 per tar**, the density the run actually sees.

## Still to come

Stage D extraction, sized from the probe rather than assumed, then
interface-aware cluster assignment and the balanced manifests. The
reconnaissance already says range-addressed extraction moves ~1.5-2.5 TB
against streaming's full 48.8 TB.
