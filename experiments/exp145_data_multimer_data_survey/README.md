---
marinfold_experiment:
  issue: 145
  title: 'Explore multimer data'
  kind: data
  branch: main
---

# Explore multimer data

**Issue:** [#145](https://github.com/Open-Athena/MarinFold/issues/145) · **Kind:** `data` · **Branch:** `main`

## Question

Our training corpora (AFDB, ESM Atlas) are monomer-only. Being able to operate on
multimers would be valuable. **How much multimer content actually exists in the
databases we already use, and in the other databases we could reach?**

This is a survey, not an experiment. Nothing is trained, nothing is generated.
The numbers below were read out of the databases' own APIs, FTP listings, and
object stores on **2026-07-31**, with the AFDB confidence-score profiling and the
Foldseek run redone on **2026-08-02**; where a number is an estimate rather than
an exact count, it says so.

## Summary

The recommended data is the AlphaFold DB complex release at
`ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/`, filtered by the gate
the release defines for itself: **ipSAE ≥ 0.6 and pDockQ2 ≥ 0.23**. That passes
9.13% of homodimers and 0.92% of heterodimers, giving **1.96 M homodimers and
70 k heterodimers**.

The heterodimer metadata ships that verdict as a `passes_quality_threshold`
column. The homodimer metadata ships no verdict column, so `profile_confidence.py`
recomputes the criterion from the raw `ipSAE_AB`, `ipSAE_BA`, `pDockQ2_AB` and
`pDockQ2_BA` columns. The recomputation agrees with the declared column on 123,597
of 123,597 sampled heterodimer rows, with no disagreement in either direction,
which is what licenses applying it to the homodimers.

The release is CC-BY-4.0 including commercial use, and every model carries UniProt
accessions, taxon IDs and gene names for both chains, so it joins to what we
already key on. At 2.03 M confident complexes it is about six times BFMD and
ModelArchive combined; the unfiltered 29 M release is about ninety times. The
confident models are individually queryable at
`https://alphafold.ebi.ac.uk/api/complex/{uniprot_accession}`, and the bulk is FTP
only. Downloading the confident slice is roughly 230 GB (a model is 466 kB of
mmCIF and 115 kB gzipped, measured over ten of them).

Three things to plan for, all from the 45-structure Foldseek-Multimer run below:

- No conversion step is needed. Foldseek read all 90 chains from the 45 ModelCIF
  files with no parse warnings.
- Deduplicate the homodimer and heterodimer sets **together**, not as separate
  jobs. The sample already produced a cluster holding a *Merluccius polli* SNX4
  homodimer and a soybean heterodimer at multimer TM 0.62.
- Sequence-level filtering will not catch the redundancy. The sample clustered
  three *Brugia malayi* complexes carrying six different gene names, all pairwise
  slices of one Sm/Lsm heptameric ring.

### Every database checked

| Database | Multimer content | Non-redundant scale |
|---|---|---|
| **AFDB `collaborations/nvda/`** | **21.5 M homodimers, 7.6 M heterodimers** | 1.96 M and 70 k pass the release's gate |
| AFDB main release v6 | none; 241,070,489 single-chain models | n/a |
| AFDB `atbc`, `bfvd`, `ntdx`, `vr3d` | none; monomer chunk-tar collections | n/a |
| ESM Atlas | none; 1,095,530,880 + 6,600,755 single-chain models | n/a |
| PDB (RCSB, experimental) | 200,372 multi-chain protein assemblies | 37,124 clusters at 30% sequence identity |
| PINDER (PDB-derived) | 2,319,564 dimers | 42,220 training clusters |
| PPIRef (PDB-derived) | 322,454 interfaces | 45,553 after `iDist` dedup |
| BFMD (Foldseek) | 297,570 aggregated multimer predictions | 51,757 representatives |
| ModelArchive | ~11,879 multi-chain of 625,966 | concentrated in ~25 deposits, CC BY-**SA** |
| RCSB computed structure models | 2,063 multi-chain of 1,062,058 | the ModelArchive deposits again |

The two monomer-only rows are the corpora we train on today, and both were checked
rather than assumed. AFDB models carry one `_entity` and one `_struct_asym` record
each, verified on P68871, P00918, P0AEX9 and P04637. For the ESM Atlas I sampled
1,600 structures across both Lance datasets; every one had exactly one `chain_id`,
`entity_id` and `sym_id`, even though the on-disk format carries those fields plus
`chain_boundaries` and could represent a complex.

### What this rules out

Experimental multimer data is capped at roughly 40,000 distinct interfaces: RCSB
sequence clustering, PINDER's structural clustering and PPIRef's `iDist` arrive at
37 k, 42 k and 46 k by three different routes, and the PDB adds about 6,000
multi-chain entries a year. A multimer capability has to be trained mostly on
predicted structures.

The ESM Atlas can contribute candidate pairs rather than structures. About half of
its 6,824,676,938 provenance records encode contig and gene position, and 74.1% of
555,846 sampled SPIRE accessions have an upstream neighbour on the same contig.
Predicting those complexes ourselves is a separate project from using the AFDB
release.

The rest of this document is the evidence. The AFDB section carries the full
confidence-gate measurements (101 gates across five scores) and the sampling
caveats; the PDB, ModelArchive and BFMD sections carry the counts in the table
above; `data/` holds every CSV, and "Reproducing the numbers" at the end lists the
commands.

## AlphaFold DB

### The main release is strictly monomeric

The current release (v6, changelog dated 2025-09-15, built on UniProt 2025_03)
holds **241,070,489 structures**, up from 214,686,924 in v4. Every one is a single
chain: AFDB is keyed on UniProt accession, one accession per file, and the
changelog names the model "AlphaFold Monomer v2.0".

Verified directly rather than assumed. Downloading
`AF-P68871-F1-model_v6.cif` (haemoglobin beta, an obligate component of the α₂β₂
tetramer, so a case where a multimeric model would be biologically warranted)
gives:

```
_entity.id                       1
_entity.pdbx_number_of_molecules 1
_struct_asym.entity_id 1
_struct_asym.id        A
```

One entity, one copy, one chain. Same for P00918, P0AEX9, P04637.

Bulk access is worth noting because it is not what the FTP listing suggests:
`download_metadata.json` describes only **48 curated archives totalling 1,656,090
structures / 0.15 TB** (Swiss-Prot 1,100,244; 24 reference proteomes 328,602;
global-health set 227,244). That 1.66 M is essentially the
[`afdb-1.6M`](https://huggingface.co/datasets/timodonnell/afdb-1.6M) corpus we
already train on. The remaining ~239 M are reachable only through the flat GCS
bucket `gs://public-datasets-deepmind-alphafold-v4` or per-accession API calls.

### The multimer content: `collaborations/nvda/`

The FTP root has a `collaborations/` area holding five partner datasets (`atbc`,
`bfvd`, `ntdx`, `nvda`, `vr3d`. Four are monomer collections. **`nvda/` is not.**

```
https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/
├── homodimer_metadata.csv       6.00 GB
├── heterodimer_metadata.csv     2.45 GB
├── homodimers/                  9,892 tar shards, 35.2 TB
├── heterodimers/                8,209 tar shards, 13.6 TB
└── msas/                        three sets: 1.2 M, 8.8 M, 13.4 M
```

The directories are new: homodimers landed 2026-06-11, heterodimers 2026-07-08,
and the heterodimer metadata was refreshed 2026-07-17.

Row counts are **estimates**, not exact. The metadata CSVs are too large to
download for a line count, so `profile_confidence.py` reads 200 windows of 200 kB
at evenly spaced byte offsets in each and divides the file size by the mean row
length measured in that sample. Row lengths are tight (279.0 B/row for
homodimers, 323.1 for heterodimers), which puts the counting error near 1%:

| | shards | model data | metadata | rows sampled | estimated rows |
|---|---|---|---|---|---|
| homodimers | 9,892 | 35.2 TB | 6.00 GB | 143,142 | **21,489,407** |
| heterodimers | 8,209 | 13.6 TB | 2.45 GB | 123,597 | **7,595,156** |

Every model carries interface confidence scores (`ipTM`, `ipSAE` in both chain
orders, `pDockQ`, `pDockQ2` in both orders, `LIS`, `numberOfInteractions`,
`N_clash_backbone`, `N_clash_heavyAtom`), plus UniProt accessions, taxon IDs and
gene names for both partners.

#### The five scores, and why they disagree

Filtering this release means choosing a score before choosing a threshold, and
that first choice turns out to matter more. The five are not five noisy estimates
of one quantity; three of them are transformations of the network's own predicted
aligned error, one is a fit to experimental agreement, and they were designed to
fail in different places.

- **ipTM** is AlphaFold-Multimer's own output: a predicted TM-score computed over
  residue pairs that span the two chains. AlphaFold's guidance treats above 0.8 as
  a confident interface and below 0.6 as probable failure. Its known weakness is
  that it averages over *every* interchain pair, so a correctly predicted
  interface hanging off a long disordered tail scores badly for a reason that has
  nothing to do with the interface, and trimming the construct raises the score
  without changing the prediction.
- **pDockQ** ([Bryant et al. 2022](https://doi.org/10.1038/s41467-022-28865-w)) is
  a sigmoid fitted to reproduce DockQ from two cheap quantities: the mean pLDDT of
  interface residues and the log of the number of interface contacts. Its 0.23
  threshold is inherited from DockQ, where 0.23 is the boundary of an acceptable
  model. It is the only one of the five calibrated against experimentally
  determined structures rather than derived from the network's own uncertainty,
  and the only one whose value therefore carries an external meaning.
- **pDockQ2** ([Zhu et al.
  2023](https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714))
  replaces the contact count with the predicted aligned error of interface residue
  pairs and scores each interface of a multimer separately, which makes it
  directional: the release ships `pDockQ2_AB` and `pDockQ2_BA`.
- **ipSAE** ([Dunbrack 2025](https://doi.org/10.1101/2025.02.10.637595)) is ipTM
  with the averaging problem fixed. It keeps only residue pairs whose interchain
  PAE is good, and it rescales the TM-score's `d0` by the number of residues that
  survived that filter instead of by the length of the whole chain, so a small
  well-predicted interface is no longer diluted by everything around it. The
  release computes it at `ipSAE_PAE_cutoff = 10` and `ipSAE_dist_cutoff = 15`, and
  ships the `d0chn` and `d0dom` variants beside the primary score. It is also
  directional.
- **LIS** ([Kim et al. 2024](https://doi.org/10.1101/2024.02.19.580970)) keeps
  interchain residue pairs with PAE below 12 Å, inverts them onto a 0 to 1 scale
  so that low error scores high, and averages. It deliberately ignores physical
  proximity, which is what makes it the most permissive of the five here and the
  one aimed at flexible interfaces that pDockQ's contact count would miss.

Two of these are scores of a *complex* (ipTM, pDockQ) and three are scores of an
*interface as seen from one chain* (pDockQ2, ipSAE, LIS in their AB and BA forms).
That distinction is why the tables below report ipSAE twice.

#### How much of it survives a confidence filter

Measured with `profile_confidence.py`: 200 windows of 200 kB range-read from each
metadata CSV, giving **143,142 homodimer rows (0.67% of the file) and 123,597
heterodimer rows (1.6% of it)**. Cells give the pooled pass rate, the range across
the 200 windows, and the pass rate scaled to the full file (21.5 M homodimers,
7.6 M heterodimers). The per-window range is not a confidence interval; it is
there because the rows are blocked by organism, which the next section takes
apart. `ipSAE` takes the better of the two chain orders, matching what the
release's own `max_ipSAE` column does, and `ipSAE both ways` requires both.

| Confidence gate | homodimers | heterodimers |
|---|---|---|
| ipTM ≥ 0.23 | 39.12% (range 4.1 to 98.6) → 8.41 M | 33.99% (range 10.9 to 75.2) → 2.58 M |
| ipTM ≥ 0.3 | 29.54% (range 2.6 to 98.3) → 6.35 M | 18.52% (range 4.9 to 35.3) → 1.41 M |
| ipTM ≥ 0.4 | 22.19% (range 0.5 to 98.1) → 4.77 M | 9.85% (range 2.6 to 18.2) → 748 k |
| ipTM ≥ 0.5 | 17.00% (range 0.2 to 98.1) → 3.65 M | 5.76% (range 0.9 to 11.7) → 437 k |
| ipTM ≥ 0.6 | 12.81% (range 0.1 to 98.1) → 2.75 M | 3.35% (range 0.3 to 8.8) → 255 k |
| ipTM ≥ 0.7 | 9.21% (range 0.1 to 98.1) → 1.98 M | 1.76% (range 0.0 to 7.1) → 134 k |
| ipTM ≥ 0.75 | 7.58% (range 0.0 to 98.0) → 1.63 M | 1.25% (range 0.0 to 6.8) → 95 k |
| **ipTM ≥ 0.8** | 5.90% (range 0.0 to 98.0) → 1.27 M | 0.75% (range 0.0 to 4.1) → 57 k |
| ipTM ≥ 0.9 | 2.69% (range 0.0 to 96.9) → 577 k | 0.09% (range 0.0 to 1.6) → 7 k |
| **pDockQ ≥ 0.23** | 32.06% (range 2.5 to 98.1) → 6.89 M | 17.32% (range 0.2 to 48.3) → 1.32 M |
| pDockQ ≥ 0.3 | 26.87% (range 1.2 to 98.1) → 5.77 M | 11.33% (range 0.2 to 37.9) → 861 k |
| pDockQ ≥ 0.4 | 21.23% (range 0.6 to 98.1) → 4.56 M | 6.30% (range 0.0 to 25.1) → 479 k |
| pDockQ ≥ 0.5 | 16.01% (range 0.2 to 98.0) → 3.44 M | 3.21% (range 0.0 to 17.0) → 244 k |
| pDockQ ≥ 0.6 | 10.45% (range 0.1 to 98.0) → 2.25 M | 1.33% (range 0.0 to 6.6) → 101 k |
| pDockQ ≥ 0.7 | 3.59% (range 0.0 to 98.0) → 771 k | 0.19% (range 0.0 to 1.9) → 14 k |
| pDockQ ≥ 0.75, 0.8, 0.9 | none (see saturation, below) | none |
| **pDockQ2 ≥ 0.23** | 11.36% (range 0.1 to 98.0) → 2.44 M | 1.55% (range 0.0 to 5.3) → 118 k |
| pDockQ2 ≥ 0.3 | 10.13% (range 0.1 to 98.0) → 2.18 M | 1.22% (range 0.0 to 4.9) → 93 k |
| pDockQ2 ≥ 0.4 | 8.61% (range 0.0 to 98.0) → 1.85 M | 0.88% (range 0.0 to 4.0) → 67 k |
| pDockQ2 ≥ 0.5 | 7.27% (range 0.0 to 98.0) → 1.56 M | 0.60% (range 0.0 to 3.7) → 46 k |
| pDockQ2 ≥ 0.6 | 6.09% (range 0.0 to 97.7) → 1.31 M | 0.39% (range 0.0 to 2.8) → 30 k |
| pDockQ2 ≥ 0.7 | 4.84% (range 0.0 to 96.4) → 1.04 M | 0.23% (range 0.0 to 2.1) → 18 k |
| pDockQ2 ≥ 0.75 | 4.09% (range 0.0 to 95.2) → 879 k | 0.16% (range 0.0 to 1.6) → 12 k |
| pDockQ2 ≥ 0.8 | 3.30% (range 0.0 to 92.7) → 709 k | 0.09% (range 0.0 to 1.4) → 7 k |
| pDockQ2 ≥ 0.9 | 1.02% (range 0.0 to 17.6) → 220 k | 0.02% (range 0.0 to 1.0) → 1 k |
| ipSAE ≥ 0.23 | 22.22% (range 0.2 to 98.3) → 4.77 M | 5.37% (range 1.0 to 13.9) → 408 k |
| ipSAE ≥ 0.3 | 20.63% (range 0.1 to 98.1) → 4.43 M | 4.51% (range 0.8 to 13.2) → 343 k |
| ipSAE ≥ 0.4 | 18.26% (range 0.1 to 98.1) → 3.92 M | 3.46% (range 0.3 to 11.3) → 263 k |
| ipSAE ≥ 0.5 | 15.29% (range 0.1 to 98.1) → 3.29 M | 2.56% (range 0.3 to 7.7) → 194 k |
| **ipSAE ≥ 0.6** | 12.29% (range 0.1 to 98.0) → 2.64 M | 1.75% (range 0.0 to 7.0) → 133 k |
| ipSAE ≥ 0.7 | 9.03% (range 0.1 to 98.0) → 1.94 M | 1.01% (range 0.0 to 4.1) → 77 k |
| **ipSAE ≥ 0.75** | 7.35% (range 0.1 to 98.0) → 1.58 M | 0.63% (range 0.0 to 3.3) → 48 k |
| ipSAE ≥ 0.8 | 5.68% (range 0.0 to 98.0) → 1.22 M | 0.34% (range 0.0 to 2.0) → 26 k |
| ipSAE ≥ 0.9 | 1.68% (range 0.0 to 96.9) → 361 k | 0.01% (range 0.0 to 0.5) → 860 |
| ipSAE both ways ≥ 0.23 | 22.05% (range 0.1 to 98.3) → 4.74 M | 3.50% (range 0.3 to 11.1) → 266 k |
| ipSAE both ways ≥ 0.3 | 20.49% (range 0.1 to 98.1) → 4.40 M | 2.92% (range 0.2 to 10.8) → 222 k |
| ipSAE both ways ≥ 0.4 | 18.09% (range 0.1 to 98.1) → 3.89 M | 2.19% (range 0.0 to 9.0) → 166 k |
| ipSAE both ways ≥ 0.5 | 15.14% (range 0.1 to 98.1) → 3.25 M | 1.54% (range 0.0 to 5.4) → 117 k |
| ipSAE both ways ≥ 0.6 | 12.18% (range 0.1 to 98.0) → 2.62 M | 0.99% (range 0.0 to 4.0) → 75 k |
| ipSAE both ways ≥ 0.7 | 8.94% (range 0.1 to 98.0) → 1.92 M | 0.48% (range 0.0 to 2.1) → 36 k |
| ipSAE both ways ≥ 0.75 | 7.28% (range 0.1 to 98.0) → 1.56 M | 0.26% (range 0.0 to 1.6) → 19 k |
| ipSAE both ways ≥ 0.8 | 5.61% (range 0.0 to 98.0) → 1.21 M | 0.11% (range 0.0 to 1.4) → 8 k |
| ipSAE both ways ≥ 0.9 | 1.65% (range 0.0 to 96.9) → 356 k | 0.00% (range 0.0 to 0.2) → 307 |
| **LIS ≥ 0.203** | 24.67% (range 1.5 to 98.3) → 5.30 M | 7.37% (range 2.9 to 15.6) → 560 k |
| LIS ≥ 0.23 | 23.17% (range 1.4 to 98.1) → 4.98 M | 6.23% (range 1.9 to 14.7) → 473 k |
| LIS ≥ 0.3 | 19.48% (range 0.4 to 98.1) → 4.19 M | 4.01% (range 0.5 to 11.3) → 305 k |
| LIS ≥ 0.4 | 14.20% (range 0.2 to 98.0) → 3.05 M | 2.04% (range 0.2 to 7.6) → 155 k |
| LIS ≥ 0.5 | 9.04% (range 0.2 to 98.0) → 1.94 M | 0.83% (range 0.0 to 3.8) → 63 k |
| LIS ≥ 0.6 | 4.29% (range 0.0 to 98.0) → 921 k | 0.26% (range 0.0 to 2.3) → 19 k |
| LIS ≥ 0.7 | 0.65% (range 0.0 to 5.8) → 140 k | 0.05% (range 0.0 to 1.5) → 4 k |
| LIS ≥ 0.75 | 0.14% (range 0.0 to 1.6) → 29 k | 0.00% (range 0.0 to 0.2) → 246 |
| LIS ≥ 0.8 | 0.01% (range 0.0 to 0.4) → 2 k | none |
| **authors' gate** (ipSAE ≥ 0.6 and pDockQ2 ≥ 0.23) | 9.13% (range 0.1 to 98.0) → **1.96 M** | 0.92% (range 0.0 to 4.7) → **70 k** |
| declared `passes_quality_threshold` | (column absent) | 0.92% (range 0.0 to 4.7) → 70 k |
| ipTM ≥ 0.8 and zero backbone clashes | 5.63% (range 0.0 to 98.0) → 1.21 M | 0.52% (range 0.0 to 3.6) → 39 k |

Bold rows are each score's own published threshold, plus the release's combined
gate. The rest of this section is what the table means.

#### The release's own gate can be applied to both sets

The last three rows are the important ones, and the middle of the three is a
check rather than a result.

The heterodimer table ships a verdict column, `passes_quality_threshold`, together
with the two constants that define it: `quality_ipsae_threshold = 0.6` and
`quality_pdockq2_threshold = 0.23`. The homodimer table ships no verdict column at
all, which is why an earlier version of this report fell back on ipTM ≥ 0.8 as a
proxy and said so.

The proxy is unnecessary. Recomputing the criterion from the raw `ipSAE_AB`,
`ipSAE_BA`, `pDockQ2_AB` and `pDockQ2_BA` columns reproduces the declared verdict
on **123,597 of 123,597 heterodimer rows, with zero disagreements in either
direction** (`profile_confidence.py` checks this on every run and records the
result in `data/nvda_confidence_sampling.json`). Both directions matter: no row is
declared true and recomputed false, and none the reverse. So the recomputation is
the authors' gate rather than an approximation of it, and applying it to the
homodimer scores measures that set by the release's own standard:

| | pass rate | complexes |
|---|---|---|
| homodimers under the authors' gate | 9.13% | **1.96 M** |
| heterodimers under the authors' gate | 0.92% | **70 k** |

The homodimer number is worth comparing against the announcement, which claims
1.7 M high-confidence homodimers without publishing the criterion behind it. The
authors' heterodimer gate gives 1.96 M and an ipTM ≥ 0.8 cut gives 1.27 M, so
1.7 M sits between the two. That is consistent with the announced figure coming
from something close to the heterodimer gate, and it is the reason this report
quotes 1.96 M with the gate named rather than quoting 1.7 M as though it were
measured here.

#### Heterodimers are harder by roughly an order of magnitude

At every threshold of every score, the heterodimer pass rate is far below the
homodimer one, and the gap widens as the threshold rises. At ipTM ≥ 0.5 the ratio
is 3:1 (17.00% against 5.76%); at ipTM ≥ 0.8 it is 8:1 (5.90% against 0.75%); at
ipTM ≥ 0.9 it is 30:1 (2.69% against 0.09%). Under the authors' gate it is 10:1.

This is a property of the task, not a defect in the pipeline. A homodimer is two
copies of one fold, so the model has to place a chain against something whose
structure it already predicted well, and the answer is usually constrained by
symmetry. A heterodimer candidate is a pair drawn from an interaction screen, and
many of those pairs do not form a complex at all; there is no correct answer for
the model to find. The pipeline's ipTM early-stopping threshold of **0.1** is the
tell: rather than discarding a pairing that looks hopeless after the first pass,
it stops recycling and writes the model out anyway. The 7.6 M heterodimers are
therefore best read as a screen with its negatives retained, not as 7.6 M
complexes, and the 70 k that pass the gate are the actual claim.

For us that reframes what the heterodimer half of this release is. It is not a
7.6 M training corpus. It is a 70 k training corpus shipped alongside 7.5 M
labelled negatives, and the negatives may be the more unusual asset: nothing else
in this survey offers a large set of protein pairs that a good predictor examined
and judged not to interact.

#### Which score you filter on changes the answer by 10x

Hold the threshold at 0.5 and vary only the score:

| score at ≥ 0.5 | homodimers | heterodimers |
|---|---|---|
| ipTM | 3.65 M | 437 k |
| pDockQ | 3.44 M | 244 k |
| ipSAE | 3.29 M | 194 k |
| LIS | 1.94 M | 63 k |
| pDockQ2 | 1.56 M | 46 k |

A factor of 2.3 on homodimers and **9.5 on heterodimers**, from a decision that is
easy to make without noticing. The ordering is not fixed across the grid either:
pDockQ2 is the strictest score from 0.23 through 0.5, but at 0.6 and above LIS
overtakes it (homodimers 4.29% against 6.09%), and pDockQ's saturation drops it
from second-most-permissive to fourth by 0.7.

What separates the scores is mostly how they treat a *bad* model, not how they
rank good ones. ipTM never returns zero anywhere in either sample (its 5th
percentile is 0.09 for homodimers and 0.084 for heterodimers), so a pairing with
no recognisable interface still lands somewhere around 0.1 to 0.3 and a cut at 0.5
is only a few tenths above the noise floor. ipSAE and LIS return **exactly** zero
when their PAE filter retains no interchain residue pairs, which happens for 59.4%
and 51.6% of homodimers and 76.7% and 67.4% of heterodimers. Those scores
effectively answer a prior question, "is there an interface here at all", before
answering "how good is it".

That difference is what the release's combined gate exploits. `ipSAE ≥ 0.6` alone
passes 1.75% of heterodimers and `pDockQ2 ≥ 0.23` alone passes 1.55%; if the two
ranked models identically the conjunction would equal the stricter one, 1.55%, and
if they were statistically independent it would be 0.027%. The measured
conjunction is **0.92%**, about 60% of the stricter score alone. So the two
agree substantially about which heterodimers are good while disagreeing about how
many, and requiring both is a meaningful tightening rather than a redundant one.

#### A pDockQ gate above 0.75 is empty by construction

`pDockQ ≥ 0.75` returns nothing, and neither does anything above it. This is not a
statement about the release. pDockQ is a fitted sigmoid whose upper asymptote sits
below 0.75, so no structure anywhere can score higher: the largest value in the
143,142 homodimer rows is **0.742** and in the 123,597 heterodimer rows **0.737**.

This is worth flagging because "pDockQ ≥ 0.8" is the sort of threshold that gets
carried over from a filter written for a different score. It would return an empty
set, silently, with no error and no warning. The equivalent strict pDockQ cut is
0.7, which passes 3.59% of homodimers and 0.19% of heterodimers.

#### Directionality costs nothing on homodimers and half the heterodimers

ipSAE is computed once per chain order, and the release ships both. Taking the
better of the two (the `ipSAE` rows) against requiring both (the `ipSAE both ways`
rows):

| | homodimers | heterodimers |
|---|---|---|
| ipSAE ≥ 0.5, better direction | 15.29% | 2.56% |
| ipSAE ≥ 0.5, both directions | 15.14% | 1.54% |
| ipSAE ≥ 0.75, better direction | 7.35% | 0.63% |
| ipSAE ≥ 0.75, both directions | 7.28% | 0.26% |

On homodimers the two readings are within a percent of each other at every
threshold, which is what symmetry predicts: the two chains are the same sequence,
so the interface looks the same from either side. On heterodimers the strict
reading removes **40% of the survivors at 0.5 and about 60% at 0.75**. Those are the
models where one chain is confidently placed against the other but not the
reverse, which is the signature of a small or ill-defined interface, and they are
exactly the models a training set should probably not include. Note that the
release's own gate uses the permissive reading, so `passes_quality_threshold`
retains them.

#### The scores are zero-inflated, so percentiles mislead

A failed interface does not score low on ipSAE and LIS, it scores exactly zero,
because the PAE filter retains no residue pairs at all and there is nothing left
to average. That makes the median uninformative for three of the five scores:

| | ipTM | pDockQ | pDockQ2 | ipSAE | LIS |
|---|---|---|---|---|---|
| homodimer median / p90 / max | 0.19 / 0.67 / 0.97 | 0.097 / 0.607 / 0.742 | 0.010 / 0.309 / 0.957 | 0.00 / 0.671 / 0.950 | 0.00 / 0.480 / 0.828 |
| heterodimer median / p90 / max | 0.194 / 0.398 / 0.964 | 0.085 / 0.322 / 0.737 | 0.010 / 0.018 / 0.941 | 0.00 / 0.020 / 0.933 | 0.00 / 0.156 / 0.767 |
| fraction scoring exactly 0, homo / hetero | 0% / 0% | 1.2% / 7.3% | 1.2% / 7.3% | **59.4% / 76.7%** | 51.6% / 67.4% |

Heterodimer ipSAE has a median of 0, a p90 of 0.020 and a p99 of 0.702: the
distribution is a spike at zero, a long flat stretch of near-zero values, and a
thin tail of real interfaces starting somewhere in the last two percent. That
shape is why the pass rates fall so steeply between 0.23 and 0.3 and then so
gently afterwards, and it is why a mean or a median of these columns tells you
almost nothing. Threshold counts, which is what the big table gives, are the only
honest summary. Full distributions in `data/nvda_confidence_quantiles.csv`.

#### Why these totals are order-of-magnitude, not three significant figures

Sampling by byte offset works here only because the row format is near-fixed-width
and nothing about a row's *contents* correlates with its position inside an
organism's block. What does correlate, strongly, is which organism's block the
window lands in. Rows are grouped by organism, so a 200 kB window is not 700
independent draws; it is one draw of an organism plus 700 near-replicates of that
organism's difficulty.

`profile_confidence.py` therefore records every window separately, and
`data/nvda_confidence_windows.csv` quantifies how severe the grouping is: in **176
of the 200 homodimer windows a single organism accounts for more than 90% of the
rows**, against 50 of 200 for heterodimers, with 1,200 distinct organisms across
the homodimer sample. That difference between the two files is itself informative:
the heterodimer set is organised as within-organism interaction screens that are
large enough to span many windows, so a window is more likely to sit in the middle
of one screen than at a boundary.

The consequence is that the pooled rate is a point estimate with a wide and
asymmetric spread behind it, much wider for homodimers:

| | pooled | window median | window range |
|---|---|---|---|
| homodimer, authors' gate | 9.13% | 7.10% | 0.1% to 98.0% |
| homodimer, ipTM ≥ 0.8 | 5.90% | 3.84% | 0.0% to 98.0% |
| heterodimer, authors' gate | 0.92% | 0.67% | 0.0% to 4.7% |
| heterodimer, ipTM ≥ 0.8 | 0.75% | 0.63% | 0.0% to 4.1% |

The pooled rate sits above the window median in all four rows, which says a
minority of high-scoring windows is pulling the mean up. The extreme case is worth
understanding rather than discarding, because it is a real feature of the data and
not a parsing error: one homodimer window scored **98%** against a 7.1% median. It
lands inside a run of `hemL1` (glutamate-1-semialdehyde aminomutase, a genuine
obligate homodimer) repeated across hundreds of bacterial strains at consecutive
byte offsets, every copy scoring ipTM 0.95. Inspecting the window shows five
consecutive rows with the same gene name and pDockQ values agreeing to four
decimal places, which is what near-identical strain sequences produce.

That window contributes 0.98/200 = 0.49 percentage points to the pooled homodimer
figure of 9.13%, so it accounts for about 5% of the total and dropping it would
not change the conclusion. The right reading of the whole table is therefore:
**heterodimer totals are good to roughly ±30%, homodimer totals to a factor of
two.** Both are far more precise than the decisions they inform, which are of the
form "is this 70 k or 7 M".

#### Distinct model IDs are not distinct proteins

That `hemL1` run is not just a sampling nuisance. It points at something the
headline counts hide, and it changes what "1.96 M confident homodimers" is worth.

The homodimer set is built from UniProt, and UniProt assigns **one accession per
strain**. A conserved bacterial enzyme sequenced in three hundred strains is three
hundred accessions, three hundred predictions, and three hundred rows in the
metadata CSV, all of them modelling what is nearly the same protein and producing
what is nearly the same structure. Counting distinct model IDs counts the
sequencing effort, not the structural diversity.

| homodimer sample | |
|---|---|
| rows | 143,142 |
| distinct UniProt accessions | 143,142 (100%) |
| distinct gene names | 107,195 (74.9%) |
| distinct (gene, taxon) pairs | 118,157 |
| distinct organisms | 1,200 |
| longest run of one gene name at consecutive offsets | 767 |
| rows sitting in a run of 2 or more identical gene names | 15.8% |

The accession column is perfectly unique and tells you nothing. Gene names
collapse the same rows by a quarter. The 767-long run is one gene repeated 767
times in a row, and 15.8% of all sampled rows sit in a run of at least two.

Under the authors' gate the picture barely improves: the 13,063 confident
homodimers in the sample carry 9,964 distinct gene names, so name-level
deduplication alone removes about 24% of them, essentially the same fraction as in
the unfiltered set. Confidence and redundancy are independent here, which makes
sense, since a protein that folds well in one strain folds well in all of them.

Heterodimers are far less repetitive: 123,597 rows give 116,284 distinct gene
pairs (94.1%), and the 1,141 confident ones give 1,079 pairs (94.6%). Requiring
two specific proteins to co-occur is a much stronger constraint than naming one,
so the pair space is sparser by construction.

Two warnings about reading these numbers. Gene name is a **loose** key across
organisms: `rplC` in *E. coli* and `rplC` in *S. aureus* are homologs with
similar folds, not copies of one protein, so collapsing on the name alone would
merge things that are genuinely distinct. It is also a **weak** key, because it
misses everything that is structurally redundant under a different name, which the
Sm-ring cluster in the Foldseek section below shows is common. The two errors run
in opposite directions, but the second dominates: the structural count will end up
**lower** than the 74.9% gene-distinct figure, not higher. Measuring how much
lower is exactly what a full Foldseek-Multimer run over the confident slice would
settle, and it is the single most useful follow-up to this survey.

#### Coverage

A separate pass (10 windows of 2 MB, giving 72,286 homodimer and 61,762
heterodimer rows) shows the two sets are shaped quite differently:

- **Homodimers are one model per accession** (every `uniprotAccession` in the
  sample was unique), so they ran a monomer set through a dimer prediction rather
  than screening pairs. One accession is not one protein, though: see the
  redundancy table above, where 143,142 accessions carry only 107,195 distinct
  gene names.
- **Heterodimers are within-organism interactome screens.** 99.5% of pairs have
  `tax_id_1 == tax_id_2`. In the sample, 61,762 pairs involved 55,046 distinct
  proteins with at most 74 pairs per protein, so the pairing is a sparse
  interaction graph, not all-versus-all.
- **Taxonomic scope is broad and plant-heavy.** The most common heterodimer taxa
  seen were *Glycine max* (25.9% of sampled rows), *Arabidopsis thaliana* (10.3%),
  *Homo sapiens* (8.0%), *Oryza sativa* (7.1%), *Rattus norvegicus* (5.3%), *Zea
  mays*, *Mus musculus*. Because of the organism blocking these shares reflect
  which blocks the sample hit, not true dataset proportions, but they do establish
  that the screen spans dozens of species rather than being human-only.

#### Access: two tiers

The high-confidence subset is a first-class part of the database, served by a
**`/api/complex/{accession}`** endpoint that is separate from the familiar
`/api/prediction/` one (querying a dimer's model entity ID against `/prediction/`
returns `{}`, which is what initially misled me into thinking the release was
FTP-only). Asking for human haemoglobin alpha returns its partners directly:

```
$ curl https://alphafold.ebi.ac.uk/api/complex/P69905
  modelEntityId  AF-0000000211046780        providerId  NVIDIA
  uniprotAccession  [P69905, P68873]        gene  [HBA1, HBB]
  assemblyType  Hetero                      oligomericState  dimer
  globalMetricValue 94.52   ipTM 0.92   ipSAE 0.87   pDockQ 0.57   pDockQ2 0.94   LIS 0.69
```

Records carry stoichiometry, a `complexName`, taxon IDs and all five interface
scores. So the split is: **high-confidence complexes through the web UI and API,
the full low-confidence bulk through FTP only.**

#### What produced them, and under what license

The FTP `README.txt` covers only the directory layout and the score schema, but
the structures themselves are fully self-describing. Pulling the first 8 MB of
`heterodimers/shard_0_batch_0.tar` gives individually zstd-compressed ModelCIF
files (`AF-<16-digit-id>-model_v1.cif.zst`, ~30 kB compressed / ~120 kB raw), and
their `_software` and `_ma_protocol_step` blocks name the pipeline exactly:

| step | tool |
|---|---|
| paired MSA | MMseqs2-GPU release 18, ColabFold-style pairing |
| inference | **AlphaFold-Multimer weights v2.3.0**, run through **OpenFold** with TensorRT + cuEquivariance acceleration |
| settings | no templates, 1 model, max 4 recycles, **ipTM early-stopping threshold 0.1** |
| scoring | ipSAE (ipSAE, pDockQ, pDockQ2, LIS) |
| annotation | PyDSSP secondary structure |

That early-stopping threshold explains the long low-confidence tail: rather than
discarding bad pairings, the pipeline runs them cheaply and keeps them. The
primary citation is listed as *"AlphaFold Database expands to proteome-scale
quaternary structures"*, marked **"To be published"**.

#### Cross-check against the official announcement

EMBL announced this on
[2026-03-16, updated 2026-05-19](https://www.embl.org/news/science-technology/first-complexes-alphafold-database/):
a four-way collaboration between EMBL-EBI, Google DeepMind, NVIDIA and Seoul
National University (Steinegger lab), about 17 million GPU-hours, with candidate
pairs drawn from **20 major studied species plus the WHO priority pathogens list**
(which explains the crop and model-organism heavy taxa I sampled). Their published
counts line up with my sampling closely enough to trust both:

| | announced | measured here |
|---|---|---|
| complexes predicted | 30 M | ~29 M |
| homodimers | 1.7 M high-confidence + 18 M lower | 21.5 M total |
| heterodimers | ~80 k high-confidence + 8.1 M lower | 7.6 M total |
| high-confidence homodimers | 1.7 M | 1.96 M (authors' gate), 1.27 M (ipTM ≥ 0.8) |
| high-confidence heterodimers | ~80 k | 70 k (authors' gate, matching their own column) |

The byte-offset sampling reproduced the official figures to within about 10% on
the totals. On the high-confidence homodimers it brackets the announced 1.7 M
rather than reproducing it: the authors' own gate applied to the homodimer scores
gives 1.96 M and an ipTM ≥ 0.8 cut gives 1.27 M. Since the criterion behind the
announced homodimer figure has not been published, 1.7 M sitting between those
two is the expected result, and it says the announced number was produced by
something close to (but not identical with) the heterodimer gate.

**The license is CC-BY-4.0**, stated in each file's `_pdbx_data_usage` record
alongside the text "AVAILABLE FOR ACADEMIC AND COMMERCIAL PURPOSES, UNDER CC-BY
4.0 LICENCE"), the same terms as the AFDB main release, and more permissive than
the ESM Atlas's CC BY-SA 4.0. Usable for us.

Each model also carries full UniProt cross-references for both chains
(`_ma_target_ref_db_details`: accession, gene name, taxon, aligned range), so the
corpus joins cleanly against anything we already key on UniProt.

### Latent multimer content: what AFDB monomers *should* be

AFDB has no multimers, but a large slice of it models proteins that do not exist
as monomers. Counting UniProt SUBUNIT annotations (queried today via the UniProt
REST API):

| | reviewed (Swiss-Prot) | all entries with an AFDB cross-reference |
|---|---|---|
| total | 575,503 | 99,108,531 |
| has a SUBUNIT comment | 301,706 | 8,357,791 |
| annotated homo-oligomer | **101,703** (17.7%) | **2,925,590** |
| annotated hetero-oligomer | **30,937** (5.4%) | **912,992** |
| annotated monomer | 31,545 | n/a |

(Homo/hetero rows use an explicit union of `homodimer`/`homotrimer`/`homotetramer`/
`homopentamer`/`homohexamer`/`homooctamer`/`homooligomer`/`homomultimer` and the
hetero equivalents, rather than a `homo*` prefix wildcard, which also matches
"homolog".)

So at minimum ~2.9 M AFDB entries model a protein that is annotated as a homo-
oligomer; the monomer prediction is a biologically incomplete model of it. In
Swiss-Prot, where annotation is dense, more than three times as many proteins are
annotated as oligomers than as monomers.

The 99.1 M figure is smaller than AFDB's 241 M because UniProt only cross-
references entries that are still current; AFDB retains accessions that have since
been merged or deleted.

### The other four collaboration datasets

`atbc`, `bfvd`, `ntdx` and `vr3d` are monomer chunk-tar collections in the same
layout as the main release (`models/` plus usually `msas/`), several hundred MB
per chunk. `bfvd` is the Big Fantastic Virus Database and `atbc` is almost
certainly AllTheBacteria; `ntdx` and `vr3d` are not identified in any README on
the FTP site.

## ESM Atlas

**No multimers, and none hiding.** `s3://esm-protein-atlas/` (us-west-2, anonymous
read) holds two fold datasets:

| dataset | rows |
|---|---|
| `v1/folds/folds_1B.lance` | 1,095,530,880 |
| `v1/folds/folds_atlas.lance` | 6,600,755 |

Together 75.7 TB across 110,241 objects. Both schemas carry exactly one `sequence`
and one `structure_blob` per row, with no chain column. ESMFold is a single-chain
model, so this is structural, not incidental.

The interesting wrinkle is that **the on-disk structure format is multimer-capable
and the capability is unused**. Decoding a `structure_blob` (brotli → msgpack-numpy)
gives an AlphaFold3-style record:

```
atom37_positions, atom37_mask, atom37_confidence, confidence,
residue_index, insertion_code, sequence, metadata,
chain_id, entity_id, sym_id, chain_boundaries      <- multimer fields
```

`chain_id`, `entity_id`, `sym_id` and `chain_boundaries` are exactly the fields you
need to represent a complex. To check whether any row uses them I sampled 800 rows
from each dataset at eight evenly spaced offsets (contiguous windows are unbiased
here because both tables are keyed on a content hash). **All 1,600 structures had
exactly one distinct `chain_id`, one `entity_id`, one `sym_id`, and
`len(chain_boundaries) == 1`.** No chain-break characters or poly-glycine linkers in
the sequences either.

The cluster annotations are no help: `v1/clusters/data/representative_proteins.parquet`
(7,723,579 rows) carries Pfam domains, taxonomy, product names and UniRef matches;
no interaction, complex, or stoichiometry field.

### The one multimer-relevant signal: genomic neighbourhood

The Atlas's provenance index is **6,824,676,938 rows** across 16 parquet files, with
schema `(protein_hash, source, accession)`. The accessions turn out to encode
genomic position, which is the classic operon signal for physical interaction.
Sampling 1.2 M provenance rows:

| source | share | accession format | positional? |
|---|---|---|---|
| `spire` | 46.3% | `study\|sample\|k141_<contig>_<orf_index>` | yes: contig + ORF index |
| `mgy` | 35.6% | `MGYP003140358282` | no |
| `uniparc` | 10.2% | `UPI001AF2A499` | no |
| `img_m` | 2.9% | `2809785504` | no |
| `umag_prok` | 2.9% | `Ga0209614_10155761`, `Ga0590176_00001086_160893_161660` | yes: scaffold + coordinates |
| `img_vr` | 1.6% | `Ga0309785_1141497` | yes: scaffold + gene index |
| `uhgg` | 0.5% | `MGYG000238883_01394` | yes: genome + gene number |
| `umag_euk` | 0.14% | `Ga0496077_00007_1026162_1026983` | yes: scaffold + coordinates |

All 555,846 sampled SPIRE accessions parse; **74.1% have ORF index ≥ 2**, meaning
they have at least one upstream neighbour on the same contig, and the largest ORF
index seen was 1894. Roughly half the Atlas is therefore addressable as
gene-neighbourhood pairs, at a scale (hundreds of millions of adjacent ORF pairs)
that nothing else in this report approaches.

To be clear about what this is and is not: the Atlas contains **zero multimer
structures**. What it contains is an enormous set of *candidate interacting pairs*
with a folded monomer already available for each partner. Turning that into
structures would mean running a complex predictor ourselves.

## PDB: the experimental baseline

Measured with `probe_pdb.py` (faceted count queries against the RCSB search API;
kilobytes transferred, no structures downloaded). CSVs in `data/`.

Of 257,179 experimental entries, **359,783 assemblies contain at least one protein
chain**, and of those:

| | assemblies |
|---|---|
| single protein chain | 159,411 |
| **multiple protein chains** | **200,372** (55.7%) |
| ...homomeric (1 distinct entity) | 117,300 |
| ...heteromeric (≥ 2 distinct entities) | 83,072 |
| ...containing nucleic acid too | 12,588 |

Chain-count distribution is dominated by dimers (105,626 two-chain assemblies),
then tetramers (29,705) and trimers (22,423), with a long tail out to 300+ chains.
Point-group symmetry mirrors this: C2 94,330, C1 74,019, D2 14,448, C3 13,318.

By method, cryo-EM is where the multimers are: **92% of EM entries are multi-chain**
(32,751 of 35,608) and 18,942 have six or more chains, versus 58% for X-ray
(117,860 of 202,794) and 14% for NMR.

**Redundancy is the real story.** 477,774 protein entities participate in a
multi-chain assembly, but they collapse hard:

| sequence identity | distinct clusters |
|---|---|
| 100% | 105,440 |
| 95% | 81,979 |
| 90% | 75,050 |
| 70% | 55,313 |
| 50% | 46,147 |
| **30%** | **37,124** |

So the PDB's 200 k multimeric assemblies represent roughly **37 k distinct protein
families**. Growth is steady but not fast: 11,795 multi-chain protein entries were
released in 2025 (6,145 of them heteromeric), and 6,943 so far in 2026.

RCSB also mirrors 1,062,058 computed structure models (AFDB and ModelArchive
deposits). Only **2,063 are multi-chain**, all heteromeric, being the ModelArchive
complex deposits.

## Other databases

### PDB-derived interface datasets, and a useful convergence

Two curated redundancy-reduced interface sets are worth knowing, mostly because
they independently corroborate the scale I measured against RCSB:

- **[PINDER](https://doi.org/10.1101/2024.07.17.603980)** (2024): 2,319,564 dimeric
  PPI systems mined from the PDB, split by structural clustering into a training
  set of **1,560,682 dimers drawn from 42,220 clusters**, a 1,958-representative
  validation set, and 1,955 high-quality test PPIs with interface leakage removed
  (plus a 180-dimer subset clean with respect to AlphaFold-Multimer's training
  data). Notably it already pairs 566,171 of its systems with AFDB monomer
  structures, which is exactly the apo/holo bridge we would otherwise have to
  build.
- **[PPIRef](https://github.com/anton-bushuiev/PPIRef)** (ICLR 2024): PPIRef300K is
  322,454 biophysically valid interfaces; deduplicating with the `iDist` interface
  similarity measure leaves **PPIRef50K at 45,553 non-redundant interfaces**.

Put next to my own RCSB census, three different methods land in the same place:

| Method | Non-redundant multimer units |
|---|---|
| RCSB sequence clustering at 30% identity (this report) | 37,124 |
| PINDER structural interface clustering (training clusters) | 42,220 |
| PPIRef `iDist` interface deduplication | 45,553 |

**Roughly 40,000 distinct protein-protein interfaces exist in the PDB**, whichever
way you count. That is the number to hold in mind: it is what all the experimental
multimer data in the world reduces to, and it is about four orders of magnitude
smaller than our monomer corpora.

### ModelArchive: the established home of predicted complexes, and it is small

ModelArchive is where the AlphaFold-Multimer interactome screens have historically
been deposited. Crawled via its API today: **625,966 models across 3,258
depositions**, of which only **~11,879 are confirmed multi-chain (1.9%)**. A
further 1,490 older single-model deposits carry no chain metadata; extrapolating
those gives ~12,600 complexes, but that number is an estimate.

The complexes are concentrated in about 25 deposits:

| Deposit | Models | Chains | Scope | Paper |
|---|---|---|---|---|
| `ma-bak-evip` | 3,203 | 2 | human bacterial pathogens (RoseTTAFold2-Lite) | [Nat Microbiol 2024](https://doi.org/10.1038/s41564-024-01791-x) |
| `ma-drew-dc2` | 2,573 | 2 | human, **AlphaFold3** (AF3 terms, not CC) | no DOI recorded |
| `ma-bak-cepc` | 1,106 | 2 | *S. cerevisiae* | [Science 2021](https://doi.org/10.1126/science.abm4805) |
| `ma-t3vr3` | 957 | 2 | human cancer interactome | [Protein Sci 2022](https://doi.org/10.1002/pro.4479) |
| `ma-low-csi` | 929 | 2 | human PPI dimers from XL-MS | [PNAS 2023](https://doi.org/10.1073/pnas.2219418120) |
| `ma-dm-prc` | 742 | 2 | human Polycomb PRC1/2 | [Sci Adv 2024](https://doi.org/10.1126/sciadv.adl4529) |
| `ma-dm-hisrep` | 268 | 2–7 | replisome H3-H4 chaperone | [Cell 2024](https://doi.org/10.1016/j.cell.2024.07.006) |

Two things to note. Every large deposit is **pairwise**; even Humphreys et al.
2021, whose paper describes assemblies of up to five subunits, deposited only
2-chain models. And the license is **CC BY-SA 4.0**, with the ShareAlike clause,
which is stricter than AFDB's CC BY and would propagate to anything we
redistribute. Bulk access is per-deposit ZIP only; there is no archive-wide dump.

RCSB mirrors 69,326 ModelArchive models, and my own RCSB census independently
agrees on the complex count: 2,063 multi-chain computed structure models, all
heteromeric, corresponding to `ma-bak-cepc` (1,106) plus `ma-t3vr3` (957).

### BFMD: the aggregated predicted-multimer database

The [Foldseek-Multimer paper](https://doi.org/10.1038/s41592-025-02593-7) (Nature
Methods, 2025) assembled the **Big Fantastic Multimer Database**, which "organizes
297,570 multimer predictions from community efforts into a single database". It is
served as a searchable Foldseek database (`bfmd`, version 20240623, the only
complex-capable database on the Foldseek server besides PDB100). This is the
closest thing to a curated, deduplicated aggregate of the predicted-complex
literature, and at ~300 k it is about 1% the size of the new AFDB dimer release.

## Foldseek-Multimer smoke test

Run on 2026-08-02 against Foldseek `718d4217` to check that the AFDB complex files
can be clustered at all before anyone plans a run over the ~2 M confident models.
Two scripts, both capped and both cheap:

- **`fetch_dimer_sample.py`** pulls 30 heterodimers and 15 homodimers, both gated
  on the authors' criterion validated above, by range-reading the FTP metadata
  CSVs at ten spread-out offsets and downloading each model from
  `https://alphafold.ebi.ac.uk/files/AF-<id>-model_v1.cif`. **45 structures,
  21.1 MB of mmCIF, 12.5 s, zero failures** (the transfer is about a quarter of
  that, since the script asks for gzip and a sample of ten models compressed
  4.1x). `MAX_STRUCTURES = 250` raises rather than truncates, so the script
  cannot be turned into a bulk downloader by accident.
- **`smoke_test_foldseek_multimer.py`** runs `easy-multimercluster` and
  `easy-multimersearch` over that sample.

Why the multimer commands and not `easy-cluster`: `easy-cluster` clusters
*chains*, so two complexes assembled from the same two folds land together no
matter how differently the chains are arranged against each other. Redundancy that
matters for training is a property of the interface, which is what
`easy-multimercluster` scores through `--multimer-tm-threshold`,
`--chain-tm-threshold` and `--interface-lddt-threshold`.

Foldseek read all 90 chains from the 45 ModelCIF files with no parse warnings, so
AFDB complex files need no conversion step, no PDB rewrite and no chain splitting.
Both commands finished in about 3 seconds.

| | |
|---|---|
| structures / chains ingested | 45 / 90 |
| clusters (`--multimer-tm-threshold 0.5`) | 40 |
| multi-member clusters | 4 (sizes 3, 2, 2, 2) |
| complex pairs in the search report | 650 |
| pairs with multimer TM ≥ 0.5 | 7 |
| pairs where both chains aligned | 199 |

The 1.12x reduction is not the result. Forty-five complexes drawn at random from a
29 M set should be almost entirely unrelated to each other, and they mostly are;
finding any structure at all in a sample this small is the surprise. What the four
multi-member clusters contain is the finding, and each one fails a different
deduplication strategy:

- **bma-lsm-6/bma-lsm-5.1, bma-snr-7/bma-snr-5, bma-lsm-6/bma-snr-6** (*Brugia
  malayi*, tax 6279), the 3-member cluster, multimer TM 0.799 to 0.819. All six
  proteins are Sm/Lsm family subunits, which assemble into a heptameric ring, so
  pairwise decomposition of one ring yields many copies of the same Sm-Sm
  interface.
- **LOC100805631/LOC100789233 with LOC100805631/LOC100796470** (soybean, tax 3847),
  multimer TM 0.876. Shared first partner, paralogous second partners.
- **GDU4/LOG2 with GDU1/LOG2** (*Arabidopsis*, tax 3702), multimer TM 0.670. GDU1
  and GDU4 are paralogous glutamine dumpers bound to the same ubiquitin ligase.
- **SNX4 homodimer (*Merluccius polli*, Benguela hake) with the
  LOC100777531/LOC100803157 heterodimer (soybean)**, multimer TM 0.619, both
  chains aligned. A sorting-nexin BAR domain dimerises into a crescent, and a
  soybean heterodimeric pair reproduces the same interface.

Three lessons follow, and they compound.

**Sequence-level filtering does not work here.** The Sm-ring cluster is the proof.
Those three complexes carry six different gene names across three unrelated-looking
pairs, so any filter keyed on gene pairs, and probably any filter keyed on sequence
identity between partners, keeps all three as distinct training examples. They are
three pairwise slices of one heptameric ring, and the interface is the same
interface. This is the multimer version of what exp41 found on monomers: 65 of 99
FoldBench structures overlapped training structurally while sitting below 30%
sequence identity. Ring-forming and filament-forming complexes make it worse than
the monomer case, because a single assembly generates many valid-looking pairs.

**The two files have to be deduplicated together.** The SNX4 cluster pairs a
homodimer from `homodimers/` with a heterodimer from `heterodimers/`, across a
fish and a legume, at multimer TM 0.62 with both chains aligned. Processing the
21.5 M homodimers and the 7.6 M heterodimers as separate jobs, which is the
obvious way to split the work given that they are separate directories with
separate metadata files, would leave that class of redundancy in place. Any
pipeline design that shards by file has to reconcile across shards at the end.

**The search report is not the clustering, and the difference is chain coverage.**
Seven pairs scored multimer TM ≥ 0.5 but only six formed clusters. The seventh is
a soybean LOC100805119/LOC547737 pair against human STK38L/MOB1A at TM 0.519,
where Foldseek aligned chain A of one against chain A of the other and nothing
else. One chain matching is a fold-level match, not an interface match, and
`easy-multimercluster` correctly declined to merge them. The
`both_chains_aligned` column in `data/foldseek_multimer_smoke_hits.csv` is what
separates the two cases: 199 of the 650 reported pairs align both chains, and only
those are candidates for interface redundancy. Filtering a search report on the
TM-score alone would have produced a wrong answer here.

One trap worth recording, because it fails silently and produces a plausible
answer. `easy-multimercluster` takes a *directory* of structures. Passing the 45
files as separate argv entries instead makes Foldseek consume all but the last one
or two as something else, and it **still exits 0**: the first run of this smoke
test clustered 2 chains and reported a clean result that read exactly like a
successful run over all 45 structures. `easy-multimersearch` fails the same way in
a different shape, silently splitting a file list into 44 queries against 1 target.

`assert_all_chains_ingested()` now parses Foldseek's own `Query database size`
line out of the log and compares it against 2 × the number of input files, raising
if they differ. It also raises when the log line is missing rather than passing
vacuously, which matters because a Foldseek version change could remove it. The
guard was tested against three failure modes: an under-ingest, a zero-structure
sample, and a log with the line absent.

Scaling this to the confident slice is the obvious next step and it is not free.
The 45-structure run says nothing useful about ~2 M, where the cost is dominated
by `createdb` over the whole set and by the ~230 GB download that has to precede
it. What the smoke test does establish is that no format conversion sits in
between, which was the open question.

## Conclusion

**Both of our current corpora are strictly monomeric, and this was verified rather
than assumed.** AFDB's 241 M main-release models are one UniProt accession and one
`_struct_asym` record each; the ESM Atlas's 1.10 B + 6.6 M models were sampled at
1,600 structures and every one had a single chain, despite a storage format that
carries `chain_id` / `entity_id` / `sym_id` / `chain_boundaries` fields ready for
complexes.

**The situation changed in 2026, and the change is large.** AFDB now carries a
four-way EMBL-EBI / DeepMind / NVIDIA / Seoul National University release of
21.5 M predicted homodimers and 7.6 M predicted heterodimers (48.8 TB),
produced with AlphaFold-Multimer v2.3.0 weights through OpenFold, scored with
ipTM/ipSAE/pDockQ, carrying UniProt cross-references for both chains, and licensed
CC-BY-4.0 for commercial use. It is roughly 100× larger than every previously
available predicted-complex collection put together. The high-confidence slice is
queryable per accession through `/api/complex/`; the bulk lives on FTP under
`collaborations/nvda/`, which is easy to miss because nothing in the main AFDB
download documentation points at it.

**Filter hard and the heterodimers shrink to something familiar in size.** At the
release's own quality gate (ipSAE ≥ 0.6 and pDockQ2 ≥ 0.23) 0.92% of heterodimers
survive, about **70 k complexes**, which lands in the same range as the ~40 k
distinct interfaces in the entire PDB and BFMD's 297 k aggregated predictions. The
homodimers hold up ten times better: **9.13%, about 1.96 M**, measured by that
same gate after checking it reproduces the authors' declared verdict on all
123,597 heterodimer rows sampled. That is a genuinely new quantity of confident
complex structure, and homodimers are also where the latent demand is: ~2.9 M AFDB
entries model proteins that UniProt annotates as homo-oligomers. Two caveats on
that 1.96 M. Which score you filter on matters more than where you put the cut
(at a fixed threshold of 0.5 the heterodimer yield ranges from 46 k by pDockQ2 to
437 k by ipTM). And it counts models, not distinct proteins: about a quarter of
confident homodimers repeat a gene name already present, because UniProt carries
one accession per strain, and structural clustering will cut deeper than that.

**Experimental multimer data is fixed at about 40,000 distinct interfaces.** RCSB
sequence clustering, PINDER's structural interface clustering and PPIRef's `iDist`
deduplication independently give 37 k, 42 k and 46 k. The PDB adds roughly 6,000
multi-chain entries a year, so this number is not going to move by an order of
magnitude. Any multimer capability has to come mostly from predicted structures,
which is what makes the AFDB release significant.

**The ESM Atlas contributes candidate pairs, not structures.** Roughly half its
provenance records (SPIRE, plus UHGG, IMG/VR and the uMAG sets) encode contig and
ORF position, and 74% of sampled SPIRE entries have an upstream neighbour on the
same contig. That is a very large supply of operon-adjacent candidate interacting
pairs, each with a folded monomer already in hand; but turning it into complex
structures means running a predictor ourselves, not downloading anything.

If we pursue multimers, the practical ordering is: the AFDB homodimer set first
(largest confident yield, permissive license, joins on UniProt), PINDER as the
evaluation anchor (it already ships leakage-controlled splits and AFDB-paired
monomers, so we would not rebuild that), then the AFDB heterodimers under their
quality gate. ModelArchive and BFMD are small enough by comparison to be rounding
errors, and ModelArchive's ShareAlike license makes it the least attractive of
the three. Whatever the order, run Foldseek-Multimer over the homodimers and
heterodimers **together**: the smoke test above found a fish SNX4 homodimer and a
soybean heterodimer sharing an interface at multimer TM 0.62, so deduplicating
the two files separately would leave that redundancy in place.

Two open questions this survey did not settle. The complex release's paper is
still listed as "to be published" (the announcement points at a preprint on
`research.nvidia.com` that I did not retrieve), so the exact pair-selection
procedure inside the 20 species plus WHO-pathogen scope is not documented. And
`ntdx/` and `vr3d/`, two of AFDB's five collaboration datasets, carry no README
and remain unidentified.

## Curated validation set (in progress)

The survey above recommended the `collaborations/nvda/` release. Turning that
recommendation into an actual held-out set is a separate, longer piece of work
tracked in **[`CURATION_PLAN.md`](CURATION_PLAN.md)**, which follows PINDER's
split protocol. Where it stands:

- **Phases 0 to 2 are done.** The val-eligible side has been downloaded:
  **880,248 structures, 103.7 GiB**, at
  `s3://marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/val/` as 441 tar shards
  of 2,000 gzipped mmCIFs. 762 of the 881,010 requested model IDs do not exist in
  the release; they are listed in `data/curation/fetch_missing_404.csv` and must
  be excluded downstream. Per-shard stats are in
  `data/curation/fetch_shard_summary.csv`; the 880k-row per-input timings CSV is
  too large for git and sits beside the shards as `fetch_timings.csv.gz`.
- **Phase 3 is done.** A single `foldseek easy-cluster` at TM >= 0.5 over 857,599
  chains gives **13,478 clusters** (58.0 min, 4.12 GB peak RSS), on top of a 37.1
  min parallel extraction. PINDER's chain graph and AsynLPA communities were
  measured and rejected: 14x the cost, and their transitivity is wrong for a
  deduplicated validation set. The corpus is far more redundant than the pilot
  implied (1.57% representatives against 21.1%), and the 145 largest clusters hold
  half the chains. Labeling complexes by their chain-cluster pair gives **17,083
  distinct pairs**, which caps the val set. See
  **[`CLUSTERING_REPORT.md`](CLUSTERING_REPORT.md)**.
- **Phases 4 and 5 are done.** Labeling every complex by its chain-cluster pair
  gives **17,083 distinct pairs**; measuring interface residues for the top three
  candidates of each (31,956 structures, one 486 s pass over the shards) and
  drawing the best that clears 7 residues on both sides gives a validation set of
  **17,067 dimers**, 40.8% of them heterodimers against 6.4% in the corpus. The
  interface filter removes 0.08% of candidates and costs 16 pairs. The set is in
  `data/curation/val_set_dimers.csv`. It is not confidence-gated: ipTM runs
  0.099 to 0.970, median 0.768, with 22.3% below 0.5, mostly pairs whose single
  member is a weak prediction.
- **Phase 6 is open**: publish to the HF bucket under `data/afdb-nvda-dimers/`.

The AWS procedure, including how to run headless in an account with no SSH key
pairs, is in **[`AWS_FETCH.md`](AWS_FETCH.md)**.

## Reproducing the numbers

- `probe_pdb.py`: the PDB census. `python3 probe_pdb.py --out data`. Faceted
  count queries only; writes the CSVs in `data/`.
- `profile_confidence.py`: the confidence-gate tables.
  `python3 profile_confidence.py --windows 200 --window-bytes 200000 --out data`.
  Range requests only, ~80 MB total, about two minutes. Writes
  `nvda_confidence_gates.csv` (every metric at every threshold, with per-window
  spread and estimated totals), `nvda_confidence_quantiles.csv` (the underlying
  distributions), `nvda_confidence_windows.csv` (one row per window, which is the
  evidence for the organism-blocking caveat) and `nvda_confidence_sampling.json`
  (row-count estimates, redundancy counts, and the gate-versus-declared check).
- `fetch_dimer_sample.py`: the capped dimer sampler.
  `python3 fetch_dimer_sample.py --n-hetero 30 --n-homo 15 --out sample`. Writes
  `sample/structures/` (gitignored) plus `sample/manifest.csv` and
  `sample/fetch_summary.json`, which are copied into `data/` as
  `dimer_sample_manifest.csv` and `dimer_sample_fetch_summary.json` so the sample
  is recorded without committing 21 MB of coordinates. `--gate iptm` swaps the
  release's ipSAE/pDockQ2 criterion for a plain ipTM cut.
- `smoke_test_foldseek_multimer.py`: the Foldseek run.
  `python3 smoke_test_foldseek_multimer.py --sample sample --out data`. Needs no
  preinstalled Foldseek; `foldseek_env.py` downloads a static build into
  `~/.cache/marinfold/foldseek` on first use.
- AFDB and ESM Atlas numbers were read interactively; the commands are quoted
  inline above (FTP directory listings, range requests against the metadata CSVs,
  and `lance` / `pyarrow` reads against `s3://esm-protein-atlas` with
  `storage_options={"aws_skip_signature": "true", "region": "us-west-2"}`).
