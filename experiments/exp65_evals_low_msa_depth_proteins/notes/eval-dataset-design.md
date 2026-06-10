# Designing a leakage-free evaluation set for MarinFold

> Research notes + concrete recommendation for building a held-out test
> dataset for MarinFold (a from-scratch LLM that predicts CB-CB contacts
> and distograms from **single** protein sequences, monomers only,
> trained on the AlphaFold Database).
>
> Status: design notes, not yet implemented. Compiled 2026-06-03 from a
> multi-source literature sweep (19 primary/secondary sources, 23
> adversarially-verified claims). Confidence and refutations are flagged
> inline. Anchor every temporal claim to **MarinFold's own AFDB snapshot
> date**, not to dates quoted in the literature (see caveats).

---

## 1. The core problem

MarinFold trains on AFDB, and AFDB is a structural mirror of UniProt.
The training corpus therefore "knows" essentially every natural sequence
that existed at the snapshot AFDB was built from, represented as an
AlphaFold *prediction*. A naive "held-out PDB" test set is **not** held
out: the test sequence (or a close homolog) is almost certainly already a
training document, just labelled with a predicted structure instead of an
experimental one.

So for MarinFold "unseen" must hold along **three independent axes at
once**, because passing only one still leaks:

1. **Sequence novelty** — no close homolog of the test sequence in the
   AFDB training set.
2. **Structure / fold novelty** — the fold is absent or rare in training.
3. **Temporal novelty** — the structure (ideally also its sequence)
   postdates the AFDB snapshot, so it cannot have been a training doc.

The consistent finding across AlphaFold2's own "recent PDB" set,
FoldBench, OpenFold, and ProteinBench is that rigorous benchmarks
**layer** these filters rather than trusting any single one. That layering
is the backbone of the recommendation in §5.

---

## 2. Candidate datasets

### Experimental ground-truth sources

| Dataset | Link | What it is | Size | Access / license | Best eval question |
|---|---|---|---|---|---|
| **PDB** | [rcsb.org](https://www.rcsb.org/) · [search API](https://search.rcsb.org/) | Experimental structures (X-ray, cryo-EM, NMR) | ~220k entries | Public domain (CC0) | Primary ground truth; filter by deposition date for temporal holdout |
| **PISCES** | [dunbrack.fccc.edu/pisces](https://dunbrack.fccc.edu/pisces/) | Pre-culled non-redundant PDB chain lists at chosen %id / resolution / R-factor | configurable | Free, web/CLI | Off-the-shelf non-redundant monomer pool |
| **CAMEO** | [cameo3d.org](https://www.cameo3d.org/) | Weekly, blind, automated CASP-complement targets | ~hundreds/yr, continuous | Public | Living, temporally-honest test stream |
| **CASP14 / CASP15** | [casp14](https://predictioncenter.org/casp14/) · [casp15](https://predictioncenter.org/casp15/) | Expert-curated blind targets incl. free-modeling (FM) | ~dozens each | Public | Hard, cross-paper-comparable reference points |
| **CATH** | [cathdb.info](https://www.cathdb.info/) | Hierarchical class/architecture/topology/homology fold classification | whole PDB | Public | **Stratification labels** for novelty tiers (not a data source) |
| **SCOP** | [scop.mrc-lmb.cam.ac.uk](https://scop.mrc-lmb.cam.ac.uk/) | Structural classification (class/fold/superfamily/family) | whole PDB | Public | Alt fold labels for stratification |
| **ECOD** | [prodata.swmed.edu/ecod](http://prodata.swmed.edu/ecod/) | Evolutionary classification of protein domains | whole PDB | Public | Fold labels used by ATLAS's non-redundancy |
| **ATLAS** | [dsimb.inserm.fr/ATLAS](https://www.dsimb.inserm.fr/ATLAS/) | All-atom MD ensembles, ECOD-fold-non-redundant chains | 1,390 chains | Free, open-access | Flexibility/dynamics probe; also a clean non-redundant chain list |

### Reference / training-side

| Dataset | Link | Size | Role |
|---|---|---|---|
| **AlphaFold DB (AFDB)** | [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/) | ~200M+ predicted structures (full DB) | The public AFDB. Your training data derives from it. |
| **`afdb-1.6M` (ours)** | [hf: timodonnell/afdb-1.6M](https://huggingface.co/datasets/timodonnell/afdb-1.6M) | ~1.6M structures (name) | The exact curated subset MarinFold trains on. **Dedup test candidates against this**, not UniProt. |
| **`protein-docs` (ours)** | [hf: timodonnell/protein-docs](https://huggingface.co/datasets/timodonnell/protein-docs) | ~1.6M documents × N layouts | The tokenized training documents (post document-structure). |
| **UniProt / UniRef** | [uniprot.org](https://www.uniprot.org/) · [uniref](https://www.uniprot.org/help/uniref) | ~250M (UniProtKB/TrEMBL); UniRef50 ~70M clusters | Sequence universe AFDB mirrors. Clustering context only; see the 631/20,504 mismatch caveat in §4. |

### Modern benchmark suites worth reusing

- **FoldBench** (2025) — [paper](https://www.nature.com/articles/s41467-025-67127-3)
  — **1,522 biological assemblies** (monomer subset is a few hundred);
  for monomers it applies a temporal cutoff **plus** sequence-similarity
  filtering **plus** a Foldseek structure filter at **TM-score < 0.5**.
  The closest published template to what MarinFold needs, and you already
  use its monomer subset (exp12/exp20/exp26/exp27).
- **ProteinBench** (2024) — [site](https://proteinbench.github.io/) ·
  [paper](https://arxiv.org/html/2409.06744v2) — single-state
  protein-folding track that explicitly controls for data leakage; folding
  eval set ~**332 CAMEO complexes** (Jan–Jul 2024) plus RFdiffusion
  backbones at lengths 100–500.
- **OpenFold** (2022) — [code](https://github.com/aqlaboratory/openfold) ·
  [paper](https://www.biorxiv.org/content/10.1101/2022.11.20.517210.full.pdf)
  — the canonical study on fold-space generalization. Filters to
  **~440,000 CATH-classified domains spanning 1,385 topologies, 42
  architectures, 4 classes** (verified from the paper's full text), then
  holds out entire Topologies (T), Architectures (A), or Classes (C);
  the T-split samples 100 held-out topologies.

---

## 3. Splitting / leakage-control techniques

### (a) Sequence-similarity splitting

- **Tools:** MMseqs2 (`--min-seq-id`), CD-HIT, BLAST.
- **Standard thresholds:**
  - **< 30% identity** = "twilight zone", the classic *novel family*
    boundary used throughout the field.
  - **40% identity** = the stricter value AF2 used for its template
    filter on the recent-PDB set.
- MMseqs2 is the practical default at scale: it clusters/searches
  hundreds of millions of sequences and is the tool used to build AFDB's
  own clustering.

### (b) Structure-similarity splitting

- **Tools:** Foldseek (3Di alphabet, ~4,000× faster than TM-align,
  scales to all of AFDB), TM-align, DALI.
- **Standard threshold:** **TM-score < 0.5** is the canonical
  same-fold / different-fold boundary. ≥ 0.5 ⇒ same fold. This is the
  exact line FoldBench uses for monomer structure filtering.
- **Fold-level holdout:** OpenFold's stricter method holds out whole
  **CATH topologies**, and found that removing fold-space hurts accuracy
  *more* than removing sequence-similar examples. This is the empirical
  reason to stratify by fold, not just sequence.

### (c) Temporal splitting

- **The CAMEO/CASP model:** evaluate only on targets released *after*
  the model's training cutoff. CAMEO automates this continuously; CASP
  does it episodically with expert FM targets.
- **AF2's recipe:** temporal cutoff **layered with** a 40% template-
  identity filter — a direct demonstration that a date alone is
  insufficient.
- **For MarinFold:** the cutoff is **your AFDB snapshot date**, full
  stop. See caveats — do *not* hard-code 2018 or 2021.

### How the major models actually split

- **AlphaFold2:** "recent PDB" held-out set = temporal cutoff + 40%
  template identity filter.
- **OpenFold:** CATH-topology holdout to measure fold-space
  generalization.
- **FoldBench:** temporal + sequence + Foldseek-TM < 0.5, layered.
- **ProteinBench:** leakage-controlled single-state folding split.

---

## 4. The AFDB/UniProt-specific leakage problem

Because AFDB covers ~all of UniProt, the usual "hold out recent PDB"
move is necessary but **not sufficient** — a recent experimental
structure can still have an old, AFDB-covered sequence.

Key points from the research:

- **Dedup against what you trained on, not UniProt.** Of the 20,504
  full-length human AFDB models, **631 conflict with their UniProt
  sequence** — AFDB is not a perfect copy of UniProt. Run MMseqs2 /
  Foldseek against the exact `afdb-1.6M` documents MarinFold ingested.
- **AFDB structures were themselves derived using PDB templates,** so
  fold information from old PDB entries is doubly baked into your
  training data (once as the AF prediction, once via templates). A
  fold that is "old" in PDB terms is effectively memorized even if the
  specific test chain is new.
- **Recommended practice for models trained on predicted structures:**
  treat the snapshot date as the temporal origin, then *additionally*
  dedup test candidates by both sequence and structure against the
  training set. Report a redundant-vs-novel breakdown so reviewers can
  see the memorization ceiling.

### ⚠️ Refuted / unconfirmed date claims

Two specific date claims **failed adversarial verification (1–2 votes)**
and must not be hard-coded:

- ❌ "AFDB = ~200M structures from UniProt as of April 2021." Unconfirmed.
- ❌ "AF2's canonical cutoff is 30 April 2018, so test structures must
  postdate it." Unconfirmed as authoritative.

Treat **any** calendar date quoted in the literature as unreliable.
The only date that matters for MarinFold's leakage argument is the
release of the `afdb-1.6M` snapshot you actually trained on.

---

## 5. Recommended pipeline

```
1. CANDIDATE POOL
   Experimental monomer chains from PDB deposited AFTER your AFDB snapshot.
   Filter: single chain, ~40–500 residues, resolution ≤ 3.0 Å.

2. TEMPORAL FILTER            (axis 3 — strongest single guarantee)
   Keep deposition date > snapshot date.

3. SEQUENCE DEDUP vs TRAINING (axis 1)
   MMseqs2 search each candidate vs afdb-1.6M sequences.
   Drop if best hit ≥ 30% identity over ≥ 50% coverage.

4. STRUCTURE DEDUP vs TRAINING (axis 2)
   Foldseek each candidate vs training structures.
   Drop if best Foldseek TM-score ≥ 0.5.

5. INTERNAL REDUNDANCY
   Cluster survivors (MMseqs2 @ 30%); keep one representative per cluster
   so big families don't dominate.

6. STRATIFY into reporting tiers (§6) + hard/special subsets (§7).
```

### Tooling commands (starting points)

```bash
# --- Sequence dedup: build a searchable DB of training sequences once ---
mmseqs createdb afdb_1_6M.fasta trainDB
mmseqs createdb candidates.fasta candDB
mmseqs search candDB trainDB resultDB tmp --min-seq-id 0.3 -c 0.5 -s 7.5
mmseqs convertalis candDB trainDB resultDB hits.tsv
# any candidate appearing in hits.tsv (>=30% id, >=50% cov) is dropped

# --- Internal redundancy reduction among survivors ---
mmseqs easy-cluster survivors.fasta clustered tmp --min-seq-id 0.3 -c 0.5
# keep one representative per cluster (clustered_rep_seq.fasta)

# --- Structure dedup: Foldseek all-vs-all candidate-vs-training ---
foldseek createdb training_structures/ trainStructDB
foldseek createdb candidate_structures/ candStructDB
foldseek search candStructDB trainStructDB aln tmp -a --alignment-type 1
foldseek convertalis candStructDB trainStructDB aln aln.tsv \
    --format-output query,target,alntmscore,qtmscore,ttmscore
# drop candidates whose best alntmscore >= 0.5
```

(Parameters are conservative defaults; tune `-s` sensitivity and coverage
mode to taste, and verify Foldseek's TM-score output column on your
installed version.)

---

## 6. Generalization tiers (axis #2 deliverable)

Report accuracy **per tier**. The aggregate number is misleading for an
AFDB-trained model; the **Tier 0 → Tier 2 gap is the headline metric** —
it quantifies folding vs retrieval.

| Tier | Definition | What it measures |
|---|---|---|
| **0 — Redundant** | close sequence homolog in training (≥ 30% id) | Memorization ceiling / sanity check |
| **1 — Novel family** | < 30% seq id to training, but fold present (Foldseek TM ≥ 0.5) | Structural-pattern transfer |
| **2 — Novel fold** | < 30% id **and** best Foldseek TM < 0.5 | **True generalization** |
| **3 — Hard/special** | de novo, orphans, disorder, CASP/CAMEO FM | Honest worst case |

For the strictest published variant, also report a **CATH-topology
holdout** (OpenFold-style): exclude every chain whose CATH topology
appears in training. OpenFold did exactly this — sampling 100 held-out
topologies (and analogously whole architectures / classes) from its
~440k-domain CATH set — and found accuracy degrades progressively as you
hold out larger units of fold space (T < A < C). Holding out at the
**homologous-superfamily (H)** level is the structural analogue of
sequence-redundancy reduction; holding out at **T/A/C** is the genuine
fold-novelty test.

---

## 7. Hard / special-case subsets

- **De novo designed proteins** — no evolutionary homologs at all; the
  purest single-sequence generalization test for a model that uses no
  MSA. **Expanded in detail in §7.1.**
- **Orphan / singleton sequences** — UniRef singletons; sparse
  evolutionary signal.
- **Intrinsically disordered regions** — high-disorder chains stress a
  contact/distogram model that assumes a single folded state.
- **CASP/CAMEO free-modeling targets** — community-standard hard set,
  temporally honest by construction.

---

## 7.1 Designed / unnatural proteins (priority subset)

Designed proteins are the single best generalization probe for MarinFold,
for three reasons:

1. **No MSA, no homologs.** A de novo sequence has essentially no
   evolutionary neighbours, so MarinFold's single-sequence setting is on
   exactly equal footing with MSA-based models here (a rare apples-to-
   apples comparison). Co-evolution-based methods lose their main signal;
   a single-sequence model does not.
2. **Often genuinely novel folds.** Many designs occupy regions of fold
   space absent from nature, so they land in Tier 2/3 automatically.
3. **Mostly outside AFDB's training universe.** AFDB mirrors UniProt;
   synthetic designs are largely *not* in UniProt, so they are plausibly
   unseen by an AFDB-trained model. Still verify with the §5 dedup — some
   designs do get UniProt accessions, and designs based on natural
   scaffolds can be sequence/structure-close to training.

### A crucial distinction: "structure-validated" vs "stability-validated"

Two very different kinds of "designed protein dataset" exist, and they
support different eval questions:

- **Structure-validated** — an experimental 3D structure (X-ray / cryo-EM
  / NMR) was actually solved. Gives a true experimental CB-CB ground
  truth. Use these exactly like natural PDB targets.
- **Stability-validated only** — hundreds of thousands of designs were
  screened for *folding stability* (proteolysis / cDNA-display), but no
  experimental structure was solved. The only "structure" available is
  the **design model** (the intended backbone). Here the eval question
  shifts to: *does MarinFold reproduce the intended design model?* — a
  designability check, not a match-to-experiment check. Label these
  clearly and report them separately; do not mix design-model targets
  into the experimental-accuracy headline.

### Datasets

| Dataset | Link | Size | What it is | Ground truth | Notes for MarinFold |
|---|---|---|---|---|---|
| **PDB `DE NOVO PROTEIN` class** | [RCSB search](https://www.rcsb.org/) · [search API](https://search.rcsb.org/) | **~2,007 entries** (verified via API, 2026-06-03); monomer/length-filtered subset likely ~1,000–1,500 | PDB entries classified as de novo designed | **Experimental** (X-ray/cryo-EM/NMR) | The canonical pool of structure-validated designs. Reproducible query below. |
| **RFdiffusion** (Watson et al. 2023) | [Nature](https://www.nature.com/articles/s41586-023-06415-8) · [code](https://github.com/RosettaCommons/RFdiffusion) · [data (figshare)](https://figshare.com/s/439fdd59488215753bc3) · [weights](http://files.ipd.uw.edu/pub/RFdiffusion/) | Hundreds of designs (in silico + experimental); figshare holds design structures + AF2 models + measurements; 1 cryo-EM complex deposited (PDB 8SK7, HA binder) | Designed symmetric assemblies, metal-binders, binders | **Experimental** (small deposited subset) + design/AF2 models | Most "structures" here are *design models / AF2*, not crystallography. Verified data link is **figshare, not Zenodo** (the earlier Zenodo/8K7Z attribution was a conflation). |
| **RFdiffusion antibodies** (Bennett et al. 2024/25) | [Nature](https://www.nature.com/articles/s41586-025-09721-5) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.14.585103.full.pdf) | A handful of cryo-EM structures (e.g. PDB 9NFU, 9NH7; EMD-49373, EMD-49405) | De novo VHH/antibody designs, cryo-EM validated, "highly dissimilar from any known PDB structure" | **Experimental** (cryo-EM) | Hard, very-novel; antibody-specific geometry. |
| **Megascale stability** (Tsuboyama & Rocklin 2023) | [Nature](https://www.nature.com/articles/s41586-023-06328-6) · [Zenodo 7992926](https://zenodo.org/records/7992926) | **331 natural + 148 de novo** domains, 40–72 aa; ~776k high-quality stabilities (1.8M raw); AF model PDBs ~14 MB zip | Mega-scale folding-stability screen | **Stability** + AlphaFold model PDBs + design blueprints | Coordinates are *models*, not experiments. Great designability/short-domain probe; keep separate. |
| **Rocklin 2017 miniproteins** | [Science](https://www.science.org/doi/10.1126/science.aan0693) · [Baker Lab PDF](https://www.bakerlab.org/wp-content/uploads/2017/12/Science_Rocklin_etal_2017.pdf) | 15,000+ designed miniproteins (~40–43 aa); **>2,500 stable** across 4 idealized folds (ββαβ, βαββ, αββα, ααα) | High-throughput design + proteolysis screen | **Stability** + design models | Classic; small idealized topologies. Design-model eval only. |
| **Well-folded de novo set** (Goverde et al. 2022) | [PMC9581288](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9581288/) | 518 de novo monomers (diverse folds) + 2,112 binders | Designs re-analyzed with AF2/RoseTTAFold + fragment-quality | Design models / predicted | Useful curated list of designed monomers and their predicted reliability. |
| **PDB-Struct benchmark** | [arXiv 2312.00080](https://arxiv.org/abs/2312.00080) | Curated subsets (CATH test split + de novo + megascale mutagenesis); order 10²–10³ per subset | Benchmark with refoldability + stability metrics | Mixed | Reusable splits + metric ideas tailored to designed proteins. |
| **PDFBench** (2025) | [arXiv 2505.20346](https://arxiv.org/html/2505.20346) | Benchmark spec (4 tasks / 16 metrics); sample sets order 10²–10³ | De-novo-design-from-function benchmark | Mixed | Newer; more function-design oriented but tracks designed-protein eval practice. |
| **Foldit / citizen-science designs** | [PDB-101 MotM 259](https://pdb101.rcsb.org/motm/259) · e.g. [6MSP](https://www.rcsb.org/structure/6MSP) | Dozens of solved entries (subset of the de novo class) | Crowd-designed, experimentally solved novel folds | **Experimental** | Subset of the PDB de novo class; some genuinely new folds. |

### Reproducible PDB query for structure-validated designs

```bash
# Count / list all experimentally-solved "DE NOVO PROTEIN" entries
curl -s 'https://search.rcsb.org/rcsbsearch/v2/query' \
  --data-urlencode 'json={
    "query":{"type":"terminal","service":"text","parameters":{
      "attribute":"struct_keywords.pdbx_keywords",
      "operator":"contains_phrase","value":"DE NOVO PROTEIN"}},
    "return_type":"entry",
    "request_options":{"return_counts":true}}' -G
# -> total_count: 2007  (verified 2026-06-03)

# Swap "return_counts" for "return_all_hits":true to get the ID list,
# then add a deposition-date range terminal (rcsb_accession_info.deposit_date)
# to enforce your AFDB-snapshot temporal cutoff, and an entity-polymer
# length filter for monomers.
```

To enforce the §5 pipeline on these: take the ID list, pull mmCIFs,
apply the temporal cutoff (deposition date > AFDB snapshot), run the
MMseqs2 + Foldseek dedup against `afdb-1.6M`, and tag survivors as a
**`designed`** subset that cross-cuts the Tier 0–3 stratification (most
will fall in Tier 2/3, but verify rather than assume).

### Recommended use in the eval

- **Primary designed subset:** the structure-validated pool (PDB de novo
  class + RFdiffusion/antibody deposits), filtered through §5. These get
  the same experimental-accuracy metrics as natural targets.
- **Secondary designability subset:** megascale + Rocklin design models,
  reported separately under a "match-the-design-model" framing. Beware
  these are short (40–72 aa) and idealized, so they probe a narrow,
  easy-ish slice of fold space.
- **Watch length/fold bias:** designed sets skew toward small,
  regular, helix-rich topologies. Report length- and
  secondary-structure-stratified numbers so a high designed-set score
  isn't just "MarinFold is good at short helical bundles".

---

## 7.2 Metagenomic / "dark-matter" sequences (MGnify, ESM Atlas)

MGnify is the natural counterpart to designed proteins on the *novelty*
axis: instead of synthetic sequences with no homologs, it is a vast pool
of **environmental sequences far from the well-studied UniProt core** —
"microbial dark matter". Because AFDB mirrors UniProt and the ESM
Metagenomic Atlas mirrors MGnify, large parts of MGnify are plausibly
**outside MarinFold's AFDB training universe**, which makes it attractive
for a novel-family / orphan stress test.

### ⚠️ The catch: ground truth is predicted, not experimental

This is the decisive caveat. MGnify proteins almost never have an
experimental structure. The "structures" that exist are **ESMFold
predictions** (the ESM Metagenomic Atlas). So if you score MarinFold
against MGnify "structures" you are comparing one model's output to
**another model's prediction (silver standard)**, not to experiment. That
measures *agreement with ESMFold*, not accuracy. Treat MGnify as:

- a **source of novel sequences** to mine for orphan / novel-family hard
  cases (then find the rare ones that *do* have a later experimental
  structure, and use those as real ground truth); and/or
- a **distribution-shift / silver-standard probe** clearly labelled as
  ESMFold-referenced, never mixed into the experimental-accuracy headline.

### Datasets

| Dataset | Link | Size | Ground truth | Role for MarinFold |
|---|---|---|---|---|
| **MGnify protein DB** | [EBI MGnify](https://www.ebi.ac.uk/metagenomics) · [docs](https://docs.mgnify.org/src/docs/mgnify-proteins.html) | **~2.4–2.5 billion** sequences (2024 release; ~50M at 2017 launch) | Sequence only | Mine for sequences far from AFDB/UniProt; orphan/novel-family sourcing. |
| **MGnify90** | [esmatlas.com/about](https://esmatlas.com/about) | ~600–770M representative sequences (90%-id clustered) | Sequence only | The clustered set the ESM Atlas covers. |
| **ESM Metagenomic Atlas** | [esmatlas.com](https://esmatlas.com/) · [Meta AI blog](https://ai.meta.com/blog/protein-folding-esmfold-metagenomics/) | **~772M** predicted structures (v2023_02); ~1 TB high-confidence, ~15 TB full | **Predicted (ESMFold)** — silver standard | Distribution-shift probe only; not experimental ground truth. |
| **MGnifams** | [mgnifams-demo.mgnify.org](http://mgnifams-demo.mgnify.org/) · [code](https://github.com/EBI-Metagenomics/mgnifams) | Families from the 717M-seq 2024_04 release; many novel vs Pfam | Family HMMs + predicted models | Novel-**family** definitions to stratify "never-before-seen family" tier. |

### Recommended use in the eval

1. **Orphan-sourcing:** filter MGnify for sequences with no UniRef hit,
   then intersect with **recently-deposited experimental PDB entries**
   whose sequence traces back to a metagenomic source. That intersection
   (small, but real) is gold: novel, dark-matter, *and*
   experimentally-resolved.
2. **Silver-standard distribution test:** optionally report
   MarinFold-vs-ESMFold agreement on a MGnify90 sample as a
   coverage/robustness signal, explicitly labelled as non-experimental.
3. **Family-novelty labels:** use MGnifams (or Pfam-absence) to tag a
   "novel family" flag that complements the sequence-identity and
   Foldseek-fold tiers in §6.

---

## 8. Metrics (open decision)

lDDT-Cα and TM-score are full-structure-model metrics. MarinFold emits
**CB-CB distograms / contacts**, so the natural primary metrics are:

- **Contact precision** at L/1, L/2, L/5 in the long-range band
  (|i−j| ≥ 24), the standard contact-prediction report.
- **Distogram cross-entropy** and/or **MAE** vs the true CB-CB distance
  matrix.
- Optionally derive an **lDDT-style local score from the distance map**
  for cross-paper comparability.

Decide and document this before building the set, since it dictates which
ground-truth fields you must extract (CB coordinates, masking of missing
residues, etc.). This was flagged as the main open question by the
research sweep.

---

## 9. Practical recommendations summary

1. **Combine:** post-snapshot PDB monomers + snapshot CAMEO weekly +
   CASP14/15 monomers, with CATH/ECOD labels for stratification and ATLAS
   as an optional dynamics/non-redundant pool.
2. **Filter** with the layered §5 pipeline (temporal → seq dedup →
   struct dedup → internal dedup).
3. **Thresholds:** 30% sequence identity (novel family), Foldseek
   TM < 0.5 (novel fold), 40% as the stricter alt for sensitivity checks.
4. **Stratify** into Tiers 0–3 and report per-tier; the Tier 0→2 gap is
   the key number.
5. **Anchor temporally** to MarinFold's own AFDB snapshot date.
6. **Wire CAMEO** as a recurring, auto-refreshing leakage-free stream.

---

## 10. References

Primary sources (verified, 3-0 unless noted):

- **AlphaFold2** — Jumper et al., *Highly accurate protein structure
  prediction with AlphaFold*, Nature 2021.
  https://www.nature.com/articles/s41586-021-03819-2
  (recent-PDB set = temporal + 40% template filter)
- **FoldBench** — Nat. Commun. 2025.
  https://www.nature.com/articles/s41467-025-67127-3
  (1,522 assemblies; monomer temporal + seq + Foldseek-TM<0.5)
- **OpenFold** — Ahdritz et al., bioRxiv 2022.
  https://www.biorxiv.org/content/10.1101/2022.11.20.517210.full.pdf
  (CATH-topology holdout; fold-space generalization)
- **ProteinBench** — arXiv 2024.
  https://arxiv.org/html/2409.06744v2
  (leakage-controlled single-state folding)
- **CAMEO** — continuous automated model evaluation.
  https://www.cameo3d.org/
- **ATLAS** — MD ensemble database, Nucleic Acids Res. 2024.
  https://academic.oup.com/nar/article/52/D1/D384/7438909
  (1,390 ECOD-fold-non-redundant chains)
- **Foldseek** — Nat. Biotechnol. 2023 (3Di; ~4,000× faster than
  TM-align).
  https://www.nature.com/articles/s41587-023-01773-0
  Usage examples: https://deepwiki.com/steineggerlab/foldseek/1.3-basic-usage-examples
- **MMseqs2** — Steinegger & Söding.
  https://github.com/soedinglab/MMseqs2
- **ESMFold / ESM-2** — Lin et al., Science 2023 (single-sequence,
  MSA-free; relevant precedent for MarinFold's setting).
  https://www.nature.com/articles/s41586-023-06510-w
- AFDB / UniProt mismatch + leakage discussion — bioRxiv 2025.
  https://www.biorxiv.org/content/10.1101/2025.06.22.660930.full.pdf
  (631/20,504 human AFDB models conflict with UniProt)

Additional / supporting:

- https://arxiv.org/abs/2505.22674 — benchmark datasets survey
- https://www.biorxiv.org/content/10.1101/2025.05.22.655600v1.full
- https://arxiv.org/html/2312.00080 — hard targets / tiers
- https://onlinelibrary.wiley.com/doi/full/10.1002/prot.26652 — CASP-era
  assessment
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10990103/ — special-case
  evaluation
- https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1.full

Designed / de novo proteins (see §7.1):

- **RFdiffusion** — Watson et al., *De novo design of protein structure
  and function with RFdiffusion*, Nature 2023.
  https://www.nature.com/articles/s41586-023-06415-8 ·
  code https://github.com/RosettaCommons/RFdiffusion ·
  data (design structures + AF2 models + experimental measurements) on
  **figshare**: https://figshare.com/s/439fdd59488215753bc3 ·
  weights / example scaffolds http://files.ipd.uw.edu/pub/RFdiffusion/ ·
  cryo-EM HA binder = PDB 8SK7. (Data-availability statement verified
  verbatim via EuropePMC full text; there is **no** Zenodo record for
  this paper — the earlier Zenodo reference was a conflation.)
- **RFdiffusion antibodies** — Bennett et al., Nature 2025.
  https://www.nature.com/articles/s41586-025-09721-5
- **Megascale folding stability** — Tsuboyama, …, Rocklin, Nature 2023.
  https://www.nature.com/articles/s41586-023-06328-6 ·
  data https://zenodo.org/records/7992926
- **Rocklin 2017** — *Global analysis of protein folding using massively
  parallel design, synthesis, and testing*, Science 2017.
  https://www.science.org/doi/10.1126/science.aan0693
- **Well-folded de novo proteins** — Goverde et al., 2022.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9581288/
- **PDB-Struct** (designed-protein benchmark) —
  https://arxiv.org/abs/2312.00080
- **PDFBench** — https://arxiv.org/html/2505.20346
- **PDB de novo classification** — RCSB; ~2,007 entries via the
  `struct_keywords.pdbx_keywords = "DE NOVO PROTEIN"` query (§7.1).
  https://www.rcsb.org/ · https://search.rcsb.org/ ·
  background https://pdb101.rcsb.org/motm/259

Metagenomic / dark matter (see §7.2):

- **MGnify** — EBI metagenomics resource; ~2.4–2.5B protein sequences.
  https://www.ebi.ac.uk/metagenomics ·
  docs https://docs.mgnify.org/src/docs/mgnify-proteins.html ·
  2.4B release note
  https://www.ebi.ac.uk/about/news/updates-from-data-resources/2-4-billion-sequences-now-available-in-the-latest-mgnify-protein-database-release/
- **ESM Metagenomic Atlas** — ~772M ESMFold-predicted structures over
  MGnify90 (predicted, silver standard).
  https://esmatlas.com/ · https://ai.meta.com/blog/protein-folding-esmfold-metagenomics/
- **MGnifams** — metagenomic protein families (release 2024_04, ~717M seqs).
  http://mgnifams-demo.mgnify.org/ · https://github.com/EBI-Metagenomics/mgnifams

### Verification caveats

- ❌ "AFDB ≈ 200M structures, UniProt as of April 2021" — **refuted**
  (1-2). Do not cite as fact.
- ❌ "30 April 2018 is the canonical AF2 cutoff to postdate" — **refuted**
  (1-2). Use MarinFold's own snapshot date.
- One fetched source
  (`biorxiv.org/.../2022.07.20.500902v1`) was rated **unreliable** and
  contributed no verified claims.
