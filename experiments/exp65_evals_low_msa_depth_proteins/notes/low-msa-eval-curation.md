# Curating low-MSA-depth eval datapoints for MarinFold

> Research + concrete plan for **point 1 of issue #41**: "Curating more
> low MSA-depth proteins (both natural proteins and de novo designed),"
> starting from "just saving the MSA depth for the 100 FoldBench protein
> monomers we curated in [#12]."
>
> Companion to [`eval-dataset-design.md`](eval-dataset-design.md) (the
> leakage / train-test split design) and
> [`eval-strategy-summary.md`](eval-strategy-summary.md). That pair covers
> the **structural-novelty** axis (Foldseek clustering vs the training
> set, the reusable tool for which is exp41's point 2). This doc covers
> the **orthogonal MSA-depth axis**. Compiled 2026-06-06 from a two-track
> literature sweep; load-bearing numbers are sourced inline and
> unverified items are flagged.

---

## 1. Why MSA depth is the axis that matters for MarinFold

MarinFold predicts CB-CB contacts/distograms from a **single sequence** —
no MSA at inference. MSA-based predictors (AlphaFold2, Protenix-MSA) get
their accuracy from coevolution signal in the alignment, and that signal
scales with the *number of independent homologs*. So the regime where a
single-sequence model has something to prove — and the most headroom to
win — is exactly where the MSA is **shallow**:

- AF2's own ablation: *"accuracy decreases substantially when the median
  alignment depth is less than around 30 sequences,"* and *"improvements
  in MSA depth over around 100 sequences lead to small gains"* (Jumper et
  al., Nature 2021, [PMC8371605](https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/)).
- Single-sequence models (ESMFold, OmegaFold, trRosettaX-Single, RGN2)
  are reported to **match or beat** AF2/RoseTTAFold specifically on
  orphan / low-Neff targets — e.g. trRosettaX-Single's Orphan25 set:
  mean TM 0.48 (trRX-Single) vs 0.42 (AF2) vs 0.38 (RoseTTAFold)
  (Wang et al., Nat. Comput. Sci. 2022,
  [s43588-022-00373-3](https://www.nature.com/articles/s43588-022-00373-3)).

The FoldBench-100 we currently eval on does **not** probe this regime:
Tim's MSA-depth plot on the issue shows very few proteins with depth
< 10. That is expected — FoldBench monomers are recent, well-resolved PDB
structures, which skew toward well-studied (homolog-rich) families. To
make the MarinFold-vs-Protenix comparison meaningful, we need to
**deliberately oversample the shallow-MSA tail**, which barely exists in
a depth-agnostic PDB sample.

This axis is **independent of** the structural-novelty axis in
`eval-dataset-design.md`: a protein can have a common fold but a shallow
MSA (a fast-evolving member of a known fold), or a novel fold with a deep
MSA (a large but homolog-rich family). We want both axes labelled on
every eval datapoint. See §6.

---

## 2. The metric — how to measure "MSA depth"

### 2.1 Neff, not raw count

A ColabFold MSA can contain hundreds of near-identical orthologs that
carry almost no *independent* evolutionary signal. The field-standard
depth measure down-weights that redundancy:

```
weight_i = 1 / |{ j : seqid(i, j) >= theta }|      # cluster reweighting
Neff     = sum_i weight_i
```

- **theta = 0.80** is AlphaFold2's threshold. AF2 computes per-residue
  Neff *"weighting the sequences using the Neff scheme with a threshold
  of 80% sequence identity measured on the region that is non-gap in
  either sequence"* (Jumper et al. 2021, methods).
- **theta = 0.62** also appears in the literature but belongs to the
  GREMLIN/CCMpred/DCA coevolution tradition, not the AF2/trRosetta
  folding tradition. Use 0.80 to stay comparable to AF2/Protenix. *(The
  62% figure is well-established in DCA tooling but I did not pin a
  single authoritative primary citation — flag if cited.)*
- The **entropy-based "Neff"** in HH-suite's `.hhm` header (exp of mean
  per-column sequence entropy, bounded ~0-16) is a *different* quantity
  on a different scale. Do **not** mix it with clustering Neff in the
  same table.

### 2.2 Length normalization — report all three

| Measure | Formula | Comparable to |
|---|---|---|
| `neff` | sum of cluster weights (theta=0.8) | DeepMSA; absolute depth |
| `neff_per_L` | Neff / L | **AF2 per-residue depth** |
| `neff_per_sqrtL` | Neff / sqrt(L) (trRosetta's `Nf`) | trRosetta contact work |
| `n_seqs` | raw aligned-sequence count | AF2's "median alignment depth" figure |

Raw `n_seqs` is kept because "MSA depth" is used loosely for it in the
literature (and it is what Tim's existing plot likely shows) — but the
**primary curation signal is Neff** (or Neff/L). Reporting all three lets
us reconcile with any depth number anyone else quotes.

### 2.3 Thresholds for "low" (graded, not a single cut)

| Tier | Cut | Provenance |
|---|---|---|
| **Orphan** | Neff ~= 1 | true single-sequence; trRX-Single Orphan25 ("no homologs in DB") |
| **Low** | **Neff < 10** | trRosettaX-Single's **Orphan54** set is defined as "effective number of homologous sequences < 10" — a ready-made, citable cut |
| **Marginal** | 10 <= Neff < 30 | AF2 accuracy "decreases substantially" below ~30 (depth) |
| **Deep** | Neff >= 30 (>~100 saturates) | AF2 gains flatten above ~100 |

Recommendation: **stratify** by these tiers rather than applying one
threshold; the per-tier accuracy curve (MarinFold vs Protenix-MSA vs
Protenix-single-seq) is the headline deliverable issue #41 asks for.

### 2.4 The tool — `msa_depth.py`

[`msa_depth.py`](msa_depth.py) (in this folder) is a dependency-light
(numpy-only) reference implementation: it parses ColabFold `non_pairing.a3m`
files (stripping a3m lowercase insertion columns), computes clustering
Neff at configurable theta, and emits a CSV with all four measures, one
row per protein, sorted shallowest-first. It has a built-in self-test
(`python msa_depth.py selftest`) and handles both the exp12 nested layout
(`<stem>/msa/0/0/non_pairing.a3m`) and a flat `<stem>.a3m` dir.

For an externally-maintained, citable cross-check, **NEFFy**
([github](https://github.com/Maryam-Haghani/NEFFy), Bioinformatics 2025)
computes the same clustering Neff from an a3m and supports the same
normalizations; running it on a handful of proteins is a good validation
of our numbers but is not a runtime dependency.

---

## 3. Step 0 (do first): measure the FoldBench-100 depths

The MSAs already exist — exp12 precomputed them once via ColabFold
MMseqs2 and persisted them to:

- the Modal Volume `protenix-foldbench-msa`
  (`<stem>/msa/0/0/non_pairing.a3m`), and
- the HF bucket `open-athena/MarinFold` under
  `data/protenix-foldbench-monomers/msa/<stem>/msa/0/0/non_pairing.a3m`.

So Step 0 is **not** an MSA recompute — just stage those a3m files and run
the tool:

```bash
# option A: from the HF bucket (needs an HF token with bucket access)
hf download open-athena/MarinFold --repo-type dataset \
    --include 'data/protenix-foldbench-monomers/msa/*/msa/0/0/non_pairing.a3m' \
    --local-dir /tmp/fb100_msa
python msa_depth.py dir \
    /tmp/fb100_msa/data/protenix-foldbench-monomers/msa \
    --layout exp12 --out data/foldbench100_msa_depth.csv

# option B: straight off the Modal Volume
modal volume get protenix-foldbench-msa / /tmp/fb100_msa   # MODAL_PROFILE set
python msa_depth.py dir /tmp/fb100_msa --layout exp12 \
    --out data/foldbench100_msa_depth.csv
```

The output CSV joins onto exp12's `data/scores.csv` on `stem`
(`<pdb>_<chain>`), so the immediate payoff is a **performance-vs-Neff**
scatter for all three predictors using data we already have. This
confirms quantitatively what Tim's plot shows (few low-depth points) and
gives us the FoldBench baseline distribution to compare any curated set
against.

*(I could not run Step 0 from this environment — the HF bucket is private
and unauthenticated here, and Modal needs the workspace profile. The
tooling above is ready for whoever has creds; the tool itself is tested.)*

---

## 4. Where to get MORE low-MSA datapoints

Two complementary sources. Both must yield **experimentally-solved
monomer structures** (predicted structures as ground truth would be
circular — same rule as `eval-dataset-design.md` §1).

### 4.1 De novo designed proteins — the cleanest, largest single bucket

Designed proteins have **no natural evolutionary lineage**, so a ColabFold
search returns ~just the query → Neff ~= 1 by construction. This is also
the priority subset in `eval-dataset-design.md` §7.1 (no homologs → a rare
apples-to-apples comparison vs MSA models), so the MSA-depth axis and the
designed-protein subset reinforce each other.

- **PDB `DE NOVO PROTEIN` class — ~2,007 experimental entries** (verified
  live via the RCSB search API, June 2026; matches the `eval-dataset-design.md`
  §7.1 count). The reproducible query is in `eval-dataset-design.md` §7.1.
- **Caveats** (verify, don't assume):
  1. The keyword is deposition-assigned. **Scaffold-grafted / motif-
     scaffolding / consensus designs can still hit natural homologs** —
     run them through `msa_depth.py` and keep only the genuinely shallow
     ones rather than trusting the label.
  2. Designed sets **skew short, idealized, and helical** — see §7. Report
     length- and SS-stratified numbers so a high designed-set score isn't
     just "good at short helical bundles."
  3. Some entries are designed binders/complexes — filter to single chain.

### 4.2 Natural low-Neff tail — pull and filter

Some natural classes are intrinsically homolog-poor, but the only
reliable selector is **measured Neff**, not the class label:

| Class | Why shallow | Usable as low-MSA experimental source? |
|---|---|---|
| Orphan / taxonomically-restricted genes | no detectable homologs outside a narrow clade | Yes but **sparse** — most orphans are uncharacterized / disordered, few solved structures |
| Viral proteins | high mutation rate; novel folds (11% of viral proteins are "orphan", [Sci. Adv. 2025](https://www.science.org/doi/10.1126/sciadv.adz8560)) | **Partial** — many still have deep *within-family* MSAs (spike/HA/env). Filter on Neff, never on "is viral" |
| Antibodies / nanobodies | hypervariable CDR loops | **Risky** — frameworks have deep MSAs; only loops are low-info, and structures are near-clones. Treat as a labelled *loop-accuracy* stratum, dedupe hard. (Measured Neff of nanobody vs antibody is similar — "nanobody = shallow" is a myth) |
| Fast-evolving / intrinsically disordered | rapid divergence erases signal | **Avoid** as structure targets — often no single stable structure |

The robust recipe (mechanical, uses infra we already have): **pull a large
pool of recent PDB monomers, compute their Neff with the same ColabFold
MMseqs2 path exp12 uses, and keep the low-Neff tail.**

### 4.3 Ready-made comparator sets (reuse, don't rebuild)

These published low-MSA sets give us instant comparators and validated
threshold definitions:

- **Orphan25 / Orphan54 / Design55** (trRosettaX-Single, Nat. Comput.
  Sci. 2022) — 25 orphans (no DB homologs), 54 targets at **Neff < 10**,
  55 human-designed. Published per-method TM-scores to compare against.
- **RGN2** (Chowdhury et al., Nat. Biotechnol. 2022,
  [s41587-022-01432-w](https://www.nature.com/articles/s41587-022-01432-w))
  — a curated **orphan + designed** benchmark scored by dRMSD. *(Exact
  sizes/Neff definition: pull from the paper + repo.)*
- **CASP Free-Modeling (FM) targets** — the canonical "no template, few
  homologs" hard set (CASP14: ~23 FM EUs; CASP15 FM "lacked templates …
  few homologues"). Small (tens) but gold-standard, temporally honest.
- **CAMEO hard targets** — rolling difficult monomers (difficulty is not
  purely MSA-depth, so re-label with measured Neff).

---

## 5. Concrete curation pipeline (natural tail)

```
1. CANDIDATE POOL  (RCSB search API)
   - experimental only (X-ray / cryo-EM / NMR); EXCLUDE computed models
   - single polymer entity instance (true monomer)
   - length 40-400 aa
   - resolution <= 3.0 A
   - deposition date > MarinFold's AFDB-snapshot date  (leakage; see
     eval-dataset-design.md caveats -- anchor to OUR snapshot, not 2018/2021)

2. CHEAP PRE-FILTER  (avoid MSA-ing everything)
   - OpenProteinSet (AWS Open Data, NeurIPS 2023): precomputed AF2-style
     MSAs for ~140k PDB chains -> read Neff straight off for older chains
   - UniRef50/90 cluster size of the chain's UniProt accession as a coarse
     proxy (no alignment) -- but it MISSES environmental homologs that
     ColabFold's MMseqs2 finds, so it OVER-calls "low MSA". Pre-filter only.
   - optional: ESM-2 perplexity as an MSA-free difficulty proxy
     (ESMFold paper: perplexity vs TM Pearson -0.55 CAMEO / -0.67 CASP14)

3. MEASURE Neff for real  (the gate that counts)
   - run the exp12 ColabFold MMseqs2 path (Modal precompute_msa) on the
     short-listed candidates -> non_pairing.a3m
   - msa_depth.py -> Neff, Neff/L, Neff/sqrtL

4. SELECT + STRATIFY
   - keep the Neff tier bins from 2.3 (orphan / low / marginal)
   - cluster survivors at ~30% identity (MMseqs2) to dedupe families
     (critical for designed + nanobody redundancy)

5. CROSS-LABEL with the structural-novelty axis (exp41 point 2 / Foldseek)
   - so every datapoint carries BOTH (Neff tier, fold-novelty verdict)
```

The Modal `precompute_msa` function in
`experiments/exp12_data_protenix_foldbench_monomers/modal_app.py` already
does exactly the ColabFold MSA build (idempotent, persisted); the curation
job is "feed it a new candidate list" plus `msa_depth.py`, not new MSA
infra.

---

## 6. Two axes, one labelled eval set

`eval-dataset-design.md` (+ exp41 Foldseek tool) gives every candidate a
**structural-novelty** verdict (nearest training fold's TM: redundant /
same-fold / novel-fold). This doc gives every candidate an **MSA-depth**
tier (Neff: orphan / low / marginal / deep). They are orthogonal and we
want the **2-D label** on each eval datapoint:

|              | Deep MSA | Shallow MSA (Neff<10) |
|---|---|---|
| **Fold in training** | easy / memorization ceiling | MSA-models lose edge; folding-vs-retrieval |
| **Novel fold** | structural generalization | **hardest cell — the headline test** |

The headline number issue #41 asks for — MarinFold-1B vs Protenix-single
vs Protenix-MSA **as a function of MSA depth** — is the marginal over the
columns of this table; the rows let us check it isn't confounded by fold
novelty.

---

## 7. Pitfalls

- **Designed sets skew short/helical/idealized.** Mix in natural low-Neff
  entries; report length- and SS-stratified numbers.
- **Nanobodies/antibodies are near-clones** with deep frameworks. A model
  can "win" by memorizing the Ig fold. Labelled loop-accuracy stratum at
  most; dedupe hard. ("Nanobody = shallow MSA" is false — measured Neff
  ~= antibody.)
- **Viral != shallow.** Spike/HA/env have thousands of relatives. Gate on
  measured Neff, never the label.
- **Leakage is the real risk.** Designed proteins and recent orphans may
  already be training documents. Filter by deposition date **and** by
  sequence/structure identity vs the training set (the exp41 Foldseek tool
  + an MMseqs2 sequence pass), not by MSA depth alone.
- **Orphans skew disordered** — screen for a single well-ordered
  experimental structure (ordered fraction / B-factors), or the regression
  target is noise.
- **Pre-computed Neff tables use a fixed DB snapshot.** Use them only for
  the §5 pre-filter; the *final* Neff label must come from the same
  ColabFold MMseqs2 + current DBs that the Protenix-MSA baseline actually
  consumes, so the comparison is apples-to-apples.

---

## 8. Recommendation / next steps

1. **Now (no new compute):** run `msa_depth.py` on the existing
   FoldBench-100 a3m files (§3) → `data/foldbench100_msa_depth.csv`,
   join to exp12 scores, plot accuracy-vs-Neff for the three predictors.
   This is the literal "save the MSA depth for the 100 monomers" ask and
   establishes the baseline depth distribution.
2. **Cheap, high-return:** pull the **PDB `DE NOVO PROTEIN`** list
   (~2,007), filter to single-chain experimental monomers post-snapshot,
   MSA them via exp12's `precompute_msa`, keep Neff~=1 survivors. Largest
   guaranteed-shallow bucket; overlaps the designed-protein priority in
   `eval-dataset-design.md` §7.1.
3. **Natural tail:** run the §5 pipeline for a stratified natural low-Neff
   set; reuse Orphan25/54 + Design55 + RGN2 + CASP-FM as comparators.
4. **Label both axes** (§6): cross with the exp41 Foldseek verdict so each
   datapoint carries (Neff tier, fold-novelty).

Where this code should ultimately live: under issue #41's experiment dir
`experiments/exp41_evals_foldseek_train_similarity/` (AGENTS rule: one
issue → one experiment dir), coordinated with the point-2 work already in
flight there — `msa_depth.py` is a natural sibling of `query_similarity.py`
(both "label a candidate structure"). Staged here in `.dev/` first to
avoid colliding with that in-flight work and so this doc can be attached
to issue #41 like its companions.

---

## 9. References

Verified inline above. Key sources:

- **AlphaFold2** — Jumper et al., Nature 2021.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/ (80% per-residue
  Neff; depth<30 falloff; ~100 saturation; 6.1 GDT BFD/MGnify ablation).
- **trRosettaX-Single** — Wang et al., Nat. Comput. Sci. 2022.
  https://www.nature.com/articles/s43588-022-00373-3 (Orphan25,
  **Orphan54 = Neff<10**, Design55; orphan TM vs AF2/RoseTTAFold).
- **trRosetta** — Yang et al., PNAS 2020.
  https://www.pnas.org/doi/10.1073/pnas.1914677117 (`Nf = Neff/sqrt(L)`;
  accuracy ~ log Nf). *(√L wording not quoted verbatim — PDF blocked.)*
- **NEFFy** — Haghani et al., Bioinformatics 2025.
  https://doi.org/10.1093/bioinformatics/btaf222 ·
  https://github.com/Maryam-Haghani/NEFFy (clustering-Neff formula;
  none / L / √L normalizations; the citable cross-check tool).
- **DeepMSA** — Zhang et al., Bioinformatics 2020.
  https://academic.oup.com/bioinformatics/article/36/7/2105/5628221
  (the "Neff >= 128 = deep enough" escalation cutoff).
- **HH-suite** — https://github.com/soedinglab/hh-suite/wiki
  (entropy-based `.hhm` NEFF — different quantity, do not mix).
- **RGN2** — Chowdhury et al., Nat. Biotechnol. 2022.
  https://www.nature.com/articles/s41587-022-01432-w (orphan+designed
  benchmark).
- **ESMFold** — Lin et al., Science 2023 / bioRxiv 2022.
  https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
  (perplexity vs TM as an MSA-free difficulty proxy).
- **OmegaFold** — Wu et al., bioRxiv 2022.
  https://www.biorxiv.org/content/10.1101/2022.07.21.500999 (single-seq
  wins on orphans).
- **OpenProteinSet** — Ahdritz et al., NeurIPS 2023.
  https://registry.opendata.aws/openfold/ (~140k precomputed PDB-chain
  MSAs → free Neff for the §5 pre-filter).
- **Viral AlphaFold DB** — Sci. Adv. 2025.
  https://www.science.org/doi/10.1126/sciadv.adz8560 (11% viral orphans;
  novel folds).
- **RCSB search API** — `struct_keywords.pdbx_keywords = "DE NOVO PROTEIN"`
  → 2,007 entries (verified June 2026). Query in `eval-dataset-design.md`
  §7.1.

### Flagged as not fully verified

- The **62%** identity cutoff (DCA tradition; no single primary citation
  pinned — use 80% for AF2 comparability).
- trRosetta's exact `Nf = Neff/√L` wording (PDF blocked; corroborated via
  the trRosettaRNA follow-up).
- Exact **Orphan54 / RGN2** composition and provenance — pull from the
  papers' supplements / repos before final use.
- Per-class PDB counts for natural orphan/viral structures — order-of-
  magnitude only.
