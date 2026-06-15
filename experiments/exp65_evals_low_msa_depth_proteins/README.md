---
marinfold_experiment:
  issue: 65
  title: "Collect proteins with low MSA depth for evaluation"
  kind: evals
  branch: exp/65-low-msa-depth-proteins
---

# Low-MSA / structurally-novel eval proteins (issue #65)

**Issue:** [#65](https://github.com/Open-Athena/MarinFold/issues/65) · **Kind:** `evals` · **Branch:** `exp/65-low-msa-depth-proteins`

Continuation of [#41](https://github.com/Open-Athena/MarinFold/issues/41):
PR #59 landed the Foldseek train-similarity utilities + DB
(`exp41_evals_foldseek_train_similarity/`), but not the actual candidate
dataset. This experiment collects that dataset.

## Question

MarinFold predicts contacts/distograms from a **single sequence**. The regime
where a single-sequence model has the most to prove is where MSA-based models
lose their coevolution signal: **shallow-MSA** proteins, and proteins whose
**fold is far from the AFDB training set**. Our current eval (FoldBench-100)
barely probes either regime (the #41 Foldseek tool measured 48 redundant / 51
same-fold / 1 novel vs the training set). This experiment asks: **where do we
get candidate structures that are deliberately far from training, on both the
MSA-depth and fold-novelty axes?**

## Approach

This experiment is the **data-sourcing** half of the #41/#65 effort: fetch candidate
structures from three sources that are, for different reasons, far from the
AFDB/UniProt training universe, and emit a shared manifest so each candidate
can later be labelled on **both** novelty axes (the 2-D label in
[`notes/low-msa-eval-curation.md`](notes/low-msa-eval-curation.md) §6):

- **fold-novelty** ← the #41 Foldseek verdict (`exp41_evals_foldseek_train_similarity`)
- **MSA-depth** ← `msa_depth.py` Neff tier (graduated into this dir)

### Sources

| Script | Source | Why it's far from training |
|---|---|---|
| `fetch_denovo_pdb.py` | PDB `DE NOVO PROTEIN` class (RCSB search API) | Designed proteins have no evolutionary lineage (Neff ≈ 1) and often novel folds; mostly outside UniProt/AFDB. |
| `fetch_casp_fm.py` | CASP14/15 **free-modeling** target domains | The community "no usable template, few homologs" gold standard; blind + temporally honest. |
| `fetch_cameo_hard.py` | CAMEO **hard** modeling targets | Rolling weekly hard monomers; difficulty is not purely MSA depth, so re-label with measured Neff. |

### Files

| File | What it does |
|---|---|
| `_pdb_io.py` | Shared HTTP download (idempotent) + the common manifest schema + gemmi residue counting. |
| `fetch_denovo_pdb.py` | RCSB search → filtered de novo monomers → mmCIFs + `data/denovo_pdb_manifest.csv`. |
| `fetch_casp_fm.py` | CASP target tarballs + committed FM classification → FM domain PDBs + `data/casp_fm_manifest.csv`. Domains absent from the public monomer tarballs (oligomeric / late-release targets) are recovered by clipping the FM domain out of the deposited RCSB entry, per `data/casp_fm_pdb_fallback.csv`. |
| `data/casp_fm_pdb_fallback.csv` | Committed map (domain → `pdb_id, chain, casp_range`) for FM domains not in the predictioncenter tarballs; drives the RCSB clip-out fallback. Rows marked `unavailable` have no released PDB (or are only partially modelled) and stay unresolved. |
| `fetch_cameo_hard.py` | CAMEO raw-targets tarball + difficulty labels → hard-target PDBs + `data/cameo_hard_manifest.csv`. |
| `msa_depth.py` | Neff / Neff-per-L / Neff-per-√L from a ColabFold `.a3m` (numpy-only; `selftest`). Graduated from `.dev/`. |
| `data/casp_fm_domains.csv` | Committed CASP14/15 FM + FM/TBM domain classification (parsed from the assessors' `results.cgi`). |
| `notes/` | The point-1 research notes (curation strategy, the leakage-free split design, the eval-strategy summary). |

### Shared manifest schema

Every fetch script writes the same columns so the three sources stack and join
onto exp12/exp41 CSVs and the future Foldseek + Neff labels (`stem` =
`<id>_<chain>` is the join key):

```
source, stem, pdb_id, chain, length, resolution, deposit_date,
category, novelty_axis, local_path
```

## Running it

Each script is idempotent (re-runs skip files already on disk) and takes a
`--limit N` for a smoke test. Downloaded structures land under `structures/`
(gitignored, large/re-fetchable); the small manifests under `data/` are
committed.

```bash
uv sync --extra test

# de novo: count first (sanity), then fetch the filtered monomer set
uv run python fetch_denovo_pdb.py --count-only          # -> 396 (filtered) ; bare keyword = 2007
uv run python fetch_denovo_pdb.py                        # all filtered monomers -> data/denovo_pdb_manifest.csv
uv run python fetch_denovo_pdb.py --limit 5             # smoke

# CASP14/15 free-modeling target domains
uv run python fetch_casp_fm.py --casp both              # -> data/casp_fm_manifest.csv

# CAMEO hard targets (default 1-year window)
uv run python fetch_cameo_hard.py                        # -> data/cameo_hard_manifest.csv

# MSA depth on any ColabFold a3m tree (downstream; needs precomputed MSAs)
uv run python msa_depth.py selftest
uv run python msa_depth.py dir MSA_ROOT --layout exp12 --out data/<set>_msa_depth.csv
```

## Cross-labelling pipeline (downstream of the fetch)

Each candidate gets a **3-axis novelty label**, then they're merged into one
per-protein table ([`data/candidate_2d_label.csv`](data/candidate_2d_label.csv)):

| Axis | How | Script | Output |
|---|---|---|---|
| **Fold novelty** | Foldseek TM vs the 1.33M AFDB-24M train reps (exp41 tool) | `query_similarity.py` (exp41) | `data/*_vs_afdb_reps_similarity.csv` |
| **Sequence leakage** | MMseqs2 vs the same reps' sequences (≥30% id, ≥50% cov to a *train* rep) | `seq_leakage.py` | `data/candidate_seq_leakage.csv` |
| **MSA depth (Neff)** | ColabFold MMseqs2 MSA (no Modal) → effective homolog count | `fetch_msa_colabfold.py` + `msa_depth.py` | `data/candidate_msa_depth.csv` |
| **merge** | join the three by candidate (+ `deposit_date` from each source manifest); crosstabs | `combine_axes.py` | `data/candidate_2d_label.csv` |

Sequences are pulled from the structures by `extract_sequences.py`.

`candidate_2d_label.csv` also carries each candidate's `deposit_date` (the
temporal axis, joined from the per-source manifests by source + stem, so the
three de-novo/CAMEO cross-listings each keep their own date). CASP FM rows are
blank — those domains come from the prediction-center tarballs and carry no PDB
deposition date.

## Results

457 candidates (454 unique; `8k7z_A`, `9ded_A`, `9mrb_A` are in both the de
novo and CAMEO sets). All three sources deliver real novelty that
FoldBench-100 does not.

**Fold novelty** (vs AFDB-24M train reps; novel = best-train qtm < 0.5):

| dataset | redundant | same_fold | **novel_fold** | median train-qtm |
|---|---|---|---|---|
| CASP FM | 2 | 11 | **13 (50%)** | 0.50 |
| CAMEO hard | 6 | 17 | **12 (34%)** | 0.73 |
| de novo | 186 | 187 | **23 (6%)** | 0.89 |

(FoldBench-100 for comparison: **1/100 novel**.)

**Sequence leakage** (≥30% id & ≥50% cov to a train rep): **182/454 (40%)**
leaked; CASP FM lowest (19%), de novo 41%, CAMEO 40%. De novo designs are
mostly *structurally* close to known folds but *sequence*-novel.

**MSA depth** (ColabFold Neff @ θ=0.8): **221/454 (49%) have Neff < 10** — the
shallow regime the experiment targets. By dataset (orphan/low/marginal/deep):
de novo 89/104/11/192, CAMEO 17/5/2/11, CASP FM 3/6/2/15. CASP FM skews deep
(its assessor-released monomers are fold-novel but homolog-rich); the
oligomeric / late-release FM targets recovered from the PDB (§Limitations 3),
by contrast, are mostly orphan-MSA.

(Global counts here are over the **454 unique** structures; the per-dataset
rows above count each source's members, so the 3 proteins cross-listed in both
de novo and CAMEO are counted under both and the per-dataset totals sum to
457.)

**The headline grid — fold novelty × MSA depth** (454 unique structures; [`plots/two_axis_label.png`](plots/two_axis_label.png)):

| fold ↓ \ Neff → | orphan | low | marginal | deep |
|---|---|---|---|---|
| redundant | 14 | 49 | 0 | 131 |
| same_fold | 70 | 55 | 13 | 74 |
| **novel_fold** | **22** | **11** | 2 | 13 |

**33 candidates are novel-fold AND shallow-MSA** (orphan+low) — the
hardest, most-informative cell, which FoldBench-100 essentially cannot
populate (21 de novo, 6 CAMEO, 6 CASP FM). See `plots/novelty_by_source.png`
and `plots/candidates_overview.png` for the other views.

## Limitations & caveats

Read the table with these in mind (all are deliberate trade-offs, not bugs):

1. **`msa_neff` is a subsample estimate for deep MSAs.** Neff is O(N²·L), so
   for the ~29% of candidates with more than 5,000 aligned sequences it is
   computed over a deterministic (seeded) 5,000-sequence subsample. The
   `neff_tier` ("deep") is unaffected and the *shallow* regime we actually care
   about is exact, but the absolute `msa_neff` value for a deep protein is a
   subsample lower bound and is **not** comparable across deep proteins. Don't
   read deep Neffs as precise.
2. **The pool is de-novo-dominated.** 396 of 457 candidates are de novo
   designs, which are orphan *by construction* (no evolutionary history) and
   mostly structurally close to known folds. The headline novel-fold ×
   shallow-MSA cell is therefore carried mostly by de novo designs; the
   natural, blind signal (CASP / CAMEO) in that cell is thinner. When drawing
   the eval conclusion, look at a natural-only (CASP + CAMEO) slice too, not
   just the pooled count.
3. **CASP FM coverage is 26/32.** Six FM domains remain unresolved: five have
   no released PDB (`T1052-D3`, `T1053-D1`, `T1053-D2`, `T1130-D1`,
   `T1131-D1`), and `T1125-D6` is only partially modelled (21/66 residues) in
   `8h2n`. The seven recovered via the RCSB clip-out fallback rely on the
   deposited chains carrying author residue numbers equal to the CASP target
   numbering (verified per entry). The blind gold-standard stratum is still the
   smallest of the three.
4. **The sequence-leakage threshold is generous.** `redundant_seq` = ≥30%
   identity over ≥50% query coverage to a training representative. 30% identity
   is the twilight zone, so the leakage count is closer to an *upper* bound;
   the label conflates a remote homolog with memorization. Tighten
   `REDUNDANT_ID` in `seq_leakage.py` for a stricter definition.
5. **MSA reproducibility.** The ColabFold MSAs depend on the
   UniRef30 / ColabFoldDB snapshot served by `api.colabfold.com` at query time;
   re-running later will drift Neff. The Neff subsample itself is seeded, so it
   is deterministic given a fixed `.a3m`.
6. **Three proteins are double-counted.** `8k7z_A`, `9ded_A`, `9mrb_A` appear
   in both the de novo and CAMEO sets, so the per-dataset crosstabs count them
   twice (457 rows, 454 unique stems).

## Conclusion

This experiment turns three "far-from-training" sources into a labelled
candidate pool that spans **both** novelty axes the eval needs. Where
FoldBench-100 is ~entirely redundant/deep-MSA, this set provides 48 novel-fold
and 221 shallow-MSA candidates — and crucially **33 that are both** (12 of them
natural CASP/CAMEO targets, not just de novo designs). The
per-protein [`data/candidate_2d_label.csv`](data/candidate_2d_label.csv) is the
deliverable: it lets the eval stratify MarinFold vs MSA-based baselines by
(fold novelty × MSA depth × sequence leakage). Next: compute MarinFold /
Protenix accuracy on these strata (the headline #41/#65 number).
