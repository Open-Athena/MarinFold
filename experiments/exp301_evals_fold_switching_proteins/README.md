---
marinfold_experiment:
  issue: 301
  title: 'exp: does MarinFold represent both folds of fold-switching proteins?'
  kind: evals
  branch: claude/marinfold-fold-switching-be3610
---

# exp: does MarinFold represent both folds of fold-switching proteins?

**Issue:** [#301](https://github.com/Open-Athena/MarinFold/issues/301) · **Kind:** `evals` · **Branch:** `claude/marinfold-fold-switching-be3610`

## Question

Does MarinFold's distribution contain **both** conformations of fold-switching
proteins — and if it does not sample them spontaneously, how much explicit
contact evidence does it take to force the alternative fold?

## Hypothesis

MarinFold prefers one fold, usually the one its own training document encoded;
spontaneous bimodality sits at the single-fold null; the alternative fold is
improbable rather than impossible under teacher forcing; and conditioning works
with **k\* ≫ 1**, somewhere in 10–40 contacts.

## Background

Fold switchers adopt two distinct stable folds from one sequence, which makes
them the field's sharpest generalisation probe. AlphaFold2 predicts only one
conformation for **94%** of them ([Chakravarty & Porter 2022](https://doi.org/10.1002/pro.4353));
pooling >282,000 predictions from every AF2/AF3 variant gives **35% success on
fold switchers likely in the training set and 1/7 outside it**
([Chakravarty et al. 2024](https://doi.org/10.1038/s41467-024-51801-z)), which
is why that paper's conclusion is *memorization*.

MarinFold is a different instrument, in three ways that matter:

1. **`contacts-v1` is generative over contact sets.** "Is the other fold in the
   distribution?" is answered by sampling, where AF2 needs MSA hacks (AF-Cluster,
   SPEACH_AF) to manufacture a second conformation. MarinFold is MSA-free, so
   that entire debate is inapplicable by construction.
2. **It is promptable.** Conditioning is extending the prefix past
   `<begin_statements>`, so "how much evidence moves the model" is a measurable
   number with no AF2 analog.
3. **We can read our own training data**, so the memorization test Porter's group
   had to infer for AF2 can be run directly here.

[Lee & Porter 2026](https://arxiv.org/abs/2601.01740) frame fold switching as
precisely **contact redistribution** — "single folders tend to undergo
conformational changes that largely preserve their residue-residue contacts,
fold switchers can interconvert between conformations with substantially
different contacts" — so the contact map is the right coordinate system.

Closest precedent: [#224](https://github.com/Open-Athena/MarinFold/issues/224)
(circular permutation) sets the rigour bar — every confound gets its own control.
Related: [#254](https://github.com/Open-Athena/MarinFold/issues/254) (one contact
is almost no conditioning), [#163](https://github.com/Open-Athena/MarinFold/issues/163)
(conditioning on true partial *sets* moved R-precision 0.145 → 0.556),
[#213](https://github.com/Open-Athena/MarinFold/issues/213) (the 70.9M-sequence
training DB), [#142](https://github.com/Open-Athena/MarinFold/issues/142)
(under-generation as a difficulty symptom).

## Approach

**Model:** `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, the `MODELS.yaml` default.
**Compute:** CoreWeave `cw-rno2a` H100 fan-out at batch priority (exp82's recipe).

**Eval set.** `supporting tables/TableS1.xlsx` from
[`ncbi/AF2_benchmark`](https://github.com/ncbi/AF2_benchmark) — 93 fold-switching
pairs, each as Fold1/Fold2 PDB+chain plus the fold-switching region sequence.
178 unique entries, 177 of them already in the local RCSB mirror.

Ground truth is pyconfind side-chain contact degree under exactly `contacts_v1`'s
geometry — marinfold's `analyze_structure` with exp74's `PYCONFIND_KWARGS`,
imported rather than forked, with the numba backend the corpora are defined by.
The two chains are merged into one **union reference** so both folds' contacts
share a coordinate frame, positions where the chains carry different amino acids
are dropped, and everything is restricted to residues **resolved in both**. That
yields the sets every metric is computed on: `S` shared, **`A` Fold1-unique,
`B` Fold2-unique** — the discriminative universe. Nothing here is computed on the
global map, which the shared core dominates.

Measurements (`prepare_inputs.py` → dispatch/worker → `analyze.py` → `plot.py`):

- **M1 observational** — exp82's recipe (100 rollouts × 10 seeds, T=1.0, top-p
  0.95, top-k off, 6L+128); per-rollout fold score `φ = |r∩A|/|A| − |r∩B|/|B|`;
  bimodality against a **calibrated single-fold null**.
- **M2 teacher-forced ΔNLL** — identical document realizations for both folds, so
  the contact set is the only difference; matched statement counts, reported
  per-statement.
- **M3 memorization audit** — MMseqs2 against exp213's training DB at `-s 7.5`,
  then recover the matching AFDB / ESM-Atlas training documents and score them
  against both folds for a per-protein **training-fold label**.
- **M4 forcing** — dose–response `k ∈ {0,1,2,5,10,20,40}` fold-B contacts in the
  prefix with the symmetric fold-A control, scoring recovery of the *remaining*
  B contacts; headline **k\***.
- **M5 controls** — single-fold null, crystal-replicate floor, difficulty control
  against eval-val at matched L, sequence-KNN null, contacts/rollout and
  `frac_finished`.

## Success criteria

Per-fold R-precision and AUC on the discriminative universe with seed error bars;
the φ distribution with a bimodality test calibrated against the single-fold null;
ΔNLL per protein; the training-fold label and its partial correlation with φ; and
the conditioning dose–response with k\* and its symmetric control. A clean
negative is a result, and is directly comparable to AF2's 12–35%.

## Results

### Phase 0 — the eval set exists and the premise holds

`prepare_inputs.py` builds all **93/93 pairs with no failures**;
`build_noise_floor.py` puts a measurement floor under them.

**The gate.** A pair has to be able to discriminate two folds at all, so three
independent criteria each kill it: fewer than 10 contacts unique to either fold,
under 50% overlap between the two chains, or a fold-switching region that cannot
be located. **68 of 93 pass.**

| tier | what | identical-sequence | homolog |
|---|---|---:|---:|
| **A** | both folds are two chains of one PDB entry | 6 | 0 |
| **B** | X-ray / X-ray, two entries | 47 | 0 |
| **C** | NMR or cryo-EM involved | 12 | 3 |
| | **total** | **65** | **3** |

The 25 that fail do so for: `min(|A|,|B|)` too small (17), chain coverage under
50% (14), region not located (3) — some fail on more than one.

**The premise, against its own noise floor.** The two folds of a pair share a
median Jaccard of **0.515**. That is only meaningful against how much two contact
maps differ when nothing has switched, so `build_noise_floor.py` measures it from
the same entries, with no new downloads: **two chains of the same sequence in the
same fold within one crystal** — same molecule, same resolution, same refinement.

| set | n | median Jaccard | p10 | range |
|---|---:|---:|---:|---|
| same-fold replicates (81 chain pairs, 51 entries) | 81 | **0.873** | 0.728 | 0.617–1.000 |
| fold-switch pairs passing the gate | 68 | **0.515** | — | 0.008–0.801 |

**All 68 sit below the replicate median and 62 of 68 below its p10.** The floor of
0.873 independently reproduces #224's DsbA crystal-replicate figure of 0.85–0.88
on a completely different set of proteins, which is a real check on the pipeline
rather than a restatement of it.

**What the 68 give us to work with** (medians): L = 291, common universe = 254
residues, **|A| = 71 and |B| = 58** discriminative contacts, summing to **5,077
`min(|A|,|B|)` contacts** across the set.

**One confound the data surfaced, now carried explicitly.** A pair's two crystals
also differ for reasons that have nothing to do with fold switching — ligands,
domain motions, different constructs — and for a long protein those dominate.
`1h38d/1qlna` is the extreme case: L = 882, and only **2%** of its discriminative
pairs touch the switching region. So FS-region-restricted sets (`n_only1_fs`,
`n_only2_fs`) are computed alongside the global ones for every pair. The median
pair puts **36%** of its discriminative contacts in the switching region, and
**46 of the 68 pairs** keep ≥ 10 on both sides after restriction — that is the
strict subset the FS-only analysis will lead with.

Artifacts: [`data/foldswitch_universe.jsonl`](data/foldswitch_universe.jsonl)
(one record per pair, the Phase-1 input),
[`data/premise_gate.csv`](data/premise_gate.csv),
[`data/noise_floor.csv`](data/noise_floor.csv).

### Phases 1–4 — pending

M1/M2 on CoreWeave, M3 locally in parallel, then M4 sized by what M1 shows.

## Conclusion

Pending. Phase 0 establishes that the question is well-posed: 68 fold-switching
pairs whose two contact maps differ far beyond the crystallographic noise floor,
with a median of 71 and 58 contacts unique to each fold, in one coordinate frame,
ready to score.
