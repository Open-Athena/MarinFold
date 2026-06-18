---
marinfold_experiment:
  issue: 78
  title: "exp: update experiment 64 to include ESMFold and ESMFold2 results"
  kind: evals
  branch: exp/78-esmfold-esmfold2-contacts
---

# exp: update experiment 64 to include ESMFold and ESMFold2 results

**Issue:** [#78](https://github.com/Open-Athena/MarinFold/issues/78) · **Kind:** `evals` · **Branch:** `exp/78-esmfold-esmfold2-contacts`

## Question

How do [ESMFold](https://github.com/facebookresearch/ESM) and [ESMFold2](https://github.com/Biohub/esm) perform on our eval set

## Hypothesis

N/A

## Background

We have run Protenix v2 in single sequence and in MSA mode on our eval set. Here we want to compute the same metrics for ESMFold and ESMFold2

## Approach

Run this on modal

Generate structure predictions on our eval set. Use the recommended ESMFold / ESMFold2 settings (log details about what settings you are using).

Let's score contact prediction accuracy from the predicted structures (rather than the distograms). Do this the same way we did it for Protenix v2's structure output - i.e. run pyconfind on the predicted structures and rank by contact degree.

Make sure you save all the predicted structures to the MarinFold huggingface bucket so that we can go back and re-score them using different criteria without re-running predictions

## Method (reproduction spec)

This experiment **extends [exp74](../exp74_evals_protenix_pyconfind_contacts/)**
(Protenix v2 contact eval) by adding two more *structure* predictors,
ESMFold and ESMFold2, on the **same eval set, same ground truth, same
metric**. We reuse exp74's two eval manifests verbatim
([`data/eval_manifest_foldbench.csv`](data/eval_manifest_foldbench.csv),
[`data/eval_manifest_exp65.csv`](data/eval_manifest_exp65.csv)) and its
`pyconfind_contacts.py` (copied unchanged). Protenix v2 results are
**not re-run** — they are pulled from exp74's
`contact_precision_all.csv` and spliced into the comparison table.

### Eval set (554 proteins, identical to exp74)
FoldBench-100 + all 454 exp65 low-MSA / novel-fold candidates (396
de-novo, 32 CAMEO-hard, 26 CASP-FM). Input sequences and ground-truth
structures are exactly those exp74 used; the candidate-pair universe is
restricted to GT-resolved residues, so every model's numbers are
comparable.

### Predictors (both single-sequence; Modal, H100)
- **ESMFold** — [`esmfold_app.py`](esmfold_app.py). `facebook/esmfold_v1`
  via HF `transformers.EsmForProteinFolding`. Single-sequence,
  deterministic. Settings logged: `num_recycles=4` (model default), ESM2
  language-model stem cast to fp16, trunk attention `chunk_size=128`. One
  prediction per protein.
- **ESMFold2** — [`esmfold2_app.py`](esmfold2_app.py). `biohub/ESMFold2`
  (ESMC-6B + diffusion all-atom; released 2026-05-27) via
  `transformers.models.esmfold2.ESMFold2Model` +
  `esm.models.esmfold2.ESMFold2InputBuilder`. **Single-sequence** mode.
  Diffusion settings logged: `num_loops=20`, `num_sampling_steps=100`
  (documented defaults), **best-of-N** = `n_samples` diffusion draws
  (distinct seeds), keeping the top-1 by the model's confidence (mirrors
  exp74's Protenix top-1-of-40 selection). The exact API was pinned by a
  feasibility spike ([`spike_esmfold2.py`](spike_esmfold2.py)) before the
  full fan-out.

Each predicted structure is persisted as mmCIF (`{stem}/structure.cif`)
to a Modal Volume, then to the HF bucket — so we can re-score under
different contact criteria without re-running predictions (an explicit
issue requirement). Per-protein timings are captured at prediction time
(AGENTS rule 9).

### Ground-truth + the metric (identical to exp74)
Ground truth = pyconfind side-chain contacts on the experimental
structure, run exactly like `contacts_v1` (`native_only=True`,
`contact_distance=3.0`, `dcut=25.0`, `clash_distance=2.0`); a "true"
contact is degree ≥ 0.001 and primary-sequence separation ≥ 6. Each ESM
prediction is scored with the **structure** predictor: run pyconfind on
the predicted CIF (same knobs) and rank candidate pairs by predicted
contact degree. **Metric — contacts @ L** (precision among the top-L
ranked pairs), also @ L/2, @ L/5, and **R-precision** (precision@R, R =
the bin's GT contact count, ceiling 1.0 per protein), reported in
aggregate (sep ≥ 6) and split short [6,11] / medium [12,23] / long [≥24].

### The 4-config comparison
The success-criteria comparison is the `predictor == "structure"` slice
across four model-configs — **Protenix-v2 · single-seq**, **Protenix-v2 ·
MSA**, **ESMFold**, **ESMFold2** — all ranking by pyconfind contact
degree on the predicted structure, scored against one shared pyconfind
ground truth.

### Files
| File | Role |
|---|---|
| `pyconfind_contacts.py` | pyconfind GT/prediction wrapper + chain extraction + alignment (copied from exp74) |
| `esmfold_app.py` | ESMFold-on-Modal harness (single-seq, deterministic) |
| `esmfold2_app.py` | ESMFold2-on-Modal harness (single-seq, best-of-N diffusion) |
| `spike_esmfold2.py` | one-shot feasibility/API spike for ESMFold2 on Modal |
| `contact_eval.py` | model-agnostic structure-config contacts@L scorer (→ the three tables) |
| `combine_scores.py` | concat ESM datasets + splice in exp74 Protenix rows → unified `model`-keyed table |
| `plot.py` | 4-config comparison plots (by range; stratified by neff_tier / fold_verdict) |
| `fetch_gt_structures.py` | fetch exp65 GT structures (RCSB) for scoring (from exp74) |
| `cli.py` | `contact-eval` driver |
| `upload_to_hf.sh` | push scores + all predicted structures to the HF bucket |

## Success criteria

We have a summary pdf with plots showing protenix v2 (single sequence), protenix v2 (msa), ESMFold, ESMFold2 contact prediction accuracy using all our existing contact prediction accuracy metrics

## Results

Folded all **552 proteins** (FoldBench-100 + 452 unique exp65 candidates;
the eval manifests carry 554 rows, 552 unique stems) with both ESMFold and
ESMFold2, single-sequence, on Modal H100s — **0 failures**. pyconfind
ground-truth alignment identity was **≥ 0.95 for every structure** (mean
0.9994) and the predicted-structure alignment identity was **1.000**, so
the resolved-residue → input-sequence mapping is sound. Source data is
[`data/contact_precision_all.csv`](data/contact_precision_all.csv) (tidy,
one row per stem × model × predictor × range × cut; ESMFold + ESMFold2 +
the spliced-in Protenix-v2 rows from exp74).

**Settings used** (logged per run): ESMFold `facebook/esmfold_v1`,
`num_recycles=4`, ESM2 stem fp16, trunk `chunk_size=128`, one deterministic
prediction per protein. ESMFold2 `biohub/ESMFold2` (ESMC-6B, 6.58 B params,
~13.9 GB GPU), single-sequence, `num_loops=20`, `num_sampling_steps=100`,
**best-of-5** diffusion samples kept by the model's pTM confidence.

### Headline — R-precision (precision@R; ceiling 1.0 for every protein)

The success-criteria 4-config comparison (all rank pairs by pyconfind
contact degree on the predicted structure):

**FoldBench-100** (natural proteins):

| config | agg | short | medium | long |
|---|---|---|---|---|
| protenix-v2 · SS | 0.282 | 0.364 | 0.300 | 0.252 |
| protenix-v2 · MSA | **0.847** | 0.843 | 0.849 | 0.833 |
| ESMFold | 0.755 | 0.755 | 0.762 | 0.743 |
| **ESMFold2** | **0.805** | 0.803 | 0.813 | 0.794 |

**exp65** (low-MSA / novel-fold; 452 candidates):

| config | agg | short | medium | long |
|---|---|---|---|---|
| protenix-v2 · SS | 0.674 | 0.707 | 0.700 | 0.642 |
| protenix-v2 · MSA | **0.804** | 0.809 | 0.823 | 0.787 |
| ESMFold | 0.755 | 0.760 | 0.780 | 0.730 |
| **ESMFold2** | 0.782 | 0.789 | 0.800 | 0.763 |

Two robust findings:
1. **ESMFold2 is the best single-sequence contact predictor here.** On
   natural proteins its single-sequence R-precision (0.805) **nearly matches
   MSA-mode Protenix-v2** (0.847) and crushes single-sequence Protenix
   (0.282); ESMFold2 > ESMFold at every separation.
2. **In the low-MSA / novel-fold regime the gap to MSA-Protenix collapses.**
   On exp65, single-sequence ESMFold2 (0.782) is within ~0.02 of MSA Protenix
   (0.804), and even ESMFold (0.755) is close. This is exactly the regime
   MarinFold cares about, and it's where a strong single-sequence folder is
   most valuable.

**By MSA depth** (exp65 long-range R-precision; `plots/contacts_at_R_by_neff_tier.png`):
for **orphan** proteins (Neff≈1) ESMFold2 (0.748) is **on par with or above**
MSA-mode Protenix (0.725) and single-sequence Protenix (0.735); the MSA
advantage only opens up for deep-MSA proteins (MSA Protenix 0.813 vs ESMFold2
0.782). The single-sequence models lose nothing in the shallow-MSA regime.

Precision@L tells the same ranking but is bounded by pyconfind's contact
density for short proteins (ESMFold2 FoldBench @L = 0.73 agg / 0.52 long);
**R-precision is the fair read** and is flat across short/medium/long for
every model — i.e. the models rank contacts equally well at all separations.

**Timing** ([`data/timings.csv`](data/timings.csv)): ESMFold median **1.5 s**
/ protein (deterministic, single pass); ESMFold2 median **21 s** / protein
(best-of-5 diffusion; mean 34 s). Both H100.

Plots in [`plots/`](plots/): contacts @ L / L2 / L5 / **R** by model and
range; neff_tier and fold_verdict stratifications (at L and R);
FoldBench-vs-exp65. Raw predicted structures (all 552 × 2 models) + the full
pyconfind contact tables are on the HF bucket under
`data/esmfold-contacts-eval-exp78/`, so the predictions can be re-scored
under different contact criteria without re-running.

## Conclusion

On our 554-row eval set, **ESMFold2 is the strongest single-sequence
contact predictor** — its single-sequence pyconfind-contact R-precision
(FoldBench 0.805) approaches MSA-mode Protenix-v2 (0.847) and far exceeds
single-sequence Protenix (0.282), with ESMFold a notch below ESMFold2 but
still well above single-sequence Protenix on natural proteins. The headline
for MarinFold: **in the low-MSA / novel-fold regime the single-sequence vs
MSA gap nearly vanishes** — on exp65, single-sequence ESMFold2 (0.782) is
within 0.02 of MSA Protenix (0.804), and for orphan proteins ESMFold2 is on
par with or better than MSA Protenix. A capable single-sequence folder loses
little by not having an MSA exactly where MSAs are shallow — encouraging for
a single-sequence contact-prediction model. ESMFold2 costs ~14× ESMFold's
wall-clock (best-of-5 6B diffusion vs one deterministic pass) for a ~0.03–0.05
R-precision gain.

Caveats: the comparison is honest but not perfectly symmetric — ESMFold v1 is
single-shot while ESMFold2 is best-of-5 and Protenix is top-1-of-40; pyconfind
contact density caps precision@L (R-precision sidesteps this). All four
configs share one pyconfind ground truth and the GT-resolved candidate-pair
universe, so the rankings are comparable.
