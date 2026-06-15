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

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
