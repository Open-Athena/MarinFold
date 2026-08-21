---
marinfold_experiment:
  issue: 250
  title: 'exp: an interactive evals-exploration notebook — per-protein contact maps and the predictor scoreboard'
  kind: evals
  branch: exp250/evals-exploration-notebook
---

# exp: an interactive evals-exploration notebook — per-protein contact maps and the predictor scoreboard

**Issue:** [#250](https://github.com/Open-Athena/MarinFold/issues/250) · **Kind:** `evals` · **Branch:** `exp250/evals-exploration-notebook`

## Question

**What does the current contacts-v1 model actually get right and wrong, protein by protein — and can we look at that interactively, from a browser, without a cluster?**

Every eval number we publish today is an aggregate: a mean R-precision over an eval
set, sometimes split by designed / natural / viral. Those aggregates are the right
thing for tracking progress ([#180](https://github.com/Open-Athena/MarinFold/issues/180))
but they are a poor instrument for *understanding* the model. The questions that keep
coming up — which proteins does it fold and which does it lose, does the predicted
contact map look like a wrong fold or like noise, what changed between two
checkpoints on one protein — all need a per-protein view that nothing in the repo
currently offers in one place.

The published artifacts to answer this already exist and are all public
(anonymous read): [#245](https://github.com/Open-Athena/MarinFold/issues/245)'s
per-protein scores for 9 predictors, its eval-set annotation, ground truth for both
universes, and [#247](https://github.com/Open-Athena/MarinFold/issues/247)'s 75
per-protein features. What is missing is a place to put them together and a GPU path
to generate a contact map for an arbitrary protein under the settled inference recipe.

## Hypothesis

Not a hypothesis experiment — this is an instrument. The success criterion is
fidelity: a contact map and score produced in the notebook must reproduce the
published number for the same protein and checkpoint, so that anything read off
the notebook is on the same axis as everything we have already filed.

The one substantive prior: per-protein accuracy is *bimodal*, not a smooth
distribution — the model either finds roughly the right fold or produces a map with
no correct long-range structure — and looking at maps will make that visible in a
way the aggregate R-precision cannot.

## Background

- [#82](https://github.com/Open-Athena/MarinFold/issues/82) settled the inference
  recipe: 100 rollouts + per-rollout document resampling, vote counting over pairs.
  Numbers under the older pairwise readout are ~0.086 lower and not interchangeable.
- [#89](https://github.com/Open-Athena/MarinFold/issues/89) is the measurement
  specification (ground-truth universe, candidate pairs, metric implementation), and
  published per-protein score matrices + three heatmaps for the #75 model.
- [#245](https://github.com/Open-Athena/MarinFold/issues/245) cut FoldBench's 334
  monomers into eval-val / eval-test / eval-denovo and published per-protein scores
  for 3 checkpoints and 5 baselines, plus the reporting rules (never a pooled mean;
  baseline comparisons only on proteins postdating the baselines' cutoffs;
  eval-test has a read budget).
- [#247](https://github.com/Open-Athena/MarinFold/issues/247) published 75
  per-protein features (contact order, homology to training, MSA depth, secondary
  structure, annotations) for 314 of them.
- [#142](https://github.com/Open-Athena/MarinFold/issues/142) established the
  notebook pattern: a Colab notebook in `notebooks/` that clones the repo, installs
  `marinfold[transformers]`, and runs rollouts on the free GPU tier.

## Approach

One Colab notebook, `notebooks/evals_exploration.ipynb`, in four parts:

1. **Scoreboard** (CPU) — R-precision / AUC by predictor for a chosen eval set,
   with bootstrap CIs, natural and designed reported separately, and the #245
   reporting rules enforced in the output rather than left to the reader.
2. **Per-protein browser** (CPU) — every protein × every predictor joined to the
   #245 annotation and the #247 features; sort by where MarinFold wins or loses,
   scatter against any baseline.
3. **Contact maps** (GPU) — run any registered checkpoint on any eval protein
   under the #82 recipe, plot the vote matrix against ground truth, and score it
   with the #89 metric implementation.
4. **Checkpoint comparison** (GPU) — the same protein under two checkpoints,
   side by side, with the paired delta.

Both universes are supported and kept strictly separate: the #245 FoldBench-monomer
sets (333 units) and the legacy 554. They overlap in 112 stems but disagree on the
input sequence for 11 of them, so a mixed comparison would be silently wrong.

## Success criteria

- A protein scored in the notebook reproduces its published #245 per-protein
  R-precision within rollout noise (the recipe is stochastic; #204 puts four
  evaluations of one unchanged checkpoint within 0.0023 in aggregate, per-protein
  variance is larger).
- Parts 1–2 run with no GPU and no HF token.
- Part 3 runs end to end on a free-tier Colab GPU for a typical (L < 300) protein.
- Selecting eval-test surfaces the read-budget warning and the pointer to
  `eval_test_reads.md`.

## Results

The notebook is [`notebooks/evals_exploration.ipynb`](../../notebooks/evals_exploration.ipynb)
([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Open-Athena/MarinFold/blob/main/notebooks/evals_exploration.ipynb)).
Four parts, in one Colab: the **scoreboard** (§1), the **per-protein browser** (§2), a
**contact map** for any eval protein under any registered checkpoint (§3), and the **same
protein under two checkpoints** (§4). §1–2 are CPU-only and token-free; §3–4 need a GPU
runtime and download a 1.5B checkpoint from the public bucket.

### 1. Where every number in it comes from

All public, all anonymous-readable, all reached over plain `https://huggingface.co/buckets/…/resolve/…`
URLs — `marinfold` pins `transformers<5`, which holds `huggingface_hub<1.0`, so the `hf_hub`
bucket API is not importable in the same environment.

| what | source |
|---|---|
| FoldBench targets + input sequences | `data/contacts-v1-foldbench-monomers-exp245/eval_targets_foldbench_monomers.parquet` |
| eval-set / designed / viral / identity annotation | `…-exp245/eval_sets.csv` |
| FoldBench ground truth | `…-exp245/gt_universe_scored.jsonl` |
| FoldBench per-protein scores, 9 predictors | `…-exp245/per_protein.csv.gz` |
| legacy-554 ground truth | `data/contacts-v1-model-eval-exp89/gt_universe.jsonl` |
| legacy-554 per-protein scores (#75, ESMFold, ESMFold2, Protenix-v2 ×2) | `…-exp89/contact_precision_all.csv` |
| legacy-554 per-protein scores (#199 cooldown) | `data/contacts-v1-model-eval-exp199/replicates/cooldown-v2-20260815-01/…/…_rows.csv.gz` |
| 75 per-protein features | `data/contacts-v1-protein-properties-exp247/protein_features.csv` |
| legacy-554 input sequences | in-repo: `experiments/exp94_evals_sequence_knn_baseline/data/eval_queries.fasta` |
| checkpoints | `marinfold.registry.resolve_model` over `MODELS.yaml` |

[`verify_sources.py`](verify_sources.py) checks every one of those is reachable and still has the
columns the notebook reads, and writes [`data/source_check.json`](data/source_check.json). Run it
if a notebook cell starts failing — bucket drift, not the notebook, is the likely cause.

### 2. Fidelity — the notebook is on the same axis as the published numbers

Five checks, each a number rather than an assurance.

**The legacy-554 sequences are the ones #89 actually prompted with.** They are not published as a
table, so the notebook reads exp94's query FASTA. Decoding the sequence section back out of
`ensemble_prompts.parquet` (#89's own prompts) and comparing gives **554/554 byte-identical** —
the FASTA is the right file.

**The metric implementation matches #89's universe exactly.** For `1qys_A` the notebook computes
3,741 candidate pairs / 76 true contacts, and for `8ah9_A` 5,995 / 132 — identical to the
`n_candidate` / `n_true` columns of the published rows.

**The scoreboard reproduces the published aggregates.** On legacy-554 the #199 cooldown comes out
0.685 designed (n=396) / 0.495 natural (n=158), which pools to **0.631** — the published figure. On
eval-val the #232 checkpoints come out **0.520** (m2-p06) and **0.473** (m1-p02), matching the
`eval-checkpoint` reference table to the digit.

**Per-protein, 100 rollouts, #199 cooldown, against the published per-protein score:**

| unit | notebook | published | delta | wall-clock (A5000) |
|---|---:|---:|---:|---:|
| `8ah9_A` (eval-denovo) | 0.909 | 0.894 | +0.015 | 36 s |
| `8arl_A` (eval-val, the worst protein on the set) | 0.150 | 0.100 | +0.050 | 45 s |
| `7y5r_A` (eval-val) | 0.825 | 0.835 | −0.010 | 27 s |
| `denovo_pdb__1qys_A` (legacy-554, Top7) | 0.684–0.697 | 0.697 | −0.013…0.000 | 23 s |

That is rollout noise on a single protein, which is much wider than the 0.0023 aggregate noise
floor #204 measured over 554 proteins — as expected, and the reason §3 is an instrument for looking
at maps rather than a way to produce numbers of record.

**Two known divergences from the eval harness**, both stated in the notebook itself: it runs under
`transformers` rather than vLLM, and it uses the packaged rollout, which adds the pairwise
tie-break (`votes + [0, 0.5)`) that #245's worker does not apply. The tie-break can only reorder
pairs that are tied on votes, so it moves R-precision when the R cut lands inside a tie group and
improves AUC by ordering the zero-vote mass.

### 3. Three hazards the notebook encodes rather than documents

**Two universes that look joinable and are not.** legacy-554 and #245's 333 FoldBench monomers
overlap in 112 stems, but 11 of those have **different input sequences** under the two (different
chain resolution). The notebook loads one universe at a time and never joins across them.

**Duplicate stems inside legacy-554.** `7ur7_A` and `8ah9_A` each appear twice, under two datasets,
with different sequences and lengths (70 vs 63, 120 vs 115 residues). Keying per-protein tables on
`stem` silently duplicates them and averages two different proteins; everything here is keyed on
`(dataset, stem)`, and `fold()` raises on an ambiguous stem rather than picking one.

**Predictor tables that cover a superset.** The #199 cooldown was also scored on #226's 23 extra
FoldBench chains, which are not in the 554. The notebook drops out-of-universe rows and says how
many, so a mean is always over exactly the set it names.

Selecting `eval-test` prints the read-budget warning and the pointer to
[`eval_test_reads.md`](../exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md); the
`#245` reporting rules (designed and natural never pooled, baseline comparisons only where the
baselines' cutoffs allow it, ~0.005 is a tie) are in the §1 prose and in the split of the output.

### 4. Cost

§1–2 are a few MB of downloads and run in seconds. §3 is ~25–45 s per protein per checkpoint at
100 rollouts on an A5000 (L ≤ 300), plus a one-time ~5.5 GB checkpoint download; a free Colab T4 is
several times slower but well within a session.

## Conclusion

The instrument exists and is calibrated: the scoreboard reproduces the published aggregates to the
digit, and a per-protein map produced in the notebook lands within rollout noise of the published
per-protein score for the same checkpoint. It reads published artifacts and re-runs a settled
recipe — it does not produce eval numbers of record, and anything worth citing still goes through
[#245](https://github.com/Open-Athena/MarinFold/issues/245)'s harness.

Adjacent notebooks, so the next person picks the right one:
[`inference_example1.ipynb`](../../notebooks/inference_example1.ipynb) runs a checkpoint on an
arbitrary RCSB entry (its own ground truth, vLLM or transformers, no eval set);
[`fold_from_contacts1.ipynb`](../../notebooks/fold_from_contacts1.ipynb) turns predicted contacts
into a 3D backbone. This one is the only one anchored to the eval universes and the published
per-protein scores.
