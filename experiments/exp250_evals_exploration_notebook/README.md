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

**The scoreboard reproduces #245's eval-test row exactly.** Reading the held-out cut through the
notebook gives 0.538 / 0.493 / 0.613 for m2-p06 / m1-p02 / the cooldown, and 0.265 / 0.753 / 0.792 /
0.845 / 0.582 / 0.426 for Protenix-v2 single-seq / ESMFold / ESMFold2 / Protenix-v2 + MSA /
seq-KNN unfiltered / seq-KNN decontaminated — every figure identical to row 1 of
[`eval_test_reads.md`](../exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md).
No new ledger row was added for it: this re-displays published numbers rather than scoring
anything, and the ledger asks specifically not to accumulate routine entries.

**Per-protein, 100 rollouts, #199 cooldown, against the published per-protein score:**

| unit | notebook | published | delta | wall-clock (A5000) |
|---|---:|---:|---:|---:|
| `8ah9_A` (eval-denovo) | 0.909 | 0.894 | +0.015 | 36 s |
| `8arl_A` (eval-val, the worst protein on the set) | 0.150 | 0.100 | +0.050 | 45 s |
| `7y5r_A` (eval-val) | 0.825 | 0.835 | −0.010 | 27 s |
| `denovo_pdb__1qys_A` (legacy-554, Top7) | 0.684–0.697 | 0.697 | −0.013…0.000 | 23 s |

**The same check for `#232 m2-p06`, folded from the copy published here** (below), which is what
proves the publish is sound end to end — bucket copy, rope repair, tokenizer and all:

| unit | notebook | published | delta |
|---|---:|---:|---:|
| `8ah9_A` | 0.902 | 0.894 | +0.008 |
| `7y5r_A` | 0.784 | 0.763 | +0.021 |
| `7t9r_A` | 0.250 | 0.167 | +0.083 |

`7t9r_A` is L=38 with 12 true contacts, so its whole `+0.083` is one extra contact called right.

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

### 4. Publishing #232's best decontaminated checkpoint

Parts 1–2 could always read `#232 m2-p06`'s published *scores*, but nothing could fold with it: the
weights existed only in CoreWeave S3, in the account that trained them
(`hf_repo_id=None` in both #232's and #245's pinned specs), and this workstation has no credentials
for `marin-us-east-02a`. [`publish_exp232_m2_p06.py`](publish_exp232_m2_p06.py) copies the 5.9 GiB
export to `checkpoints/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/hf/step-145199/` on the public
bucket, from a `cw-us-east-02a` pod beside the bytes rather than through the workstation's
~2.5 MB/s uplink.

It is exp238's `publish_cooldown.py` adapted — same mechanism, same three pre-upload checks, each
of which is silent when it goes wrong:

| check | result |
|---|---|
| every object matches the size + S3 ETag #244/#245 pinned | 6/6 |
| `config.json` rope restated in transformers-4.x terms | `rope_theta=500000`, `rope_scaling=llama3` (4.x would otherwise silently use the Qwen3 default 10000) |
| contacts-v1 vocabulary has not drifted | 2,845 tokens, `<contact>`/`<p0>`/`<p1999>` ids unmoved |

[`test_publish_specs.py`](test_publish_specs.py) asserts the by-value manifest still equals #245's
`M2_P06_CHECKPOINT` and is not `m1_p02`'s — the two finals differ only in their weight ETags, so a
copy-paste slip would publish the wrong checkpoint under the right name.

Registered as `contacts-v1-exp232-m2-p06-1.5B` in `MODELS.yaml`, and it is deliberately **not** the
default: the contaminated cooldown scores higher everywhere (0.631 vs 0.5916 on the legacy 554,
0.589 vs 0.520 on eval-val). This checkpoint is the one to reach for when the question is what the
recipe achieves on proteins it provably cannot have seen — #213 measured the eval set as 58 %
homologous to #199's training data, and that objection does not apply here.

The notebook's part 4 now defaults to the contamination contrast: the default checkpoint against
this one, same protein, side by side.

### 5. What the decontaminated checkpoint says about leakage

The notebook's paired-contrast cell exists for this question, and #232's two checkpoints are what
make it answerable. Over the 314 natural FoldBench monomers, paired protein by protein:

| cut | m2-p06 | #199 cooldown | paired delta |
|---|---:|---:|---:|
| eval-val (97) | 0.520 | 0.589 | −0.069 |
| eval-test (217) | 0.538 | 0.613 | −0.076 |
| all natural (314) | 0.532 | 0.606 | **−0.074** [−0.086, −0.062] |
| eval-denovo (19) | 0.591 | 0.619 | −0.027 |

m2-p06 is ahead on 13 % of proteins. Broken out by each protein's best identity to #199's
(un-decontaminated) training set:

| stratum | n | m2-p06 | cooldown | delta |
|---|---:|---:|---:|---:|
| no homolog | 14 | 0.257 | 0.266 | **−0.009** |
| 20–30 % | 6 | 0.401 | 0.455 | −0.054 |
| 30–50 % | 49 | 0.546 | 0.608 | −0.063 |
| 50–70 % | 105 | 0.570 | 0.644 | −0.073 |
| 70–100 % | 140 | 0.532 | 0.617 | −0.085 |

**Read this carefully rather than as a leakage result.** The gap does look smallest exactly where
the contaminated model has nothing to have memorised — 14 proteins with no training homolog, and
the 19 viral ones (−0.029) — but two things argue against reading that as the mechanism: those
proteins are far harder for *both* models (0.26 against 0.61), so a small absolute gap is partly a
floor effect; and among the 300 proteins that do have a homolog, identity does not rank-order the
gap at all (Spearman +0.001). That is the same answer #213 got from the other direction (score
uncorrelated with training identity, against seq-KNN's +0.53) and the same one #245 got from the
eval-val → eval-test drop.

Against the null that matters for it — a sequence-KNN predictor built from the corpus it actually
trained on — m2-p06 clears by **+0.112** (0.532 against 0.420). It trails ESMFold2 by 0.263 and
leads Protenix-v2 single-seq by 0.268 on the same 314 proteins.

### 6. Manuscript figure panels (part 5)

Part 5 writes standalone panels to `figures/` — PNG at 300 dpi and PDF — with no titles and no
panel letters, plus `contacts_v1_document_top7.txt`, the real contacts-v1 document for Top7 so a
format panel can be drawn from the thing itself.

Both protein classes are the same in the contact and structure figures, which is the point:
**natural** = the 314 FoldBench natural monomers (eval-val + eval-test), **designed** = FoldBench's
19 de novo monomers. eval-denovo is small, but exp65's 396 designs cannot be used against
baselines — 50.5 % of them predate Protenix-v2's 2021-09-30 cutoff — and 19 is the set Helico's
GDT-TS covers.

R-precision, from #245's published per-protein scores:

| predictor | natural (314) | designed (19) |
|---|---:|---:|
| Protenix-v2 + MSA | 0.845 | 0.844 |
| ESMFold2 | 0.795 | 0.864 |
| ESMFold | 0.752 | 0.795 |
| MarinFold (#199 cooldown) | 0.606 | 0.619 |
| MarinFold (#232 m2-p06, decontaminated) | 0.532 | 0.591 |
| Protenix-v2, single sequence | 0.264 | 0.835 |

GDT-TS, from [Open-Athena/helico](https://github.com/Open-Athena/helico) exp14's published
per-target scores (`hf://buckets/timodonnell/helico-experiments/exp14_foldbench_held_out_monomers/`,
anonymous read), restricted to the 324 targets every arm scored:

| arm | natural (305) | designed (19) |
|---|---:|---:|
| Helico + true contacts | 0.893 | 0.920 |
| Protenix-v2 + MSA | 0.868 | 0.860 |
| ESMFold2 | 0.814 | 0.934 |
| **Helico + MarinFold contacts** | **0.479** | **0.761** |
| Protenix-v2, single sequence | 0.174 | 0.892 |
| Helico, no contacts | 0.146 | 0.859 |

**The two classes tell opposite stories and the figures should not be captioned as though they
agree.** On natural monomers MarinFold's contacts take Helico from 0.146 to 0.479 GDT-TS, against
0.174 for Protenix-v2 single-sequence — the contacts are carrying real structural information no
single-sequence predictor has. On the 19 designed monomers they take it from 0.859 **down** to
0.761: designed backbones are idealised, single-sequence predictors already handle them
(Protenix-v2 single-seq scores 0.892 GDT-TS and 0.835 contact R-precision there), and imperfect
contacts subtract. The same asymmetry is in the contact panels — the designed gap between
MarinFold and Protenix-v2 single-seq is +0.24 in our favour on natural proteins and −0.24 against
us on designs.

### 7. Hardware notes, measured rather than assumed

**float16 does not work with these weights.** They were trained in bfloat16, and fp16 overflows the
residual stream: folding Top7 under `dtype="float16"` dies with a CUDA device-side assert inside
`sample_completions` — NaN logits reaching the sampler — in a fresh process, so it is not
cross-run contamination. float32 folds the same protein fine (R-precision 0.671 at 20 rollouts).
This matters because fp16 is the obvious choice on a T4, where bfloat16 has no tensor cores and
runs emulated. The notebook therefore uses **bfloat16 on every GPU**, keeps float16 in the
dropdown with a warning, and offers float32 as the safe fallback.

**Rollout batch size comes from free VRAM**, not a constant: the KV cache is 48 KiB per token per
rollout for this architecture (24 layers, 8 KV heads, head dim 64). Measured on the workstation's
A5000, Top7 runs all 100 rollouts in one pass with 42 MiB of cache each. A 16 GB T4 does the same
for Top7 and steps down to 14 rollouts per pass for the longest protein in the set (1,596
residues) instead of running out of memory. The transformers backend then applies its own
roughly-constant-cache heuristic on top, which is what actually binds past ~200 residues.

**vLLM** is used automatically at compute capability >= 8.0, which is also what exp245's harness
ran. It does not release an engine when one goes out of scope and every fold builds one, so
`GPU_MEMORY_UTILIZATION` defaults to 0.28 — enough for the three a session running parts 3, 4 and
5 accumulates.

### 8. Cost

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
