---
marinfold_experiment:
  issue: 213
  title: 'exp: sequence-overlap audit of the contact eval set vs exp199''s training data, and a homology-free re-eval'
  kind: evals
  branch: claude/eval-sequence-overlap-analysis-33a77a
---

# exp: sequence-overlap audit of the contact eval set vs exp199's training data, and a homology-free re-eval

**Issue:** [#213](https://github.com/Open-Athena/MarinFold/issues/213) · **Kind:** `evals` · **Branch:** `claude/eval-sequence-overlap-analysis-33a77a`

## Question

**How much of our headline contact-prediction number survives if we throw out every eval protein that has a sequence homolog in the training set?**

The worry is concrete. Our training labels are contacts derived from **AlphaFold2 (AFDB) and ESMFold2 (ESM Atlas) structures** — predictors that had MSAs (AF2) or a large PLM (ESMFold2) available. MarinFold sees only a single sequence at inference, but if an eval protein has a close relative in the training corpus, the coevolution/PLM signal that produced that relative's structure is effectively *baked into the weights* and reachable by homology. In that regime "single-sequence contact prediction" is partly an MSA-derived retrieval, and our eval number is inflated relative to what it claims to measure.

## Hypothesis

Two competing predictions, and the experiment distinguishes them:

- **H1 (leakage):** MarinFold's R-precision drops sharply on eval proteins with no training homolog, and its gap to Protenix-v2 single-sequence (currently a tie at 554 proteins) opens up in Protenix's favor.
- **H0 (generalization):** accuracy is roughly flat across training-identity strata, as [#94](https://github.com/Open-Athena/MarinFold/issues/94) found for the much weaker #61 model against the AFDB corpus alone.

Prior from #94 leans H0, but that analysis predates both the ESM-Atlas half of the training data and the +0.26 R-precision of the current model, and its "no homolog" bin was 123/139 de novo *designed* proteins. Both weaknesses are addressed here.

## Background

- **The model.** `contacts-v1-exp199-1.5B` — [#199](https://github.com/Open-Athena/MarinFold/issues/199)'s CoreWeave arm `prot-exp199-cw-cv1-s02-m1-p06-aug` @ step 145199, MarinFold's current default. Trained from scratch on a **50/50 AFDB + ESM-Atlas** mixture, 152.3B tokens (≈16 AFDB epochs, ≈1.1 ESM-Atlas epochs — so the whole of both corpora was seen).
- **The two training corpora**, which together define "the train set" for this analysis:
  - `data/document_structures/contacts_v1/train/` — **4,129,682** documents from `timodonnell/afdb-24M`'s train split, up to 5 per structural cluster ([#53](https://github.com/Open-Athena/MarinFold/issues/53)). Labels are pyconfind contacts on **AlphaFold2** structures.
  - `data/document_structures/contacts_v1_esm_atlas/train/` — **66,759,922** documents, the ESM Atlas 40 %-identity linclust representatives ([#139](https://github.com/Open-Athena/MarinFold/issues/139)). Labels are pyconfind contacts on **ESMFold2** structures. All rows are `split=train`; there is no held-out ESM-Atlas partition.
  - ≈ **70.9 M training sequences** total.
- **The eval set.** The fixed 554-protein benchmark from [#89](https://github.com/Open-Athena/MarinFold/issues/89) (FoldBench-100 + 454 exp65 low-MSA / novel-fold candidates), with its frozen GT universe and `compute_metrics.py`.
- **Prior overlap work, and what each one is missing:**
  - [#41](https://github.com/Open-Athena/MarinFold/issues/41)/[#65](https://github.com/Open-Athena/MarinFold/issues/65) — Foldseek structural novelty + an MMseqs2 leakage check, but only against the **1.33 M AFDB cluster representatives**, and not tied to any accuracy number.
  - [#94](https://github.com/Open-Athena/MarinFold/issues/94) — the identity-stratified R-precision curve, but against the **AFDB corpus only** and for the **#61 model** (long-range R 0.353, vs #199's 0.564).
  - **Neither has ever looked at the 66.8 M ESM-Atlas sequences**, which are 94 % of the training corpus by count and are metagenomic — precisely the sequences most likely to supply remote homologs for CASP/CAMEO natural proteins.
- **Scoring.** Per-protein rows for #199 come from **exp82's `score_rollout_worker.py`**, the reference scorer per [#209](https://github.com/Open-Athena/MarinFold/issues/209)/[#212](https://github.com/Open-Athena/MarinFold/issues/212) (R-precision 0.6110 all / 0.5645 long — *not* #199's own pipeline's 0.5873/0.5422). Baseline per-protein rows come from #89's `contact_precision_all.csv` on the identical universe. **No new inference is required** for the headline comparison; it is a re-aggregation of existing per-protein scores over a new stratification.

## Approach

**Part 1 — measure the overlap** (new work).

1. Build the union training-sequence database (~70.9 M sequences, ~17 Gaa) by
   streaming both published document corpora — download a shard, recover its
   sequences, delete the parquet — so 146 GB passes through a few GB of disk:
   - AFDB: the 2,067 `contacts_v1/train/` shards (13 GB).
   - ESM-Atlas: the 3,338 `contacts_v1_esm_atlas/train/` shards (133 GB).
   - Neither corpus carries a `sequence` column — the sequence lives *inside*
     the document — so both are decoded with `contacts_v1.read.sequence_from_document`,
     the generator's exact inverse (added to the library here; exp94 had a private
     copy, which makes this the second consumer). `validate_sequences.py` checks
     the decoded AFDB sequences against the AlphaFold DB API.
   - Keep the **arm** (`afdb` / `esm_atlas`) on every target so per-arm attribution
     is possible.
2. MMseqs2-search all 554 eval sequences against it at high sensitivity (`-s 7.5`), recording for each eval protein the best hit overall and the best hit **per arm**: identity, query/target coverage, E-value, bitscore, and the hit count.
3. Emit a per-eval-protein novelty table with a graded ladder (`no detectable hit`, `<20 %`, `20–30 %`, `30–50 %`, `50–70 %`, `≥70 %` identity at ≥50 % query coverage) joined to the existing orthogonal axes: Foldseek fold verdict (#41/#65), MSA Neff (#65), viral/OOD flag and taxonomy (#94), dataset, and length.

**Part 2 — the homology-free re-eval** (the deliverable the question asks for).

4. Re-aggregate the per-protein R-precision / AUC / precision@{L, L/2, L/5} of **all six predictors on the same proteins** — MarinFold #199, Protenix-v2 single-seq, Protenix-v2 + MSA, ESMFold, ESMFold2, and #94's seq-KNN null — over each identity stratum, with **paired** bootstrap CIs for the MarinFold-minus-baseline differences (paired, because the subsets are small and the proteins are shared).
5. Headline cut, pre-registered before looking: the **"no detectable training homolog"** subset (no MMseqs2 hit at E ≤ 1e-3 against either arm).
6. Report every stratum **split by designed vs natural** (`denovo_pdb` vs the rest). #94's no-homolog bin was 88 % de novo designs, which structure predictors find easy; pooling the two re-introduces exactly the confound this experiment exists to remove.

**Explicitly out of scope / stated as a caveat, not controlled:** the structure baselines have their own training-set overlap with these eval proteins (Protenix is PDB-trained and much of the eval set *is* PDB). This experiment removes MarinFold's homology advantage, not everyone's. Any conclusion is therefore conservative for MarinFold — the baselines keep whatever leakage they have.

## Success criteria

- A committed per-eval-protein table giving max sequence identity (and hit count) to each of the two training arms for all 554 proteins — the reusable artifact, since every future eval can join on it.
- A statement of what fraction of the 554 has a training homolog at each identity tier, **broken out by training arm**, so we know how much the ESM-Atlas half added over the AFDB-only picture #94 measured.
- The headline table: R-precision (all + long) for MarinFold #199 and the five comparators on the homology-free subset, with n and paired CIs, and the same table on the full 554 for reference.
- A clear answer to H1 vs H0: does #199's accuracy depend on training-set sequence proximity, and does the Protenix-v2 single-seq tie survive homology removal?

## Code & how to run

Everything is CPU-only and local: ~146 GB streams through a few GB of disk,
one MMseqs2 search on this 64-core workstation, then pandas over per-protein
CSVs that already exist. **No cluster job and no model inference.**

| File | Role |
| --- | --- |
| `overlap_lib.py` | Shared contracts: the two training arms, the `{arm}\|{id}` FASTA header grammar, the identity/coverage conventions, the stratum ladder, the mmseqs installer. |
| `fetch_train_sequences.py` | **Step 1** — stream both document corpora and decode every training sequence to `train_{arm}.fasta`. |
| `validate_sequences.py` | **Step 1b** — check decoded AFDB sequences against the AlphaFold DB API (the reader is unit-tested; this checks the *chain*). |
| `search_overlap.py` | **Step 2** — build `data/eval_queries.fasta`, MMseqs2-search it against the union DB, reduce to one row per eval protein → `data/eval_train_identity.csv`. |
| `stratify_and_compare.py` | **Step 3** — re-aggregate every predictor's existing per-protein rows over the strata; paired bootstrap CIs → `data/strata_metrics.csv`, `data/paired_deltas.csv`, `data/headline.csv`. |
| `plot_overlap.py` | **Step 4** — the four figures. |
| `tests/test_overlap.py` | Pure unit tests for the header round trip, the alignment reduction, the stratum ladder and the paired bootstrap. |

```bash
uv sync --extra test
uv run --extra test pytest tests/                        # pure logic, <1 s

WORK=/data/exp213_overlap                                # needs ~80 GB free
uv run python fetch_train_sequences.py --arm both --workers 24 --work $WORK
uv run python validate_sequences.py --fasta $WORK/train_afdb.fasta -n 50
uv run python search_overlap.py --work $WORK             # the long step
uv run python stratify_and_compare.py
uv run python plot_overlap.py
uv run python build_summary.py                           # plots/summary.pdf
```

Smoke test the whole fetch/search path in ~2 minutes with
`--limit-shards 2 --work /data/exp213_smoke`; the resulting DB is far too
sparse for the verdicts to mean anything, but it proves the pipeline end to end.

**Two deviations from the plan written on the issue**, both discovered while
implementing:

1. ESM-Atlas sequences come from the **documents corpus**, not from
   `selected_manifest.parquet`. That manifest turns out to carry only
   `cluster_id / protein_hash / seq_len / mean_plddt / ptm / plddt_std /
   cluster_size` — no sequences at all. The only other source is
   `structures/parts/`, which is 2.08 TB with the sequence inlined next to the
   cif, so the 133 GB document corpus is both the cheapest and the most
   faithful source (it is *by construction* what the model read).
2. The document→sequence decoder was **promoted into the `contacts_v1`
   library** rather than copied from exp94, since this is its second consumer
   (`experiments/AGENTS.md` rule 7). It ships with a round-trip test against
   the real generator.

## Results

All 554 eval proteins searched against **70,889,604** training sequences —
**4,129,682** AFDB + **66,759,922** ESM-Atlas, both matching their corpora's
published document counts exactly (#94's AFDB index and #139's ESM-Atlas
generation respectively). MMseqs2 `-s 7.5`; the whole search took 7 minutes.
Per-protein table: [`data/eval_train_identity.csv`](data/eval_train_identity.csv).

**The decoded sequences are the training set's, verified externally.** 39/39
resolvable entries in a 40-record AFDB sample are byte-identical to AlphaFold
DB's `uniprotSequence` ([`data/sequence_validation.json`](data/sequence_validation.json)).
And the pipeline reproduces #94: 434/554 eval proteins have *some* AFDB
alignment here vs its 415, so the two agree on the raw hit rate and differ only
in that #94 counted any alignment while this applies an E ≤ 1e-3 significance
line.

### 1. The eval set is homology-rich, and ESM-Atlas is the bigger contributor

| | proteins | % |
| --- | ---: | ---: |
| Significant homolog in **either** arm | **323** | 58 % |
| … in both arms | 248 | 45 % |
| … only AFDB | 27 | 5 % |
| … only ESM-Atlas | 48 | 9 % |
| **No significant homolog** (E ≤ 1e-3) | **231** | 42 % |
| **No MMseqs2 alignment at all** (E ≤ 10) | **62** | 11 % |

ESM-Atlas alone hits 296/554 proteins, *more* than AFDB's 275, and adding it
shrinks the homology-free set from 269 (AFDB only) to 231. So the half of the
training data nobody had checked does supply real homology — 48 eval proteins
have a training relative that exists **only** in the metagenomic corpus.

→ [`plots/overlap_profile.png`](plots/overlap_profile.png)

### 2. Sequence novelty is not fold novelty

Of the 231 sequence-novel proteins, only **37** are also Foldseek-novel against
the AFDB training representatives; 133 are `same_fold` and 61 `redundant`
([`data/sequence_vs_fold_novelty.csv`](data/sequence_vs_fold_novelty.csv)).
Removing sequence homologs leaves most of the *fold* space intact — and for
contact prediction the fold is the channel that carries information. #94 saw
the same thing against AFDB alone; it survives the ESM-Atlas addition.

### 3. MarinFold's accuracy does not track training-set identity

Spearman ρ between best training identity and R-precision, over the 315
proteins that have a covered hit ([`data/identity_slopes.csv`](data/identity_slopes.csv)):

| predictor | ρ (all) | ρ (natural only) |
| --- | ---: | ---: |
| **MarinFold #199** | **−0.117** [−0.231, −0.008] | **+0.042** [−0.151, +0.229] |
| Protenix-v2 single-seq | −0.354 [−0.444, −0.261] | +0.044 [−0.129, +0.228] |
| ESMFold | −0.071 [−0.180, +0.038] | −0.049 [−0.244, +0.143] |
| ESMFold2 | +0.077 [−0.029, +0.181] | +0.075 [−0.120, +0.258] |
| Protenix-v2 + MSA | +0.095 [−0.014, +0.203] | +0.030 [−0.170, +0.244] |
| **seq-KNN k=10 (null)** | **+0.534** [+0.442, +0.621] | **+0.357** [+0.161, +0.524] |

seq-KNN is the calibration: a copy-the-nearest-neighbour model *must* track
identity, and it does, strongly. MarinFold does not. **A model whose score came
from retrieving memorised homologs would look like the last row; it looks like
the rows above it.** This is #94's finding, now against the whole training set
and for a model 0.26 R-precision stronger.

→ [`plots/rprecision_vs_identity.png`](plots/rprecision_vs_identity.png)

### 4. But the parity with Protenix-v2 single-seq does not survive

R-precision (all ranges), and the paired MarinFold-minus-baseline difference
with a 95 % bootstrap CI ([`data/headline.csv`](data/headline.csv)):

| | all 554 | no homolog (231) | no hit at all (62) |
| --- | ---: | ---: | ---: |
| **MarinFold #199** | **0.611** | **0.549** | **0.496** |
| Protenix-v2 single-seq | 0.603 (**+0.008** [−0.017, +0.033]) | 0.718 (−0.169 [−0.198, −0.139]) | 0.711 (−0.215 [−0.289, −0.136]) |
| ESMFold | 0.755 (−0.144) | 0.697 (−0.149) | 0.706 (−0.210) |
| ESMFold2 | 0.786 (−0.175) | 0.748 (−0.199) | 0.778 (−0.282) |
| Protenix-v2 + MSA | 0.812 (−0.201) | 0.764 (−0.215) | 0.722 (−0.227) |
| seq-KNN k=10 (null) | 0.345 (+0.266) | **0.035** (+0.514) | **0.011** (+0.485) |

The **difference of differences** ([`data/interaction.csv`](data/interaction.csv)) is
the actual test — two subsets' CIs read side by side are not one, because the
subsets contain different proteins:

| comparator | effect, pooled | effect, natural only |
| --- | ---: | ---: |
| **Protenix-v2 single-seq** | **−0.303** [−0.346, −0.259] | **−0.327** [−0.407, −0.249] |
| ESMFold | −0.007 [−0.039, +0.025] | **+0.099** [+0.028, +0.169] |
| ESMFold2 | −0.041 [−0.078, −0.004] | **+0.077** [+0.007, +0.146] |
| Protenix-v2 + MSA | −0.025 [−0.063, +0.015] | +0.024 [−0.065, +0.114] |
| seq-KNN k=10 (null) | +0.425 [+0.380, +0.473] | +0.361 [+0.292, +0.432] |

**The effect is specific to Protenix-v2 single-seq.** MarinFold gives up
0.30–0.33 of R-precision relative to it on homology-free proteins, while its
standing against ESMFold and ESMFold2 on natural proteins *improves*, and
against Protenix+MSA does not move. That asymmetry is the opposite of what
uniform homology leakage predicts: a model reaching MSA-derived signal through
its training homologs should lose ground against **every** comparator when
those homologs are removed.

→ [`plots/delta_across_subsets.png`](plots/delta_across_subsets.png),
[`plots/headline_homology_free.png`](plots/headline_homology_free.png)

### 5. On genuinely novel proteins, MarinFold is well behind everything

The strictest cut — no sequence homolog **and** Foldseek-novel vs the AFDB
training folds (n=37, of which 19 natural):

| predictor | R-precision |
| --- | ---: |
| **MarinFold #199** | **0.328** |
| Protenix-v2 single-seq | 0.547 |
| ESMFold | 0.547 |
| ESMFold2 | 0.620 |
| Protenix-v2 + MSA | 0.670 |
| seq-KNN k=10 (null) | 0.019 |

### Two things that shape how far this can be pushed

- **The homology-free subsets are 80 % de novo designs** (184/231; 52/62 for
  the strict cut). Designed proteins have no homologs anywhere by construction
  and structure predictors find their idealised backbones easy, so every number
  above is also reported split. The natural-only homology-free set is **n=47**
  — and n=10 under the strict definition, where the CIs stop being informative.
  That is the binding constraint on this eval set, not the analysis.
- **The baselines keep their own leakage.** Protenix is PDB-trained and much of
  this eval set *is* PDB; ESMFold2's PLM saw UniRef. This experiment removes
  MarinFold's homology advantage and nobody else's, so every comparison here is
  conservative *for* MarinFold — which makes result 4 harder to explain away,
  not easier.

## Conclusion

**The headline eval number is not substantially inflated by homology leakage,
but one specific claim we make from it is.**

The eval set is homology-rich — 58 % of it has a training relative, and only
11 % has no detectable alignment at all — so the concern was well founded as a
question. The answer, though, is that MarinFold's accuracy does not depend on
that proximity. Its Spearman ρ against training identity is −0.12 pooled and
+0.04 on natural proteins, where the seq-KNN null that works *only* by
homology transfer sits at +0.53. Restricting to the 231 proteins with no
significant training homolog moves R-precision 0.611 → 0.549, and against
ESMFold/ESMFold2 on natural proteins MarinFold actually gains ground. This is
**H0**, and it extends #94's AFDB-only result to the full 70.9 M-sequence
training set and to a model 0.26 R-precision stronger.

The exception is sharp and worth acting on. **Our parity with Protenix-v2
single-sequence (#180's +0.008 tie on 554 proteins) is a homology-dependent
result.** On proteins with a training homolog MarinFold leads it by +0.13; on
proteins without one it trails by −0.17, a shift of −0.303 [−0.346, −0.259]
that is 100× the tracker's 0.0023 noise floor. On the 37 proteins that are
novel in both sequence and fold, MarinFold scores 0.328 against Protenix-SS's
0.547 and ESMFold2's 0.620. So "we have caught up with single-sequence
Protenix" should be stated as "on proteins resembling our training data" until
that gap closes.

Because the effect does not appear against ESMFold, ESMFold2 or Protenix+MSA,
the most likely reading is not that MarinFold leaks, but that **Protenix-v2
single-seq is differentially strong exactly where MarinFold is weak** — the
novel-fold, low-homology regime. Distinguishing "MarinFold is weak on novel
folds" from "Protenix-SS is strong on them" needs a predictor-side experiment,
not another eval-set audit; that is the natural follow-up.

Two things to carry into any future eval work:

1. **Report the homology-free subset alongside the headline.**
   [`data/eval_train_identity.csv`](data/eval_train_identity.csv) is committed
   per-protein and joins on `(dataset, stem)`, so it costs one merge.
2. **The current eval set cannot answer this question much better than it just
   did.** The natural, homology-free, fold-novel corner is n≈19. If we want to
   measure novel-protein performance rather than bound it, the eval set needs
   more natural low-homology proteins — which is what #65 was originally for,
   and is now the clearest gap.
