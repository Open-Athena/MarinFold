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

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
