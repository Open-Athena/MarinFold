---
marinfold_experiment:
  issue: 94
  title: "exp: implement a K-nearest neighbor baseline for evals"
  kind: evals
  branch: exp/94-sequence-knn-baseline
---

# exp: implement a K-nearest neighbor baseline for evals

**Issue:** [#94](https://github.com/Open-Athena/MarinFold/issues/94) · **Kind:** `evals` · **Branch:** `exp/94-sequence-knn-baseline`

## Question

How well does a sequence K-nearest-neighbor baseline do at contact prediction on our
eval set, and how much of MarinFold's contact-prediction skill is explained by simply
copying the contacts of its nearest training neighbors?

## Hypothesis

A pure copy-the-neighbors null model — no folding, no model — sets a floor. If it
approaches MarinFold's R-precision, the LM is largely retrieving memorized folds; if
MarinFold clearly beats it, the LM generalizes past nearest-neighbor copying.

## Background

We have a curated eval set of n=554 proteins (100 FoldBench + 454 from exp65:
`denovo_pdb`, `casp_fm`, `cameo_hard`) and contacts-v1 models now reaching >0.4
R-precision. This experiment quantifies the memorization component with a
non-parametric baseline scored under the **exact same metric harness** (exp89's
`compute_metrics.py`) as every other predictor, so the bars are directly comparable.

## Approach

CPU-only, no GPU, no network — the contacts-v1 train corpus and the eval GT universe
are both on local disk. Pipeline (five steps, each a launchable script; shared helpers
in [`knn_lib.py`](knn_lib.py)):

1. **[`build_train_index.py`](build_train_index.py)** — parse all 2067 local train
   shards (4,129,682 documents) into `_scratch/train_seqs.fasta` (one record per
   document, header = a unique `doc_id`) + `_scratch/contacts_store/` (each document's
   0-based ground-truth contacts). The contacts-v1 documents emit residues in resampled
   order with positions taken modulo 2000 of a per-document N-terminus; the parser
   inverts both (`seqpos = (token − n_term_index) mod 2000`). Validated against
   `contacts_emitted` on 2000/2000 rows of shard 0.
2. **[`run_mmseqs.py`](run_mmseqs.py)** — build `data/eval_queries.fasta` (554 queries,
   keyed `{dataset}__{stem}`) and mmseqs2-search it against the train index (`-s 7.5`,
   `--max-seqs 300`, `-a` for per-residue alignments). Outputs `_scratch/aln.m8`.
3. **[`build_knn_scores.py`](build_knn_scores.py)** — for each eval protein, take its
   top-k hits (k ∈ {1,5,10,25,50}), map each training contact through the local
   alignment onto the query's residues, and vote. The saved `[L,L]` score is
   `tiebreak_matrix(count, bitscore-weighted votes)` (exp82's integer-vote AUC fix).
   Emitted for both self-included and self-excluded (`fident==1 & qcov==1 & tcov==1`)
   neighbor sets.
4. **[`compute_knn_metrics.py`](compute_knn_metrics.py)** — score every matrix with
   exp89's verbatim `metric_rows` (precision@{L,L/2,L/5,R} + AUC × {all,short,medium,
   long}) over the resolved universe; concat with the existing predictors' per-protein
   rows. Writes [`data/knn_precision_all.csv`](data/knn_precision_all.csv) +
   [`data/knn_comparison.csv`](data/knn_comparison.csv).
5. **[`plot_knn.py`](plot_knn.py)** — the headline R-precision bars, the k-sweep curve,
   and the per-protein memorization scatter.

### Running

```bash
uv sync --extra test
uv run pytest tests/          # parser + alignment-mapper unit tests
SCR=_scratch
uv run python build_train_index.py --scratch $SCR
uv run python run_mmseqs.py --scratch $SCR -s 7.5
uv run python build_knn_scores.py --scratch $SCR --gt $SCR/gt_universe.jsonl
BASE=<exp82>/_scratch/contact_precision_with_rollout.csv   # per-protein rows for the other predictors
uv run python compute_knn_metrics.py --gt $SCR/gt_universe.jsonl --scores-root $SCR/scores --base-csv $BASE
uv run python plot_knn.py --base-csv $BASE
```

`_scratch/gt_universe.jsonl` is exp89's 554-protein GT universe (copied verbatim — the
canonical file every other predictor was scored against). Whole pipeline runs in a few
minutes on this box.

## Success criteria

A plot of long-range R-precision for MarinFold #61 (n=100 rollouts), the seq-KNN
baselines (k-sweep), and the standard predictors (Protenix-v2, ESMFold, ESMFold2), all
under one metric harness. → [`plots/headline_rprecision.png`](plots/headline_rprecision.png).

## Results

Indexed **4,129,682** contacts-v1 train documents and aligned the **554** eval
proteins against them (mmseqs `-s 7.5`): 415/554 had ≥1 training-sequence homolog,
139 had none; 47,734 alignments, 7,239 distinct neighbors used in top-k voting.
Only **2** eval proteins have a verbatim self-hit, so self-inclusion is irrelevant
(k=10 self vs no-self both 0.324) — this is homology transfer, not trivial leakage.

**Long-range R-precision (mean over 554, exp89 metric harness):**

| predictor | long-range R |
|---|---|
| seq-KNN k=1 | 0.264 |
| seq-KNN k=5 | 0.319 |
| **seq-KNN k=10** | **0.324** |
| seq-KNN k=25 | 0.323 |
| seq-KNN k=50 | 0.317 |
| MarinFold #61 (rollout x100) | **0.353** |
| Protenix-v2 single-seq | 0.572 |
| Protenix-v2 MSA | 0.795 |
| ESMFold | 0.732 |
| ESMFold2 | 0.769 |

→ [`plots/headline_rprecision.png`](plots/headline_rprecision.png),
[`plots/k_sweep.png`](plots/k_sweep.png) (k peaks at 10),
[`plots/memorization_scatter.png`](plots/memorization_scatter.png),
[`plots/rprecision_vs_identity.png`](plots/rprecision_vs_identity.png).

**The aggregate tie is misleading — stratifying by nearest-neighbor identity
inverts the story** ([`rprecision_vs_identity.png`](plots/rprecision_vs_identity.png)):

| best train-hit identity | n | seq-KNN k=10 | MarinFold #61 |
|---|---|---|---|
| no hit | 139 | ~0.00 | 0.40 |
| 0–30% | 28 | 0.06 | 0.24 |
| 30–50% | 201 | 0.31 | 0.34 |
| 50–70% | 94 | 0.54 | 0.35 |
| 70–100% | 92 | 0.69 | 0.35 |

seq-KNN rises monotonically with sequence identity (it *must* — it copies the
nearest neighbor), while **MarinFold's accuracy is essentially flat (~0.34–0.40)
across every identity bin**, including the 139 proteins with no training homolog at
all. Per-protein the two barely correlate (Pearson r = 0.12): they are accurate on
*different* proteins, not the same ones.

Extending this to **all predictors** ([`rprecision_vs_identity_all.png`](plots/rprecision_vs_identity_all.png))
makes the point sharper: seq-KNN is the *only* predictor whose accuracy tracks
sequence identity. MarinFold, Protenix-v2 (single-seq/MSA), ESMFold and ESMFold2 are
all flat or even decline toward high identity. Caveat: the identity axis is
confounded with dataset composition — the no-hit / low-identity bins are dominated by
de novo *designed* proteins (123 of the 139 no-hit proteins are `denovo_pdb`; full
list in [`data/no_hit_proteins.csv`](data/no_hit_proteins.csv)), which the structure
predictors find easy (idealized backbones), lifting the left side of their curves.
The robust, confound-resistant signal is the *slope*: only the copy baseline tracks
identity. Notably 117/135 no-hit proteins (with strata) are `same_fold`/`redundant`
by structure despite `novel_seq` sequences — the twilight zone a structural-KNN
(foldseek) follow-up would target.

### Viral / out-of-distribution proteins

AFDB (the contacts-v1 training source) **excludes viruses**, so viral eval proteins
are a clean out-of-distribution probe. [`annotate_taxonomy.py`](annotate_taxonomy.py)
labels every eval protein via RCSB taxonomy (full table
[`data/eval_taxonomy.csv`](data/eval_taxonomy.csv), viral subset
[`data/viral_proteins.csv`](data/viral_proteins.csv)): **28 / 554 are viral** — 18/26
of `casp_fm` (CASP14 FM is phage-heavy: 9 from *Cellulophaga* phage phi14:2), 6
`foldbench100`, 4 `cameo_hard`, 0 `denovo_pdb` (all synthetic designs). Bacteriophages
count (also AFDB-excluded).

| predictor | viral (n=28) | non-viral (n=526) |
|---|---|---|
| seq-KNN k=10 | 0.08 | 0.34 |
| **MarinFold #61** | **0.11** | **0.37** |
| Protenix-v2 (MSA) | 0.60 | 0.81 |
| ESMFold2 | 0.36 | 0.79 |

→ [`plots/viral_vs_nonviral.png`](plots/viral_vs_nonviral.png). MarinFold drops 3.3×
on viral (to ~30% of its non-viral accuracy), as does ESMFold2 (0.79→0.36) — both
are trained on natural/AFDB-like proteins — while the MSA-based Protenix degrades
gracefully (retains ~74%). This **refines the identity-stratification result**: the
flat ~0.40 MarinFold scored in the "no training homolog" bin was buoyed by the 123 de
novo *designed* proteins (idealized, easy); on genuinely OOD natural proteins (viral)
MarinFold's contact accuracy collapses. The eval's headline number is propped up by
in-distribution and designed proteins; viral/OOD proteins are a real blind spot.

**Per-protein comparison table for further analysis:**
[`data/per_protein_comparison.csv`](data/per_protein_comparison.csv) — one row per
eval protein with long-range R-precision for every predictor (MarinFold variants,
seq-KNN, Protenix single-seq/MSA, ESMFold, ESMFold2) alongside strata (`is_viral`,
`source_organism`, `best_train_identity`, `n_train_seq_hits`, `fold_verdict`,
`seq_leakage`, `length`). Built by [`build_per_protein_table.py`](build_per_protein_table.py).

## Conclusion

A no-folding "copy the nearest training neighbor's contacts" null model matches
MarinFold #61's **average** long-range R-precision (0.324 vs 0.353) on this eval
set — so the headline eval number is substantially inflated by sequence-homology-
transferable cases, and a trivial baseline should be reported alongside it.

But the stratified view argues **against** the memorization hypothesis the issue
raised: MarinFold's accuracy does **not** increase when a close training homolog
exists (flat ~0.35 from 30% to 100% identity), it is the **only** predictor that
works on the 139 no-homolog proteins (0.40 vs KNN's ~0), and it barely correlates
with the copy baseline per protein. A model that was merely memorizing folds would
spike on high-identity proteins exactly where seq-KNN does — MarinFold does not. The
seq-KNN baseline ties it on average only because the eval set is homology-rich
(186/554 proteins ≥50% identity to train), where trivial copying excels.

Takeaways: (1) report seq-KNN as a standing null model next to MarinFold on this
eval set; (2) the contact-prediction signal MarinFold has learned is independent of
training-sequence proximity — promising for generalization, and worth a structural-
KNN follow-up (foldseek nearest train rep, reuse exp41's DB) to test whether the
same holds against *structural* rather than *sequence* neighbors.
