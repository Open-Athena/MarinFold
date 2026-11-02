# Plan: distill an ESM Atlas subset to augment our AFDB training set

## Goal
Add a quality-filtered, novel, de-duplicated subset of the **ESM Atlas** (1.1B
ESMFold2 predicted monomers; AWS `s3://esm-protein-atlas`, us-west-2) to our
existing **~10M AFDB** training set for the new structure model.

## Key design decision: a sequence + metadata pipeline (no large-scale structural compute)
- pLDDT / pTM are already structure-quality signals and ship as **metadata columns**;
  novelty, leakage, and clustering are all **sequence** ops (MMseqs2). So the whole
  filter + cluster runs without reading the multi-TB `structure_blob` column.
- We **decode structures only for the final selected reps** (materialization), not the 1.1B.
- We **drop the structural TM-dedup vs AFDB** on merit: once a sequence is <40%
  identical to AFDB, a known-fold match is "new sequence on a known fold" — the most
  valuable distillation signal (AF2's same-fold bucket), not redundancy.

## Funnel (per structure; all CLI-tunable)
1. length 60–1000 (Atlas native)
2. mean pLDDT > 0.7
3. pTM > 0.5
4. sequence identity < 40% vs our AFDB training sequences (novelty; MMseqs)
5. eval-leakage: drop ≥40% identity to the exp65 held-out eval set (MMseqs)
6. *(deferred QC, on selected reps only)* globularity (Cα contact ratio > 0.5) +
   per-residue pLDDT std (uniform-confidence)

Sample estimate (2k-structure sample, ±~2% sampling CI): ~**32% of 1.1B ≈ ~350M**
survivors before clustering.

## Cluster + select
- **MMseqs2 linclust** survivors at **40% identity** (the Atlas is already 70%-id
  clustered, so only <70% actually collapses).
- **One rep per cluster**, chosen by the ESMFold2 rule (longest, then lowest
  per-residue pLDDT std). Size dial = identity threshold + reps/cluster; target
  10–100M vs the 10M AFDB set.
- Note: we can't reuse the Atlas's published SAE/Pfam clusters — they only cover the
  ≥50-member popular families (~12% of the data) and discard the novel tail.

## Compute / logistics
- Runs in **AWS us-west-2**, co-located with the Atlas (no egress). Reads only small
  columns; CPU MMseqs. Automated via an **EC2 launch script** (provision → run
  pipeline → write selected-hash manifest → tear down).
- Materialize selected reps' structures → CIF / training docs → publish to
  **HuggingFace** (`open-athena/MarinFold`).

## Open items
- License: Atlas is **CC BY-SA 4.0** (ShareAlike) → a republished subset must also be
  BY-SA. Deferred, not blocking.
- Final size target (10M / 30M / 100M?).
- Confirm novelty reference = our `afdb-24M` training sequences.

## Artifacts (branch `exp/91-evals-esmfold2-atlas`)
`funnel.py` (filter), `selection.py` (cluster-rep select), `atlas_io.py` (Atlas
reader), distillation-methods survey in the README; EC2 automation script in progress.
