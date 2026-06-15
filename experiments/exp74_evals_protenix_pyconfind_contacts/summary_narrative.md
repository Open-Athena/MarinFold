# Summary slides — Protenix vs pyconfind contact eval (issue #74)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Evaluate **Protenix v2 at contact prediction**, where a "contact" is a
pyconfind side-chain contact (the same definition our contacts_v1 training
documents use). We score Protenix in **four configs** = {single-seq, msa} ×
{distogram, structure} on two eval sets: the FoldBench-100 monomers from
exp12 (reusing its outputs) and all 454 low-MSA / structurally-novel
candidates from exp65.

## Ground truth & metric

Ground truth: pyconfind, `native_only=True`, contact degree ≥ 0.001,
primary-sequence separation ≥ 6 (identical to contacts_v1). Predictors:
- **distogram** — rank pairs by P(rep atoms within 8 Å) from the distogram.
- **structure** — run pyconfind on the predicted structure, rank by degree.

Metric: **contacts @ L** (precision among top-L ranked pairs; also L/2, L/5),
aggregate (sep ≥ 6) and split short / medium / long. Candidate pairs are
restricted to GT-resolved residues, identically across configs.

## Why the two predictors differ

The **structure** predictor uses pyconfind on the prediction — the *same*
side-chain-contact notion as the ground truth, so it's apples-to-apples.
The **distogram** predictor ranks by a CB-CB ≤ 8 Å distance notion against a
side-chain-contact ground truth, so it carries an inherent representation
gap. Both are what the issue asked for; expect structure ≥ distogram.

## Validation (FoldBench-100 subset)

On a 12-protein size-spanning subset: alignment identity 1.000 everywhere;
MSA ≫ single-seq for both predictors (structure aggregate: MSA 0.76 vs SS
0.22); structure > distogram in MSA mode (0.76 vs 0.42). The one failure
(8eb9_A, ~baseline across configs) is independently a Protenix failure in
exp12 (msa RMSD_CA 8.8 Å), so the eval faithfully reflects model quality,
not a scoring bug.

## Headline result

Scored all 554 proteins (FoldBench-100 + 454 exp65), 4 configs, alignment
identity ≥ 0.95 everywhere. The **structure** predictor beats the
**distogram** predictor throughout (it shares pyconfind's contact notion).
On FoldBench-100 (natural, deep-MSA) MSA ≫ single-seq (structure agg
contacts @ L 0.77 vs 0.25); on exp65 (low-MSA / novel) the gap **collapses**
(0.70 vs 0.58).

## MSA advantage scales with MSA depth

exp65 long-range structure contacts @ L by Neff tier (MSA vs single-seq):
orphan 0.41/0.42 (gap −0.01), low 0.45/0.43, marginal 0.50/0.38, deep
0.45/0.30 (gap +0.16). **For orphan/low-Neff proteins single-sequence is on
par with MSA mode** — the regime that matters for a single-sequence model.
Novel folds are the hardest stratum (long structure @ L: novel 0.39/0.28).

## Where the data lives

`data/contact_precision_all.csv` (tidy scores), `plots/` (contacts @ L by
config/range + the stratifications), and the HF bucket
`data/protenix-contacts-eval-exp74/` (raw Protenix outputs + the full
pyconfind contact tables, 696k contacts). See the README for the full
reproduction spec.
