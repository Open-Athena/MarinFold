# exp260 — contact accuracy vs. MSA depth

## The question

MarinFold predicts contacts from a single sequence. The regime where that has
something to prove is the one where MSA-based methods run out of signal:
proteins with few homologs.

PR #257 produced our best decontaminated checkpoint (#232 `m2-p06` training,
step 363,000) and scored it on `eval-val`, `eval-denovo`, and the legacy 554.
This experiment finishes the read — `eval-test`, previously unscored for this
checkpoint — and then bins every natural protein in the eval universe by the
depth of the ColabFold MSA an MSA-based competitor would have had.

## What ran

887 evaluation units on twelve single-H100 CoreWeave shards: the legacy 554 plus
all 333 scorable FoldBench monomers. 100 rollouts per unit under the fixed #82
recipe, 88,700 usable, zero unfinished, 9m41s.

Depth comes off the two Modal volumes of ColabFold a3m files that Protenix's
`+MSA` arm already ran with, measured through #74's `msa_depth.py` — raw
sequence count and Neff, one code path for both volumes.

## Validation

PR #257's published legacy-554, `eval-val`, and `eval-denovo` aggregates are the
gate: same weights, same worker, so a disagreement would mean the path moved,
not the model. All twelve reproduced, largest difference 0.0044.

The depth measurements match #247's independent count on all 314 FoldBench
natural proteins exactly. Of the 11 stems both Modal volumes hold, 9 agree
exactly and all 11 land in the same tier, so the three-week gap between the two
ColabFold runs does not move the stratification.

## The eval sets

| subset | n | R (all) | R (long) |
|---|---:|---:|---:|
| **eval-test** (first read for this checkpoint) | 217 | **0.5693** | 0.5464 |
| eval-val | 97 | 0.5561 | 0.5402 |
| eval-denovo | 19 | 0.6110 | 0.5745 |
| legacy 554 | 554 | 0.6059 | 0.5566 |

+0.032 on eval-test over the #232 sweep checkpoint, and 0.144 clear of the
seq-KNN null over its own decontaminated corpus. Protenix-v2 + MSA scores 0.8446
on the same 217 proteins.

## First: 15 "natural" proteins are designs

The CAMEO-hard / CASP-FM half was labelled natural by provenance, not by
checking. RCSB says 15 of those 58 are de novo designs — and 13 of the 15 sit
under MSA depth 10, because a designed protein has no homologs by construction.

They had to come out. Designed backbones are easy for structure predictors:
in the shallow bin Protenix-v2 *single-sequence* scores 0.722 on the 13 designs
against 0.241 on the 16 natural proteins. Pooling them produced a materially
wrong conclusion in the first cut of this experiment — the same failure #241
found in eval2-natural. The natural universe is 357, not 372.

## MSA depth — the corrected picture

Mean all-range R-precision, 357 natural proteins:

| depth | n | MarinFold | Protenix-v2 + MSA | ESMFold2 | Protenix-v2 single-seq |
|---|---:|---:|---:|---:|---:|
| <10 | 16 | 0.300 | 0.336 | 0.426 | 0.241 |
| 10–100 | 32 | 0.279 | 0.755 | 0.469 | 0.273 |
| 100–1000 | 76 | 0.413 | 0.819 | 0.671 | 0.278 |
| ≥1000 | 233 | 0.616 | 0.858 | 0.827 | 0.249 |

MarinFold degrades with depth like everything else — depth proxies how well a
family is represented anywhere, AFDB included.

## What survives — the gap closes where the MSA goes

Paired per-protein against Protenix-v2 + MSA, 95 % bootstrap:

| depth | n | MarinFold − (+MSA) |
|---|---:|---|
| <10 | 16 | **−0.036 [−0.180, +0.104]** |
| 10–100 | 32 | −0.477 [−0.545, −0.397] |
| 100–1000 | 76 | −0.407 [−0.454, −0.360] |
| ≥1000 | 233 | −0.242 [−0.263, −0.220] |

At depth under 10 MarinFold is statistically level with the MSA-based model.
The gap closes mostly because Protenix falls (0.858 → 0.336), which is what the
single-sequence thesis predicts. It never leads: ESMFold2 is ahead in every
tier, including this one (−0.126 paired).

Supporting checks: Protenix-v2 + MSA collapses toward its own single-sequence
arm in this bin (0.336 vs 0.241), which is what an empty alignment should do;
and the depths reproduce #247's independent count exactly on all 314 FoldBench
naturals.

## The low-MSA-depth cut — now a standing report

29 proteins under depth 10, with a `designed` column: 16 natural (11
CAMEO/CASP + 5 FoldBench) and 13 CAMEO designs. Frozen as
`data/low_msa_depth_set.csv`, required by the `eval-checkpoint` skill, and
browsable case by case in `dashboard/index.html`.

The CAMEO and CASP targets are generally inside the baselines' training sets, so
the only like-for-like comparison in this regime is the 5 FoldBench members.
Growing that is the follow-up this experiment argues for.

## Caveats

- 16 natural proteins under depth 10, only 5 of them uncontaminated for the
  baselines, is too thin to carry the headline on its own.
- Median length is 148 residues in the <10 bin against 290 at ≥1000, and contact
  prevalence goes as ~1/L. That bias favours the shallow bins, so the reported
  decline is conservative.
