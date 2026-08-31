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

## MSA depth — accuracy does not hold up

Mean all-range R-precision, 372 natural proteins:

| depth | n | MarinFold | Protenix-v2 + MSA | ESMFold2 | Protenix-v2 single-seq |
|---|---:|---:|---:|---:|---:|
| <10 | 29 | 0.379 | 0.510 | 0.556 | 0.457 |
| 10–100 | 33 | 0.281 | 0.758 | 0.480 | 0.291 |
| 100–1000 | 77 | 0.416 | 0.817 | 0.671 | 0.283 |
| ≥1000 | 233 | 0.616 | 0.858 | 0.827 | 0.249 |

MarinFold degrades with MSA depth much as the MSA methods do — depth is a proxy
for how well a family is represented anywhere, including in the AFDB corpus it
trained on. Paired against Protenix-v2 single-seq, MarinFold's whole advantage
is a deep-MSA phenomenon: +0.367 at ≥1000, and indistinguishable from zero in
both shallow bins.

## The low-MSA-depth cut — now a standing report

29 natural proteins with a ColabFold MSA under 10 sequences: 16 CAMEO-hard, 8
CASP-FM, 5 FoldBench (all in eval-test). Frozen as
`data/low_msa_depth_set.csv` and now a required reporting row in the
`eval-checkpoint` skill.

| predictor | the 29 | FoldBench-only (5) | all natural (372) |
|---|---:|---:|---:|
| **MarinFold #232 m2-p06 training** | **0.379** | **0.342** | 0.527 |
| Protenix-v2 + MSA | 0.510 | 0.320 | 0.813 |
| Protenix-v2 single-seq | 0.457 | 0.305 | 0.276 |
| ESMFold2 | 0.556 | 0.664 | 0.743 |

Report both rows: paired against Protenix-v2 + MSA, MarinFold is −0.131 [−0.232,
−0.030] on the 29 and +0.022 [−0.162, +0.206] on the 5. The FoldBench subset
alone turns a clear loss into an apparent tie.

That these are genuinely MSA-poor and not mis-measured: Protenix-v2 + MSA
collapses toward its own single-sequence arm here (0.510 vs 0.457, against 0.813
vs 0.276 overall).

## What survives

The **gap to Protenix-v2 + MSA is narrowest where MSAs are thinnest** — −0.131
[−0.232, −0.030] at depth <10 against −0.242 at ≥1000. It never closes, and the
narrowing is mostly Protenix falling rather than MarinFold rising. On the 29,
MarinFold is behind every structure predictor measured and ahead only of the
memorisation null.

AUC is deliberately not used for any cross-predictor claim here: #89 scores
structure predictors from a degree matrix where unpredicted pairs are exactly 0,
so ~99 % of candidate pairs tie at the bottom and roc_auc_score gives each tie
half credit — it measures output sparsity as much as ranking quality.

## Caveats

- The FoldBench half has only 5 proteins under depth 10; the pooled row is the
  one to read, and it exists because the 58 CAMEO-hard / CASP-FM targets (24 of
  them under depth 10) were brought in.
- Median length is 148 residues in the <10 bin against 290 at ≥1000, and contact
  prevalence goes as ~1/L. That bias favours the shallow bins, so the reported
  decline is conservative.
