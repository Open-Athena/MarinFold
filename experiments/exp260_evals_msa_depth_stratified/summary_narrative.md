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
recipe, zero unfinished. The scoring worker and #89's metric script are
byte-identical to PR #257's, and the checkpoint is read in place from the export
that PR wrote.

Depth comes off the two Modal volumes of ColabFold a3m files that Protenix's
`+MSA` arm already ran with, measured through #74's `msa_depth.py` — raw
sequence count and Neff, one code path for both volumes.

## Validation

PR #257's published legacy-554, `eval-val`, and `eval-denovo` aggregates are the
gate: same weights, same worker, so a disagreement would mean the path moved,
not the model. All twelve reproduced, largest difference 0.0044 — inside the
0.005 tolerance the eval-checkpoint recipe fixes and consistent with #204's
0.0023 run-to-run span.

## Results — the eval sets

_Filled from `data/coreweave_results/results/subset_aggregate_metrics.csv`._

## Results — MSA depth

_Filled from `data/depth_tiers.csv`._

## Caveats

_Filled once the bin sizes are known._
