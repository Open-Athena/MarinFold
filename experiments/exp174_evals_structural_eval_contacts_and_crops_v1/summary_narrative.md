# Summary slides — exp174: 3D coordinate-based structural eval for contacts-and-crops-v1

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Every MarinFold eval so far scores contact prediction. contacts-and-crops-v1
(#130) encodes real 3D positions — Pass-1 coarse 10 A boxes plus Pass-2 fine
0.1 A crops — so a model trained on it should be able to emit an actual
structure.

This experiment asks whether it can, scored the way structure predictors are
normally scored: CA-RMSD, all-atom RMSD, lDDT, lDDT-CA and TM-score against the
554-protein eval set from #74/#78.

## Two components, two gates

Component 1 — inference (document to coordinates) — is plans-first by design.
Six approaches are written up in PLANS.md. Plan F is now the plan of record:
neighbour-conditioned iterative refinement. Not yet implemented.

Component 2 — scoring — is done. It takes a directory of predicted coordinate
files in a documented one-PDB-per-protein contract and needs to know nothing
about how they were produced. Ground truth, model predictions and baselines all
speak the same contract.

## Ground truth: 554/554, and one library bug

Full-atom ground truth for all 554 records: 777,459 heavy atoms, zero failures,
minimum sequence-alignment identity 0.95, lengths 30 to 761.

Building it surfaced a real bug in the shared library. The coordinate parse
layer joins the pyconfind residue list to the gemmi atom walk on
(chain, residue number), and the two libraries spell a blank author chain id
differently — gemmi "", pyconfind "_". Structures with no chain id joined
nothing and produced a document with an empty coordinate section, silently.
AFDB is always chain A so the training corpus never hit it; 19 of the 554 eval
proteins did. Fixed, with the parse layer now raising instead of emitting a
coordinate-free document.

## Partial predictions are the normal case

About 96% of contacts-and-crops-v1 documents are truncated by construction, so
the harness reports two families of metric and keeps them apart.

Coverage-penalized (lDDT, lDDT-CA, TM-score) take their denominators from the
ground truth, so an atom the model never placed costs score. These are the
model-comparison numbers.

Covered-only (RMSD, and the "covered" lDDT variants) are computed over the
atoms that were placed. A predictor that emits three atoms perfectly scores
0.0 A RMSD, so these are meaningless without the coverage columns beside them.

A record with no prediction file is scored as a total miss, not skipped.

## The headline so far: the format's ceiling

Before any model number means anything, we measured what a perfect model could
score. Degrade the ground truth to each of the format's resolution tiers and
run it through the same harness.

The 0.1 A digit vocabulary costs nothing: lDDT 1.000, TM 1.000. But every atom
at its correct 10 A box centre scores only lDDT 0.32 / TM 0.51, and one
realistic document — 65% of atoms boxed, 25% refined — tops out at lDDT 0.17 /
TM 0.41.

lDDT falls as coverage squared (a contact needs both its atoms); TM-score falls
linearly (a residue needs only itself).

## What that implies for inference

An inference plan that samples one document per protein is competing for lDDT
0.17. A result from such a plan cannot distinguish "the model cannot fold" from
"the format did not get a chance", which is the central argument in PLANS.md
for spending inference compute to raise the refined fraction.

Inference cost is not the constraint: a full 8192-token document for all 554
proteins is about three GPU-minutes on one H100.

The Pass-2 refined fraction is the entire ballgame. Going from 15% to 50%
refined moves the achievable lDDT from 0.36 to 0.53 and TM from 0.58 to 0.75.
That is a format finding as much as an inference one.

## The plan of record: a scanning flashlight

Plan F. Generate Pass 1 once, which fixes the frame. Then sweep a spatially
coherent path over the occupied voxels: for each one, build a prompt ending in
its crop header, draw K sampled crop bodies, and fold them into a running
precision-weighted per-atom estimate. Repeat the sweep, because a voxel's
neighbours have moved since it was last visited, until the coordinates stop
moving.

Two things make this fit the format rather than fight it. Each crop is
conditioned on its already-refined neighbours, and Pass-2's own box selection
is 45% frontier and 10% re-show, so a local scan with revisits is the
training-time crop distribution, not a departure from it. And the prompt —
sequence, full Pass 1, up to about twenty prior crops, one new header — is
exactly the shape of a real training document, because the fine reserve holds
about twenty crops.

Sampling, not greedy: averaging K draws recovers the same estimate and hands
back a per-atom variance for free, which is what the B-factor column and the
precision weighting both want. Two independent temperatures, one for
coordinate tokens and one for structural choices.

Cost: about 1.5 to 2 H100-hours for all 554 proteins, provided the synthesized
Pass-1 section is held byte-identical within a sweep so the prefix cache hits.
That one decision is worth 10x.

## Status

Harness built, tested (34 tests) and validated against a measured ceiling.
Inference approach agreed: Plan F. Remaining work is E2 and E3 as gates, A and
C as controls, then F itself, then scoring both #137/#155 checkpoints — which
are staged and ready.
